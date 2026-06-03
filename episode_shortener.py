#!/usr/bin/env python3
"""Episode shortener.

Detect *speech* in a video and cut out everything else — music, ambience and
the long filler scenes where nobody is talking. Around every detected speech
region we keep a configurable buffer of ``x`` seconds before the speech and
``y`` seconds after it; everything outside those padded speech regions is
dropped. The result is a tighter cut that keeps only the dialogue (plus a
little breathing room) and removes the dead air that makes Turkish series drag
on.

Usage:
    python3 episode_shortener.py input.mkv -o output.mp4 -x 0.5 -y 1.0

The input may also be a *folder*, in which case every video inside it is
processed into a ``<name>_processed.mp4`` sibling (folder mode). Pass
``-log <file>`` to append a detailed report (filenames, codecs, lengths, kept
and removed time-ranges) for every processed file, and ``--ignore-errors`` to
push through corrupted/damaged streams instead of aborting.

Requires ``ffmpeg``/``ffprobe`` on PATH. The default ``silero`` detector also
needs the ``onnxruntime`` and ``numpy`` Python packages plus the bundled
``silero_vad.onnx`` model file.
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request

# Video file extensions recognised in folder mode.
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".ts", ".webm", ".flv",
    ".wmv", ".mpg", ".mpeg", ".m4v", ".3gp", ".m2ts", ".mts",
}

# Default location of the bundled Silero VAD model (sits next to this script).
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "silero_vad.onnx")
SILERO_MODEL_URL = ("https://github.com/snakers4/silero-vad/raw/master/"
                    "src/silero_vad/data/silero_vad.onnx")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def tolerant_input_flags(ignore_errors):
    """ffmpeg input flags that make it tolerate corrupted/damaged streams.

    Placed before ``-i``: ``-err_detect ignore_err`` keeps decoding past
    errors, ``+discardcorrupt`` drops corrupt packets, and ``+genpts``
    regenerates timestamps so the cut still lines up afterwards.
    """
    if not ignore_errors:
        return []
    return ["-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts"]


def require_tool(name):
    """Exit with a friendly message if an external tool is missing."""
    if shutil.which(name) is None:
        sys.exit(
            f"error: '{name}' was not found on PATH. Please install ffmpeg "
            f"(which provides ffmpeg and ffprobe) and try again."
        )


def format_duration(seconds):
    """Render a number of seconds as H:MM:SS.s for human-friendly output."""
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours:
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes}:{secs:05.2f}"


def get_duration(path, ignore_errors=False):
    """Return the duration of a media file in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            *tolerant_input_flags(ignore_errors),
            "-show_entries", "format=duration",
            "-of", "json", path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"error: ffprobe failed to read '{path}':\n{result.stderr.strip()}")
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        sys.exit(f"error: could not determine duration of '{path}'.")


# --------------------------------------------------------------------------- #
# Speech detection backends
# --------------------------------------------------------------------------- #
def _ensure_silero_model(model_path):
    """Return a usable model path, downloading the bundled model if missing."""
    if os.path.isfile(model_path):
        return model_path
    print(f"Silero model not found at '{model_path}', downloading...")
    try:
        urllib.request.urlretrieve(SILERO_MODEL_URL, model_path)
    except Exception as exc:  # network/permission issues
        sys.exit(
            f"error: could not download the Silero VAD model: {exc}\n"
            f"       Download it manually from {SILERO_MODEL_URL}\n"
            f"       and save it to {model_path}, or pass --model <path>."
        )
    return model_path


def detect_speech_silero(path, threshold, min_silence, min_speech, model_path,
                         ignore_errors=False):
    """Detect speech with the Silero VAD neural model (the default).

    Silero is trained specifically to find *human voice*, so it reliably
    rejects music, sound effects and ambience instead of treating any loud
    audio as speech. Returns a list of (start, end) speech intervals.
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        sys.exit(
            "error: the 'silero' detector needs numpy and onnxruntime.\n"
            "       Install them with:  pip install -r requirements.txt\n"
            "       or pick another detector with:  --detector vad|silence"
        )

    model_path = _ensure_silero_model(model_path)
    sample_rate = 16000
    window = 512          # required window size for 16 kHz in Silero v5
    context_size = 64     # samples of previous audio prepended to each window

    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error",
            *tolerant_input_flags(ignore_errors),
            "-i", path,
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
        ],
        capture_output=True,
    )
    if proc.returncode != 0 and not ignore_errors:
        sys.exit(f"error: ffmpeg failed to extract audio:\n{proc.stderr.decode(errors='replace').strip()}")

    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    session = ort.InferenceSession(model_path,
                                   providers=["CPUExecutionProvider"])
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, context_size), dtype=np.float32)
    sr_arg = np.array(sample_rate, dtype=np.int64)

    # Run the model over consecutive windows, carrying state + context forward.
    flags = []
    for offset in range(0, len(audio) - window + 1, window):
        chunk = audio[offset:offset + window][None, :]
        model_input = np.concatenate([context, chunk], axis=1)
        prob, state = session.run(
            None, {"input": model_input, "state": state, "sr": sr_arg}
        )
        context = chunk[:, -context_size:]
        flags.append(float(prob[0, 0]) >= threshold)

    frame_dur = window / sample_rate
    return _frames_to_segments(flags, frame_dur, min_silence, min_speech)


def detect_speech_vad(path, aggressiveness, frame_ms, min_silence, min_speech,
                      ignore_errors=False):
    """Detect speech with WebRTC voice-activity detection.

    A lightweight alternative to the Silero detector. Better than plain volume
    thresholding, but weaker at rejecting music — it can mistake music for
    voice. Prefer ``--detector silero`` when you want speech only.
    Returns a list of (start, end) speech intervals in seconds.
    """
    try:
        import webrtcvad
    except ImportError:
        sys.exit(
            "error: the 'vad' detector needs the webrtcvad package.\n"
            "       Install it with:  pip install webrtcvad-wheels\n"
            "       or pick the silence-based detector with:  --detector silence"
        )

    sample_rate = 16000  # webrtcvad supports 8k/16k/32k/48k; 16k is plenty for voice.
    if frame_ms not in (10, 20, 30):
        sys.exit("error: --frame-ms must be 10, 20, or 30 (WebRTC VAD requirement).")
    bytes_per_frame = int(sample_rate * (frame_ms / 1000.0)) * 2  # 16-bit mono

    # Decode the whole soundtrack to raw 16 kHz mono PCM and stream it in.
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error",
            *tolerant_input_flags(ignore_errors),
            "-i", path,
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
        ],
        capture_output=True,
    )
    if proc.returncode != 0 and not ignore_errors:
        sys.exit(f"error: ffmpeg failed to extract audio:\n{proc.stderr.decode(errors='replace').strip()}")

    pcm = proc.stdout
    vad = webrtcvad.Vad(aggressiveness)

    # Classify every frame as speech / not-speech.
    flags = []
    for offset in range(0, len(pcm) - bytes_per_frame + 1, bytes_per_frame):
        frame = pcm[offset:offset + bytes_per_frame]
        flags.append(vad.is_speech(frame, sample_rate))

    frame_dur = frame_ms / 1000.0
    return _frames_to_segments(flags, frame_dur, min_silence, min_speech)


def _frames_to_segments(flags, frame_dur, min_silence, min_speech):
    """Turn per-frame speech booleans into clean (start, end) segments.

    Gaps shorter than ``min_silence`` are bridged (so a brief pause mid-sentence
    doesn't split a line in two), and bursts shorter than ``min_speech`` are
    discarded (so a single noisy frame isn't treated as dialogue).
    """
    # Group consecutive speech frames into raw segments.
    raw = []
    start = None
    for i, is_speech in enumerate(flags):
        if is_speech and start is None:
            start = i
        elif not is_speech and start is not None:
            raw.append((start * frame_dur, i * frame_dur))
            start = None
    if start is not None:
        raw.append((start * frame_dur, len(flags) * frame_dur))

    # Bridge short silences between segments.
    bridged = []
    for seg in raw:
        if bridged and seg[0] - bridged[-1][1] <= min_silence:
            bridged[-1] = (bridged[-1][0], seg[1])
        else:
            bridged.append(seg)

    # Drop segments that are too short to be real speech.
    return [(s, e) for s, e in bridged if (e - s) >= min_speech]


def detect_speech_silence(path, noise_db, min_silence, duration,
                          ignore_errors=False):
    """Detect speech as 'whatever isn't silence', using ffmpeg's silencedetect.

    Fast and dependency-free, but treats any audio above the threshold
    (including music) as speech. Returns a list of (start, end) intervals.
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "info",
            *tolerant_input_flags(ignore_errors),
            "-i", path, "-vn",
            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    # silencedetect prints to stderr regardless of success.
    silences = []
    pending_start = None
    for line in proc.stderr.splitlines():
        if "silence_start:" in line:
            pending_start = float(line.split("silence_start:")[1].split("|")[0].strip())
        elif "silence_end:" in line:
            end = float(line.split("silence_end:")[1].split("|")[0].strip())
            start = pending_start if pending_start is not None else 0.0
            silences.append((start, end))
            pending_start = None
    if pending_start is not None:  # silence runs to the end of the file
        silences.append((pending_start, duration))

    # Speech = the complement of the silence intervals over [0, duration].
    speech = []
    cursor = 0.0
    for s, e in silences:
        if s > cursor:
            speech.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < duration:
        speech.append((cursor, duration))
    return speech


# --------------------------------------------------------------------------- #
# Segment maths
# --------------------------------------------------------------------------- #
def pad_and_merge(segments, pad_before, pad_after, duration):
    """Expand each speech segment by the buffers and merge any overlaps."""
    if not segments:
        return []
    padded = sorted(
        (max(0.0, s - pad_before), min(duration, e + pad_after))
        for s, e in segments
    )
    merged = [list(padded[0])]
    for s, e in padded[1:]:
        if s <= merged[-1][1]:           # overlapping or touching -> merge
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def total_length(segments):
    return sum(e - s for s, e in segments)


def complement(segments, duration):
    """Return the gaps between (sorted) kept segments over [0, duration]."""
    gaps = []
    cursor = 0.0
    for s, e in segments:
        if s > cursor:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < duration:
        gaps.append((cursor, duration))
    return gaps


# --------------------------------------------------------------------------- #
# Cutting
# --------------------------------------------------------------------------- #
def cut_and_concat(input_path, output_path, segments, video_codec, audio_codec,
                   crf, ignore_errors=False):
    """Keep only ``segments`` and concatenate them into a single output file.

    Uses an ffmpeg filtergraph (trim + concat) so cuts are frame-accurate
    regardless of keyframe placement. The filtergraph is written to a script
    file to stay clear of command-line length limits when there are many cuts.
    """
    parts = []
    for i, (s, e) in enumerate(segments):
        parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}];")
        parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}];")
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(segments)))
    parts.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[outv][outa]")
    filtergraph = "\n".join(parts)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(filtergraph)
        script_path = fh.name

    try:
        cmd = [
            "ffmpeg", "-v", "error", "-stats", "-y",
            *tolerant_input_flags(ignore_errors),
            "-i", input_path,
            "-filter_complex_script", script_path,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", video_codec, "-crf", str(crf),
            "-c:a", audio_codec,
            output_path,
        ]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            # With --ignore-errors, tolerate a non-zero exit as long as ffmpeg
            # still produced a non-empty output file (corrupt packets dropped).
            produced = (os.path.isfile(output_path)
                        and os.path.getsize(output_path) > 0)
            if ignore_errors and produced:
                print("  warning: ffmpeg reported errors but produced output "
                      "(--ignore-errors); continuing.")
            else:
                sys.exit("error: ffmpeg failed while cutting the video.")
    finally:
        os.unlink(script_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Shorten a video by keeping only speech (plus a buffer) and "
                    "removing music, ambience and silent filler scenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",
                   help="path to a video file, OR a folder to process every "
                        "video inside it (folder mode)")
    p.add_argument("-o", "--output",
                   help="path to the output video (single-file mode only; "
                        "default: <input>_short.mp4). Ignored in folder mode, "
                        "where outputs are named <name>_processed.mp4")
    p.add_argument("-x", "--pad-before", "-before", "--before",
                   dest="pad_before", type=float, default=0.5,
                   help="seconds of video to KEEP before each speech region "
                        "(everything else before it is cut)")
    p.add_argument("-y", "--pad-after", "-after", "--after",
                   dest="pad_after", type=float, default=1.0,
                   help="seconds of video to KEEP after each speech region "
                        "(everything else after it is cut)")

    p.add_argument("--detector", choices=["silero", "vad", "silence"],
                   default="silero",
                   help="speech detector: 'silero' is a neural model that keeps "
                        "speech ONLY and rejects music/noise (recommended); "
                        "'vad' is a lighter WebRTC voice detector; 'silence' "
                        "keeps any non-quiet audio (music included)")

    # Silero detector options.
    p.add_argument("--threshold", type=float, default=0.5,
                   help="[silero] speech probability cutoff, 0..1 "
                        "(raise toward 1 to be stricter and drop more music)")
    p.add_argument("--model", default=DEFAULT_MODEL_PATH,
                   help="[silero] path to the silero_vad.onnx model file")

    # WebRTC VAD detector options.
    p.add_argument("--aggressiveness", type=int, choices=[0, 1, 2, 3], default=2,
                   help="[vad] 0=lenient .. 3=aggressive at rejecting non-speech")
    p.add_argument("--frame-ms", type=int, choices=[10, 20, 30], default=30,
                   help="[vad] analysis frame size in milliseconds")

    # Silence detector options.
    p.add_argument("--noise-db", type=float, default=-30.0,
                   help="[silence] audio below this level (dB) counts as silence")

    # Shared by silero/vad.
    p.add_argument("--min-speech", type=float, default=0.2,
                   help="[silero/vad] ignore detected speech shorter than this (s)")

    # Shared.
    p.add_argument("--min-silence", type=float, default=0.5,
                   help="gaps shorter than this (s) are not treated as filler")

    # Encoding.
    p.add_argument("--video-codec", default="libx264", help="output video codec")
    p.add_argument("--audio-codec", default="aac", help="output audio codec")
    p.add_argument("--crf", type=int, default=20,
                   help="x264 quality (lower = better quality, larger file)")

    p.add_argument("--dry-run", action="store_true",
                   help="only detect and report; do not write an output file")
    p.add_argument("--ignore-errors", "-ignore-errors", dest="ignore_errors",
                   action="store_true",
                   help="tell ffmpeg to tolerate corrupted/damaged streams "
                        "(discard corrupt packets and keep going) instead of "
                        "aborting; also accepts a non-zero ffmpeg exit if an "
                        "output file was still produced")
    p.add_argument("-log", "--log", dest="log",
                   help="append a detailed report for every processed file to "
                        "this log file (filenames, codecs, lengths, kept and "
                        "removed time-ranges) — handy for folder mode")
    return p.parse_args(argv)


def run_detector(args, input_path, original):
    """Dispatch to the configured detector and return speech intervals."""
    if args.detector == "silero":
        return detect_speech_silero(
            input_path, args.threshold, args.min_silence,
            args.min_speech, args.model, args.ignore_errors,
        )
    if args.detector == "vad":
        return detect_speech_vad(
            input_path, args.aggressiveness, args.frame_ms,
            args.min_silence, args.min_speech, args.ignore_errors,
        )
    return detect_speech_silence(
        input_path, args.noise_db, args.min_silence, original,
        args.ignore_errors,
    )


def _print_ranges(title, ranges):
    print(f"{title} (start -> end, duration):")
    if not ranges:
        print("  (none)")
    for i, (s, e) in enumerate(ranges, 1):
        print(f"  {i:3d}. {format_duration(s)} -> {format_duration(e)}  "
              f"({format_duration(e - s)})")


def process_one(args, input_path, output_path):
    """Detect speech in one file, optionally cut it, and return a result dict.

    The returned dict captures everything needed for the summary and the log.
    Never raises on "no speech" — it returns a result with status set so batch
    (folder) runs can keep going.
    """
    original = get_duration(input_path, args.ignore_errors)
    result = {
        "input": input_path,
        "output": output_path,
        "detector": args.detector,
        "video_codec": args.video_codec,
        "audio_codec": args.audio_codec,
        "crf": args.crf,
        "original": original,
        "dry_run": args.dry_run,
        "ignore_errors": args.ignore_errors,
        "status": "ok",
        "keep": [],
        "removed": [],
        "shortened": 0.0,
    }

    print(f"\nAnalyzing '{input_path}' with the '{args.detector}' detector...")
    speech = run_detector(args, input_path, original)

    if not speech:
        result["status"] = "no-speech"
        result["removed"] = [(0.0, original)]
        print("  No speech detected — nothing to keep (file skipped).")
        print("  Try lowering --threshold (silero) or switching --detector.")
        return result

    keep = pad_and_merge(speech, args.pad_before, args.pad_after, original)
    removed_ranges = complement(keep, original)
    shortened = total_length(keep)
    result["keep"] = keep
    result["removed"] = removed_ranges
    result["shortened"] = shortened

    print(f"Detected {len(speech)} speech region(s); "
          f"keeping {len(keep)} segment(s) after padding/merging.\n")
    _print_ranges("Kept time-ranges", keep)
    print()
    _print_ranges("Removed time-ranges", removed_ranges)
    print()

    if not args.dry_run:
        print("Cutting and re-encoding... (this can take a while)")
        cut_and_concat(input_path, output_path, keep,
                       args.video_codec, args.audio_codec, args.crf,
                       args.ignore_errors)
        shortened = get_duration(output_path, args.ignore_errors)  # real result
        result["shortened"] = shortened

    removed = original - shortened
    percent = (removed / original * 100) if original else 0.0

    print("\n" + "=" * 44)
    print(f"  Original length:  {format_duration(original)}")
    print(f"  Shortened length: {format_duration(shortened)}")
    print(f"  Removed:          {format_duration(removed)} ({percent:.1f}%)")
    print("=" * 44)
    if not args.dry_run:
        print(f"  Saved to: {output_path}")
    return result


def write_log_entry(log_path, result):
    """Append a detailed, human-readable report for one file to the log."""
    original = result["original"]
    shortened = result["shortened"]
    removed = original - shortened
    percent = (removed / original * 100) if original else 0.0
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 70)
    lines.append(f"[{stamp}] {os.path.basename(result['input'])}")
    lines.append(f"  Original file:    {os.path.abspath(result['input'])}")
    if result["status"] == "no-speech":
        lines.append("  Output file:      (none — no speech detected, skipped)")
    elif result["dry_run"]:
        lines.append(f"  Output file:      {os.path.abspath(result['output'])} "
                     f"(dry-run, not written)")
    else:
        lines.append(f"  Output file:      {os.path.abspath(result['output'])}")
    lines.append(f"  Detector:         {result['detector']}")
    lines.append(f"  Ignore errors:    {result.get('ignore_errors', False)}")
    lines.append(f"  Video codec:      {result['video_codec']} (crf {result['crf']})")
    lines.append(f"  Audio codec:      {result['audio_codec']}")
    lines.append(f"  Original length:  {format_duration(original)}")
    lines.append(f"  Shortened length: {format_duration(shortened)}")
    lines.append(f"  Removed:          {format_duration(removed)} ({percent:.1f}%)")

    lines.append(f"  Kept time-ranges ({len(result['keep'])}):")
    for i, (s, e) in enumerate(result["keep"], 1):
        lines.append(f"     {i:3d}. {format_duration(s)} -> {format_duration(e)}  "
                     f"({format_duration(e - s)})")
    lines.append(f"  Removed time-ranges ({len(result['removed'])}):")
    for i, (s, e) in enumerate(result["removed"], 1):
        lines.append(f"     {i:3d}. {format_duration(s)} -> {format_duration(e)}  "
                     f"({format_duration(e - s)})")
    lines.append("")

    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def find_videos(folder):
    """Return sorted video files in a folder, skipping our own outputs."""
    videos = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        stem, ext = os.path.splitext(name)
        if ext.lower() not in VIDEO_EXTENSIONS:
            continue
        if stem.endswith("_processed") or stem.endswith("_short"):
            continue  # don't re-process files we (likely) produced
        videos.append(path)
    return videos


def build_jobs(args):
    """Return a list of (input_path, output_path) pairs to process."""
    if os.path.isdir(args.input):
        videos = find_videos(args.input)
        if not videos:
            sys.exit(f"error: no video files found in folder: {args.input}")
        jobs = []
        for path in videos:
            stem, _ = os.path.splitext(path)
            jobs.append((path, f"{stem}_processed.mp4"))
        return jobs, True

    if os.path.isfile(args.input):
        output = args.output
        if not output:
            base, _ = os.path.splitext(args.input)
            output = f"{base}_short.mp4"
        return [(args.input, output)], False

    sys.exit(f"error: input not found: {args.input}")


def main(argv=None):
    args = parse_args(argv)

    require_tool("ffmpeg")
    require_tool("ffprobe")
    if args.pad_before < 0 or args.pad_after < 0:
        sys.exit("error: --pad-before and --pad-after must be >= 0.")

    jobs, folder_mode = build_jobs(args)
    if folder_mode:
        print(f"Folder mode: {len(jobs)} video(s) to process in '{args.input}'.")

    results = []
    for index, (input_path, output_path) in enumerate(jobs, 1):
        if folder_mode:
            print(f"\n##### [{index}/{len(jobs)}] {os.path.basename(input_path)} #####")
        try:
            result = process_one(args, input_path, output_path)
        except SystemExit:
            raise
        except Exception as exc:  # keep a batch going if one file fails
            print(f"  ERROR processing '{input_path}': {exc}")
            if not folder_mode:
                raise
            result = {
                "input": input_path, "output": output_path,
                "detector": args.detector, "video_codec": args.video_codec,
                "audio_codec": args.audio_codec, "crf": args.crf,
                "original": 0.0, "shortened": 0.0, "dry_run": args.dry_run,
                "ignore_errors": args.ignore_errors,
                "status": "error", "keep": [], "removed": [],
            }
        results.append(result)
        if args.log:
            write_log_entry(args.log, result)

    if folder_mode:
        ok = sum(1 for r in results if r["status"] == "ok")
        print(f"\nDone. {ok}/{len(results)} file(s) processed successfully.")
    if args.log:
        print(f"Log written to: {os.path.abspath(args.log)}")


if __name__ == "__main__":
    main()
