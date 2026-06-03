#!/usr/bin/env python3
"""Episode shortener.

Detect speech in a video and cut out the long filler scenes where nobody is
talking. Around every detected speech region we keep a configurable buffer of
``x`` seconds before the speech and ``y`` seconds after it; everything outside
those padded speech regions is dropped. The result is a tighter cut that keeps
the dialogue (plus a little breathing room) and removes the dead air that makes
Turkish series drag on.

Usage:
    python3 episode_shortener.py input.mkv -o output.mp4 -x 0.5 -y 1.0

Requires ``ffmpeg``/``ffprobe`` on PATH. The default ``vad`` detector also needs
the ``webrtcvad`` (or ``webrtcvad-wheels``) Python package.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
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


def get_duration(path):
    """Return the duration of a media file in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
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
def detect_speech_vad(path, aggressiveness, frame_ms, min_silence, min_speech):
    """Detect speech with WebRTC voice-activity detection.

    Better than plain volume thresholding for content with background music,
    because it looks for voice specifically rather than just "loud audio".
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
            "ffmpeg", "-v", "error", "-i", path,
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
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


def detect_speech_silence(path, noise_db, min_silence, duration):
    """Detect speech as 'whatever isn't silence', using ffmpeg's silencedetect.

    Fast and dependency-free, but treats any audio above the threshold
    (including music) as speech. Returns a list of (start, end) intervals.
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "info", "-i", path, "-vn",
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


# --------------------------------------------------------------------------- #
# Cutting
# --------------------------------------------------------------------------- #
def cut_and_concat(input_path, output_path, segments, video_codec, audio_codec, crf):
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
            "ffmpeg", "-v", "error", "-stats", "-y", "-i", input_path,
            "-filter_complex_script", script_path,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", video_codec, "-crf", str(crf),
            "-c:a", audio_codec,
            output_path,
        ]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            sys.exit("error: ffmpeg failed while cutting the video.")
    finally:
        os.unlink(script_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Shorten a video by keeping speech (plus a buffer) and "
                    "removing silent filler scenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="path to the input video file")
    p.add_argument("-o", "--output", help="path to the output video "
                   "(default: <input>_short.mp4)")
    p.add_argument("-x", "--pad-before", type=float, default=0.5,
                   help="seconds of video to KEEP before each speech region")
    p.add_argument("-y", "--pad-after", type=float, default=1.0,
                   help="seconds of video to KEEP after each speech region")

    p.add_argument("--detector", choices=["vad", "silence"], default="vad",
                   help="speech detector: 'vad' finds human voice (best for "
                        "content with background music), 'silence' keeps any "
                        "non-quiet audio")

    # VAD detector options.
    p.add_argument("--aggressiveness", type=int, choices=[0, 1, 2, 3], default=2,
                   help="[vad] 0=lenient .. 3=aggressive at rejecting non-speech")
    p.add_argument("--frame-ms", type=int, choices=[10, 20, 30], default=30,
                   help="[vad] analysis frame size in milliseconds")
    p.add_argument("--min-speech", type=float, default=0.2,
                   help="[vad] ignore detected speech shorter than this (s)")

    # Silence detector options.
    p.add_argument("--noise-db", type=float, default=-30.0,
                   help="[silence] audio below this level (dB) counts as silence")

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
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    require_tool("ffmpeg")
    require_tool("ffprobe")
    if not os.path.isfile(args.input):
        sys.exit(f"error: input file not found: {args.input}")
    if args.pad_before < 0 or args.pad_after < 0:
        sys.exit("error: --pad-before and --pad-after must be >= 0.")

    output = args.output
    if not output:
        base, _ = os.path.splitext(args.input)
        output = f"{base}_short.mp4"

    original = get_duration(args.input)

    print(f"Analyzing '{args.input}' with the '{args.detector}' detector...")
    if args.detector == "vad":
        speech = detect_speech_vad(
            args.input, args.aggressiveness, args.frame_ms,
            args.min_silence, args.min_speech,
        )
    else:
        speech = detect_speech_silence(
            args.input, args.noise_db, args.min_silence, original,
        )

    if not speech:
        sys.exit("No speech detected — nothing to keep. Try a different detector "
                 "or relax the thresholds (e.g. lower --aggressiveness or "
                 "--noise-db).")

    keep = pad_and_merge(speech, args.pad_before, args.pad_after, original)
    shortened = total_length(keep)
    removed = original - shortened
    percent = (removed / original * 100) if original else 0.0

    print(f"Detected {len(speech)} speech region(s); "
          f"keeping {len(keep)} segment(s) after padding/merging.\n")

    if not args.dry_run:
        print("Cutting and re-encoding... (this can take a while)")
        cut_and_concat(args.input, output, keep,
                       args.video_codec, args.audio_codec, args.crf)
        shortened = get_duration(output)  # report the real result
        removed = original - shortened
        percent = (removed / original * 100) if original else 0.0

    # ----- summary -------------------------------------------------------- #
    print("\n" + "=" * 44)
    print(f"  Original length:  {format_duration(original)}")
    print(f"  Shortened length: {format_duration(shortened)}")
    print(f"  Removed:          {format_duration(removed)} ({percent:.1f}%)")
    print("=" * 44)
    if not args.dry_run:
        print(f"  Saved to: {output}")


if __name__ == "__main__":
    main()
