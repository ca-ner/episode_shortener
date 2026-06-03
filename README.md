# Episode Shortener

Turkish series (and plenty of others) are full of long stretches where nobody
is talking — slow zooms, dramatic stares, sweeping landscape shots, and scenes
carried by background music. This tool detects the **speech** in an episode and
cuts out *everything else* — music, ambience and silence alike — keeping only
the dialogue plus a small buffer around it so the cuts don't feel abrupt. The
result is a much shorter video you can watch in a fraction of the time.

By default it uses [Silero VAD](https://github.com/snakers4/silero-vad), a
neural voice-activity model that reliably tells human speech apart from music
and noise, so a scene scored with music (but no dialogue) gets removed too.

## How it works

1. Reads the input video.
2. Detects where speech happens (see [detectors](#detectors)).
3. Keeps each speech region **plus `x` seconds before and `y` seconds after**,
   merges anything that overlaps, and drops the rest (the filler).
4. Re-encodes the kept pieces into a single output file.
5. Prints the **original length, the shortened length, and how much was removed.**

> Note on `x`/`y`: these are the buffers **kept** around speech. Bigger values =
> safer, gentler cuts but a longer result; smaller values = tighter, more
> aggressive shortening.

## Requirements

- `ffmpeg` and `ffprobe` on your `PATH` (e.g. `apt install ffmpeg` /
  `brew install ffmpeg`).
- `pip install -r requirements.txt` (installs `onnxruntime` + `numpy` for the
  default Silero detector).
- The Silero model file `silero_vad.onnx` ships in this repo next to the
  script. If it's missing it will be downloaded automatically on first run.

## Usage

```bash
# Default: Silero speech-only detection, keep 0.5s before / 1.0s after each line
python3 episode_shortener.py episode.mkv -o episode_short.mp4

# Keep 3s before and 3s after each speech, cut everything else
python3 episode_shortener.py episode.mkv -o episode_short.mp4 -before 3 -after 3

# Tighter cut: keep less padding
python3 episode_shortener.py episode.mkv -x 0.3 -y 0.5

# Be stricter about what counts as speech (drops borderline music more readily)
python3 episode_shortener.py episode.mkv --threshold 0.6

# Just see how much would be removed, without encoding anything
python3 episode_shortener.py episode.mkv --dry-run

# Tolerate corrupted/damaged streams instead of aborting
python3 episode_shortener.py episode.mkv --ignore-errors

# Folder mode: process EVERY video in a folder, writing <name>_processed.mp4
python3 episode_shortener.py /path/to/season -x 1 -y 1

# Folder mode + a detailed log of every file
python3 episode_shortener.py /path/to/season -log season.log
```

## Folder mode

Pass a **folder** instead of a file and every video inside it is processed,
each producing a `<name>_processed.mp4` next to the original:

```
season/
  ep01.mkv          ->  season/ep01_processed.mp4
  ep02.mkv          ->  season/ep02_processed.mp4
```

Recognised extensions: `.mp4 .mkv .avi .mov .ts .webm .flv .wmv .mpg .mpeg
.m4v .3gp .m2ts .mts`. Files already ending in `_processed` (or `_short`) are
skipped, so re-running the folder won't re-process its own output. If one file
fails it's logged and the batch continues with the rest. (In folder mode `-o`
is ignored, since names are generated automatically.)

## Logging

Add `-log <file>` (or `--log`) to append a detailed report for **every**
processed file — especially useful in folder mode. Each entry records the
original and output filenames, the video/audio codecs, the original /
shortened / removed lengths, and the full kept and removed time-ranges:

```
======================================================================
[2026-06-03 21:57:37] ep01.mkv
  Original file:    /path/season/ep01.mkv
  Output file:      /path/season/ep01_processed.mp4
  Detector:         silero
  Video codec:      libx264 (crf 20)
  Audio codec:      aac
  Original length:  45:12.30
  Shortened length: 22:01.10
  Removed:          23:11.20 (51.3%)
  Kept time-ranges (128):
       1. 0:05.04 -> 0:07.53  (0:02.49)
       ...
  Removed time-ranges (129):
       1. 0:00.00 -> 0:05.04  (0:05.04)
       ...
```

The log is appended to (never overwritten), so a folder run produces one entry
per file in a single file.

## Corrupted streams (`--ignore-errors`)

Damaged rips sometimes have corrupt packets that make ffmpeg bail out. Pass
`--ignore-errors` to push through them: it adds ffmpeg's
`-err_detect ignore_err -fflags +discardcorrupt+genpts` to every read of the
input (so corrupt packets are dropped and timestamps regenerated), and it
accepts a non-zero ffmpeg exit code as long as an output file was still
produced. Handy in folder mode where one bad file shouldn't stop the batch.
The output may have small glitches around the damaged spots, but you get a
usable file instead of a hard failure.

### Key options

| Option | Default | Meaning |
| --- | --- | --- |
| `-x`, `-before`, `--pad-before` | `0.5` | Seconds of video to keep **before** each speech region |
| `-y`, `-after`, `--pad-after` | `1.0` | Seconds of video to keep **after** each speech region |
| `--detector` | `silero` | `silero` (speech only, rejects music), `vad` (lighter WebRTC), or `silence` |
| `--threshold` | `0.5` | [silero] Speech probability cutoff 0–1; raise it to drop more music |
| `--min-silence` | `0.5` | Gaps shorter than this aren't treated as filler |
| `--min-speech` | `0.2` | Ignore detected speech shorter than this |
| `--dry-run` | off | Detect & report only; don't write a file |
| `--ignore-errors` | off | Tolerate corrupted streams (see below) |
| `-log`, `--log` | off | Append a detailed per-file report to this log file |
| `--crf` | `20` | Output quality for x264 (lower = better/larger) |

Run `python3 episode_shortener.py --help` for the full list.

## Detectors

- **`silero` (default)** — [Silero VAD](https://github.com/snakers4/silero-vad),
  a small neural voice-activity model. Trained specifically to recognise human
  speech, so it **rejects music, sound effects and ambience** rather than
  keeping them just because they're loud. This is what gives you a speech-only
  cut. Tune it with `--threshold`.
- **`vad`** — WebRTC voice-activity detection. Lightweight, but weaker at
  rejecting music (it can mistake music for voice). A fallback if you can't
  install `onnxruntime`.
- **`silence`** — ffmpeg's `silencedetect`. Keeps anything louder than a dB
  threshold. No Python dependencies and very fast, but it keeps music/ambience
  too. Good only when the filler is genuinely quiet.

## Tips

- If real dialogue is being cut, **lower** `--threshold` (e.g. `0.4`) or
  increase the `-x`/`-y` buffers.
- If music or noise is still slipping through, **raise** `--threshold`
  (e.g. `0.6`–`0.7`).
