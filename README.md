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
```

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
