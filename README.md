# Episode Shortener

Turkish series (and plenty of others) are full of long, silent filler scenes
where nobody is talking — slow zooms, dramatic stares, sweeping landscape shots.
This tool detects the **speech** in an episode and cuts out everything else,
keeping only the dialogue plus a small buffer around it so the cuts don't feel
abrupt. The result is a much shorter video you can watch in a fraction of the
time.

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
- For the default voice detector: `pip install -r requirements.txt`.

## Usage

```bash
# Default: voice-activity detection, keep 0.5s before / 1.0s after each line
python3 episode_shortener.py episode.mkv -o episode_short.mp4

# Tighter cut: keep less padding
python3 episode_shortener.py episode.mkv -x 0.3 -y 0.5

# Just see how much would be removed, without encoding anything
python3 episode_shortener.py episode.mkv --dry-run
```

### Key options

| Option | Default | Meaning |
| --- | --- | --- |
| `-x`, `--pad-before` | `0.5` | Seconds of video to keep **before** each speech region |
| `-y`, `--pad-after` | `1.0` | Seconds of video to keep **after** each speech region |
| `--detector` | `vad` | `vad` (detects human voice) or `silence` (keeps any non-quiet audio) |
| `--min-silence` | `0.5` | Gaps shorter than this aren't treated as filler |
| `--dry-run` | off | Detect & report only; don't write a file |
| `--crf` | `20` | Output quality for x264 (lower = better/larger) |

Run `python3 episode_shortener.py --help` for the full list, including
VAD tuning (`--aggressiveness`, `--frame-ms`, `--min-speech`) and the
`silence` detector's `--noise-db` threshold.

## Detectors

- **`vad` (default)** — WebRTC voice-activity detection. Looks specifically for
  *human voice*, so it handles scenes that have background music without keeping
  them just because they're not silent. Best choice for most series.
- **`silence`** — ffmpeg's `silencedetect`. Keeps anything louder than a dB
  threshold. No extra Python dependencies and very fast, but it will keep
  music/ambience too. Good when the filler is genuinely quiet.

## Tips

- If too much is being cut, try a lower `--aggressiveness` (e.g. `1`) or larger
  `-x`/`-y` buffers.
- If filler is being kept, raise `--aggressiveness` to `3`, or for the
  `silence` detector lower `--noise-db` (e.g. `-35`).
