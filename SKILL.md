---
name: podcast-byeee-extraction
description: "Extract recurring outro segments (e.g. byebye) from podcast episodes via RSS → download → Whisper → regex → clip."
version: 1.0.0
author: hermes
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [podcast, audio, whisper, speech-recognition, rss, ffmpeg]
prerequisites:
  commands: [ffmpeg, ffprobe, curl, python3]
  python_packages: [faster-whisper]
---

# Podcast Byebye Extraction

Extract a recurring vocal outro (e.g. "拜拜/掰掰/bye bye") from every episode of a podcast, given its RSS feed.

## Use Cases

- Isolate a host's signature sign-off phrase across hundreds of episodes
- Collect recurring audio motifs from any podcast for remix/compilation
- Detect which episodes lack the pattern (quality/consistency check)

## Pipeline

```
RSS feed → parse XML → batch download m4a → ffmpeg tail extract → Whisper STT → regex match → ffmpeg clip
```

No LLM involved. Pure local CPU computation (Whisper tiny, int8 quantized).

## Key Design Decisions

1. **Batch processing** — Download N episodes (default 10), process, delete, next batch. Avoids filling disk.
2. **Tail-only transcription** — Only extract last 30-60 seconds for Whisper, not the full episode. Huge time saving.
3. **Whisper tiny + int8** — Fast on CPU (~2-3s per 30s clip). Accuracy is sufficient for short sign-off phrases.
4. **Regex over LLM** — Pattern matching is reliable for known phrases. No need for semantic understanding.

## Implementation

A ready-to-use script is at `templates/process.py`. A retry script for failed episodes is at `scripts/retry.py`. Edit the CONFIG section at the top of each, then run with the venv's python.

### Prerequisites

```bash
# Create venv and install faster-whisper (use uv, pip may not be available)
uv venv /path/to/venv
uv pip install -p /path/to/venv/bin/python faster-whisper
```

### Quick Start

```bash
# 1. Copy template
cp templates/process.py ~/my_project/process.py

# 2. Edit CONFIG section: RSS_URL, CLIP_DIR, TARGET_PATTERN, WHISPER_LANG

# 3. Run
/path/to/venv/bin/python process.py
```

## Pitfalls

1. **m4a duration parsing** — Some RSS feeds use HH:MM:SS, others MM:SS. Handle both.
2. **Whisper language** — Explicitly set `language="zh"` for Chinese podcasts to improve accuracy.
3. **Filename safety** — Sanitize Chinese titles for filesystem. Remove special chars, truncate to 80 chars.
4. **Download failures** — Network errors happen. Log and continue, don't abort the batch.
5. **Virtual env** — `pip` may not be available on system Python. Use `uv` to create venv and install packages.
6. **Disk space** — Always delete downloaded originals after processing each batch. Only keep extracted clips.
7. **mp3 vs m4a format** — Some episodes are mp3, not m4a. When clipping with `-c copy` from mp3 to m4a container, the output will be 0 bytes. Detect the source format and use `-c:a aac` (re-encode) when the source is mp3.
8. **Whisper transcribe() returns a generator** — `model.transcribe()` returns `(segments_generator, info)`, NOT a list. You must unpack the tuple: `segments_gen, info = model.transcribe(...)`, then iterate `for seg in segments_gen:`. Do NOT do `list(model.transcribe(...))` and access `.text` — it will fail with `'generator' object has no attribute 'text'`.
9. **ffmpeg -c copy seek on mp3** — `ffmpeg -ss $time -c copy` may fail silently on mp3 files (producing empty output). When the source is mp3, always re-encode: `-c:a aac` or `-ar 16000 -ac 1`.

## Adapting to Other Podcasts

- Change `RSS_URL` to the target feed
- Adjust `BYEBYE_PATTERN` regex for the target phrase
- Adjust `TAIL_SECONDS` if the sign-off is earlier/later
- Adjust `language` parameter in Whisper for non-Chinese podcasts

## Output Structure

```
/home/ubuntu/podcast_byebye/
├── clips/
│   ├── {guid}_{title}_byebye.m4a
│   └── ...
├── report.txt
└── process.py
```

## Workflow: Test Before Batch

Always validate with 3-5 known examples before running the full batch:

1. User provides example episodes + timestamps where the target phrase appears
2. Download those episodes, extract tail, transcribe with Whisper — confirm detection works
3. Clip the exact segment, present to user for ear-verification
4. Only then proceed to batch processing

This catches issues (wrong regex, insufficient tail length, Whisper misrecognition) early.

## Pitfalls (Additional)

7. **Do NOT use spectrogram/vision for speech detection** — `songsee` can generate spectrograms and `vision_analyze` can look at them, but this only shows frequency/energy patterns, not speech content. It cannot identify specific words. Use Whisper for speech-to-text.
8. **Whisper transcription varies** — The same word may be transcribed differently across episodes (e.g. "掰掰" vs "拜拜" vs "bye bye"). Always include multiple variants in the regex pattern.
9. **Long episodes need longer download timeouts** — Episodes over 2 hours (100MB+) can take 20-30s to download. Set curl timeout accordingly.
10. **Mixed format RSS feeds** — Some episodes are mp3, others m4a, within the SAME feed. The RSS `<enclosure type="audio/mp4">` is not always accurate. Using `ffmpeg -c copy` to clip an mp3 source into an m4a container produces a **0-byte file silently** (no error, no warning). Detection: check `ffprobe -show_entries format=format_name` on the source. Fix: use `-c:a aac` (re-encode) instead of `-c copy` when the source is mp3. Safer approach: always verify `clip_path.stat().st_size > 0` after clipping; if 0, retry with re-encoding.
11. **Whisper `model.transcribe()` return value** — Returns a **tuple** `(segments_generator, info)`, NOT a list. You must unpack: `segments_gen, info = model.transcribe(...)`. Writing `segments = list(model.transcribe(...))` and then `segments[0].text` crashes with `'generator' object has no attribute 'text'`.
12. **EXTRACT FAILED on some episodes** — Rare ffmpeg failures. Log and continue. These can be retried individually later.
13. **Retry strategy** — After batch processing, retry all errors (download_failed, extract_failed, transcribe_failed). Most are transient network issues. For extract_failed on mp3 sources, switch to re-encoding (`-c:a aac`). A retry script is at `scripts/retry.py`.
