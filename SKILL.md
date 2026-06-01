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

### Prerequisites

```bash
# Create venv and install faster-whisper
uv venv /tmp/podcast_whisper_venv
uv pip install -p /tmp/podcast_whisper_venv/bin/python faster-whisper
```

### Script Structure (process.py)

```python
import os, re, subprocess, xml.etree.ElementTree as ET
from pathlib import Path
from faster_whisper import WhisperModel

# --- Config ---
RSS_URL = "https://feed.xyzfm.space/y9qnpfdrctnx"
WORK_DIR = Path("/tmp/podcast_work")
CLIP_DIR = Path("/home/ubuntu/podcast_byebye/clips")
TAIL_SECONDS = 30
BATCH_SIZE = 10
BYEBYE_PATTERN = re.compile(r"(掰掰|拜拜|bye\s*bye)", re.IGNORECASE)
```

### Step 1: Parse RSS

```python
def parse_rss():
    """Fetch RSS XML, extract episode title/url/guid/duration."""
    # Use urllib to fetch RSS
    # Parse with xml.etree.ElementTree
    # Extract from each <item>:
    #   - title
    #   - enclosure url (audio URL)
    #   - guid
    #   - itunes:duration (parse HH:MM:SS → seconds)
    # Returns list of dicts
```

### Step 2: Batch Download + Process

```python
def main():
    episodes = parse_rss()
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    
    for batch in chunked(episodes, BATCH_SIZE):
        for ep in batch:
            # 1. Download full m4a with curl
            # 2. ffmpeg -ss (duration - 30) -t 30 -c copy → tail.m4a
            # 3. Whisper transcribe tail.m4a
            # 4. Regex search for byebye pattern
            # 5. If found: ffmpeg -ss (absolute_start) -c copy → clip.m4a
            # 6. Delete downloaded files (keep only clips)
```

### Step 3: FFmpeg Operations

```bash
# Extract tail (last 30s)
ffmpeg -y -i input.m4a -ss $START -t 30 -c copy tail.m4a

# Clip from byebye start to end
ffmpeg -y -i input.m4a -ss $BYEBYE_START -c copy clip.m4a
```

### Step 4: Report

Generate report.txt with:
- Total episodes, found count, not-found count, error count
- For each found: title, timestamp, clip filename
- For each not-found: title

## Pitfalls

1. **m4a duration parsing** — Some RSS feeds use HH:MM:SS, others MM:SS. Handle both.
2. **Whisper language** — Explicitly set `language="zh"` for Chinese podcasts to improve accuracy.
3. **Filename safety** — Sanitize Chinese titles for filesystem. Remove special chars, truncate to 80 chars.
4. **Download failures** — Network errors happen. Log and continue, don't abort the batch.
5. **Virtual env** — `pip` may not be available on system Python. Use `uv` to create venv and install packages.
6. **Disk space** — Always delete downloaded originals after processing each batch. Only keep extracted clips.

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

## Verification

After running, spot-check 3-5 clips by ear. Verify:
- Clip starts at the sign-off phrase (not too early/late)
- Audio quality is preserved (using -c copy, no re-encoding)
- No false positives (background music mistaken for speech)
