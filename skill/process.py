#!/usr/bin/env python3
"""
Batch extract recurring outro segments (e.g. byebye) from podcast episodes.

Pipeline: RSS → parse XML → batch download → ffmpeg tail → Whisper STT → regex → clip

Usage:
    /path/to/venv/bin/python process.py
"""

import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from faster_whisper import WhisperModel

# ============ CONFIG ============
RSS_URL = "https://feed.xyzfm.space/y9qnpfdrctnx"
WORK_DIR = Path("/tmp/podcast_work")
CLIP_DIR = Path("/home/ubuntu/podcast_byebye/clips")
TAIL_SECONDS = 30          # How many seconds from the end to transcribe
BATCH_SIZE = 10             # Episodes per batch
WHISPER_MODEL = "tiny"      # tiny/base/small/medium/large
WHISPER_LANG = "zh"         # Whisper language hint
# Pattern to match in transcription. Multiple variants for the same phrase.
TARGET_PATTERN = re.compile(r"(掰掰|拜拜|bye\s*bye)", re.IGNORECASE)
# ================================


def parse_rss():
    """Fetch RSS and extract episode info."""
    import urllib.request
    with urllib.request.urlopen(RSS_URL) as resp:
        xml_data = resp.read()
    root = ET.fromstring(xml_data)
    episodes = []
    for item in root.findall(".//item"):
        title = item.findtext("title", "")
        enclosure = item.find("enclosure")
        if enclosure is None:
            continue
        url = enclosure.get("url", "")
        guid = item.findtext("guid", "")
        dur_str = item.findtext(
            "{http://www.itunes.com/dtds/podcast-1.0.dtd}duration", "00:00:00"
        )
        parts = dur_str.split(":")
        if len(parts) == 3:
            duration = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            duration = int(parts[0]) * 60 + int(parts[1])
        else:
            duration = int(parts[0])
        episodes.append({
            "title": title,
            "url": url,
            "guid": guid,
            "duration": duration,
            "duration_str": dur_str,
        })
    return episodes


def is_mp3_source(url, file_path=None):
    """Check if source is mp3 (by URL extension or ffprobe)."""
    if url.lower().endswith(".mp3"):
        return True
    if file_path and file_path.exists():
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=format_name",
             "-of", "csv=p=0", str(file_path)],
            capture_output=True, text=True, timeout=10
        )
        return "mp3" in (r.stdout or "")
    return False


def download(url, dest):
    """Download file with curl."""
    r = subprocess.run(
        ["curl", "-L", "-s", "-o", str(dest), url],
        timeout=600, capture_output=True
    )
    return r.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def extract_tail(src, dest, duration, is_mp3=False):
    """Extract last TAIL_SECONDS of audio."""
    start = max(0, duration - TAIL_SECONDS)
    # For mp3 sources, re-encode to avoid container mismatch issues
    if is_mp3:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ss", str(start),
             "-t", str(TAIL_SECONDS), "-ar", "16000", "-ac", "1", str(dest)],
            timeout=120, capture_output=True
        )
    else:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ss", str(start),
             "-t", str(TAIL_SECONDS), "-c", "copy", str(dest)],
            timeout=60, capture_output=True
        )
    # Fallback: if copy failed, try re-encoding
    if r.returncode != 0 and not is_mp3:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ss", str(start),
             "-t", str(TAIL_SECONDS), "-ar", "16000", "-ac", "1", str(dest)],
            timeout=120, capture_output=True
        )
    return r.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def clip_segment(src, dest, start_time, is_mp3=False):
    """Clip from start_time to end of episode."""
    if is_mp3:
        # Re-encode mp3 → m4a to avoid 0-byte silent failure
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ss", str(start_time),
             "-c:a", "aac", str(dest)],
            timeout=120, capture_output=True
        )
    else:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ss", str(start_time),
             "-c", "copy", str(dest)],
            timeout=60, capture_output=True
        )
    return r.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def transcribe(audio_path, model):
    """Transcribe audio. Returns list of (start, end, text)."""
    # IMPORTANT: model.transcribe() returns (segments_generator, info)
    segments_gen, info = model.transcribe(str(audio_path), language=WHISPER_LANG)
    return [(seg.start, seg.end, seg.text) for seg in segments_gen]


def find_target(segments):
    """Find target pattern in transcription segments. Returns (start, end, text) or None."""
    for start, end, text in segments:
        if TARGET_PATTERN.search(text):
            return (start, end, text)
    return None


def sanitize_filename(title):
    """Make title safe for filename."""
    clean = re.sub(r"^\d+-", "", title).strip()
    clean = re.sub(r"[^\w\u4e00-\u9fff\-\s]", "", clean)
    clean = re.sub(r"\s+", "_", clean).strip("_")
    return clean[:80] if clean else "untitled"


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)

    print("Parsing RSS feed...")
    episodes = parse_rss()
    print(f"Found {len(episodes)} episodes\n")

    print(f"Loading Whisper model ({WHISPER_MODEL}, cpu, int8)...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("Model loaded\n")

    results = []
    total_batches = (len(episodes) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(episodes))
        batch = episodes[batch_start:batch_end]

        print(f"=== Batch {batch_idx + 1}/{total_batches} "
              f"(episodes {batch_start + 1}-{batch_end}) ===")

        for ep in batch:
            safe_name = sanitize_filename(ep["title"])
            guid_short = ep["guid"][:8]
            url = ep["url"]
            is_mp3 = url.lower().endswith(".mp3")
            ext = "mp3" if is_mp3 else "m4a"
            src_path = WORK_DIR / f"{guid_short}.{ext}"
            tail_path = WORK_DIR / f"{guid_short}_tail.m4a"

            print(f"  [{ep['title'][:50]}] ", end="", flush=True)

            # Download
            if not download(url, src_path):
                print("DOWNLOAD FAILED")
                results.append({**ep, "status": "download_failed"})
                continue

            # Detect mp3 if URL didn't reveal it
            if not is_mp3:
                is_mp3 = is_mp3_source(url, src_path)

            # Extract tail
            if not extract_tail(src_path, tail_path, ep["duration"], is_mp3):
                print("EXTRACT FAILED")
                results.append({**ep, "status": "extract_failed"})
                continue

            # Transcribe
            try:
                segments = transcribe(tail_path, model)
            except Exception as e:
                print(f"TRANSCRIBE FAILED: {e}")
                results.append({**ep, "status": "transcribe_failed"})
                continue

            # Find target
            hit = find_target(segments)
            if hit:
                target_start_in_tail, target_end_in_tail, text = hit
                tail_start = max(0, ep["duration"] - TAIL_SECONDS)
                absolute_start = tail_start + target_start_in_tail

                clip_name = f"{guid_short}_{safe_name}_byebye.m4a"
                clip_path = CLIP_DIR / clip_name

                if clip_segment(src_path, clip_path, absolute_start, is_mp3):
                    print(f"FOUND at {absolute_start:.0f}s → {clip_name}")
                    results.append({
                        **ep,
                        "status": "found",
                        "byebye_time": absolute_start,
                        "byebye_text": text,
                        "clip_file": clip_name,
                    })
                else:
                    print("CLIP FAILED")
                    results.append({**ep, "status": "clip_failed"})
            else:
                print("no byebye")
                results.append({**ep, "status": "not_found"})

            # Cleanup downloaded files
            for f in [src_path, tail_path]:
                f.unlink(missing_ok=True)

        print()

    # Summary report
    report_path = CLIP_DIR.parent / "report.txt"
    found = [r for r in results if r["status"] == "found"]
    not_found = [r for r in results if r["status"] == "not_found"]
    errors = [r for r in results if r["status"] not in ("found", "not_found")]

    with open(report_path, "w") as f:
        f.write(f"=== Podcast Byebye Extraction Report ===\n")
        f.write(f"Total episodes: {len(results)}\n")
        f.write(f"Byebye found: {len(found)}\n")
        f.write(f"Byebye not found: {len(not_found)}\n")
        f.write(f"Errors: {len(errors)}\n\n")

        if found:
            f.write(f"=== Found ({len(found)}) ===\n")
            for r in found:
                f.write(f"  {r['title']}\n")
                f.write(f"    Time: {r['byebye_time']:.0f}s  "
                        f"Text: {r['byebye_text']}\n")
                f.write(f"    Clip: {r['clip_file']}\n\n")

        if not_found:
            f.write(f"\n=== Not Found ({len(not_found)}) ===\n")
            for r in not_found:
                f.write(f"  {r['title']}\n")

        if errors:
            f.write(f"\n=== Errors ({len(errors)}) ===\n")
            for r in errors:
                f.write(f"  {r['title']}: {r['status']}\n")

    print(f"\n{'=' * 50}")
    print(f"DONE! Total: {len(results)}")
    print(f"  Found:     {len(found)}")
    print(f"  Not found: {len(not_found)}")
    print(f"  Errors:    {len(errors)}")
    print(f"\nClips: {CLIP_DIR}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
