#!/usr/bin/env python3
"""Batch process podcast episodes to extract byebye segments."""

import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from faster_whisper import WhisperModel

RSS_URL = "https://feed.xyzfm.space/y9qnpfdrctnx"
WORK_DIR = Path("/tmp/podcast_work")
CLIP_DIR = Path("/home/ubuntu/podcast_byebye/clips")
TAIL_SECONDS = 30
BATCH_SIZE = 10
BYEBYE_PATTERN = re.compile(r"(掰掰|拜拜|bye\s*bye)", re.IGNORECASE)


def parse_rss():
    """Fetch RSS and extract episode info."""
    import urllib.request
    with urllib.request.urlopen(RSS_URL) as resp:
        xml_data = resp.read()
    root = ET.fromstring(xml_data)
    episodes = []
    ns = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}
    for item in root.findall(".//item"):
        title = item.findtext("title", "")
        enclosure = item.find("enclosure")
        if enclosure is None:
            continue
        url = enclosure.get("url", "")
        guid = item.findtext("guid", "")
        dur_str = item.findtext("{http://www.itunes.com/dtds/podcast-1.0.dtd}duration", "00:00:00")
        # Parse HH:MM:SS or MM:SS
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


def download(url, dest):
    """Download file with curl."""
    r = subprocess.run(
        ["curl", "-L", "-s", "-o", str(dest), url],
        timeout=300, capture_output=True
    )
    return r.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def extract_tail(src, dest, duration):
    """Extract last TAIL_SECONDS of audio."""
    start = max(0, duration - TAIL_SECONDS)
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ss", str(start), "-t", str(TAIL_SECONDS), "-c", "copy", str(dest)],
        timeout=60, capture_output=True
    )
    return r.returncode == 0


def transcribe(audio_path, model):
    """Transcribe audio and return segments."""
    segments, info = model.transcribe(str(audio_path), language="zh")
    return [(seg.start, seg.end, seg.text) for seg in segments]


def find_byebye(segments):
    """Find byebye in transcription segments. Returns (start_time, text) or None."""
    for start, end, text in segments:
        if BYEBYE_PATTERN.search(text):
            return (start, end, text)
    return None


def clip_byebye(src, dest, episode_duration, tail_start, byebye_start_in_tail):
    """Clip from byebye start to end of episode."""
    absolute_start = tail_start + byebye_start_in_tail
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ss", str(absolute_start), "-c", "copy", str(dest)],
        timeout=60, capture_output=True
    )
    return r.returncode == 0


def sanitize_filename(title):
    """Make title safe for filename."""
    # Remove episode number prefix like "342-"
    clean = re.sub(r"^\d+-", "", title).strip()
    # Remove unsafe chars
    clean = re.sub(r"[^\w\u4e00-\u9fff\-\s]", "", clean)
    clean = re.sub(r"\s+", "_", clean).strip("_")
    return clean[:80] if clean else "untitled"


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)

    print("Parsing RSS feed...")
    episodes = parse_rss()
    print(f"Found {len(episodes)} episodes\n")

    print("Loading Whisper model (tiny, cpu, int8)...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("Model loaded\n")

    results = []
    total_batches = (len(episodes) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(episodes))
        batch = episodes[batch_start:batch_end]

        print(f"=== Batch {batch_idx + 1}/{total_batches} (episodes {batch_start + 1}-{batch_end}) ===")

        for ep in batch:
            safe_name = sanitize_filename(ep["title"])
            guid_short = ep["guid"][:8]
            m4a_path = WORK_DIR / f"{guid_short}.m4a"
            tail_path = WORK_DIR / f"{guid_short}_tail.m4a"

            print(f"  [{ep['title'][:50]}] ", end="", flush=True)

            # Download
            if not download(ep["url"], m4a_path):
                print("DOWNLOAD FAILED")
                results.append({**ep, "status": "download_failed"})
                continue

            # Extract tail
            if not extract_tail(m4a_path, tail_path, ep["duration"]):
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

            # Find byebye
            hit = find_byebye(segments)
            if hit:
                byebye_start_in_tail, byebye_end_in_tail, text = hit
                tail_start = max(0, ep["duration"] - TAIL_SECONDS)
                absolute_start = tail_start + byebye_start_in_tail

                clip_name = f"{ep['guid'][:8]}_{safe_name}_byebye.m4a"
                clip_path = CLIP_DIR / clip_name

                if clip_byebye(m4a_path, clip_path, ep["duration"], tail_start, byebye_start_in_tail):
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
            for f in [m4a_path, tail_path]:
                f.unlink(missing_ok=True)

        print()

    # Summary report
    report_path = Path("/home/ubuntu/podcast_byebye/report.txt")
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
                f.write(f"    Time: {r['byebye_time']:.0f}s  Text: {r['byebye_text']}\n")
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
