#!/usr/bin/env python3
"""Retry failed episodes."""

import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from faster_whisper import WhisperModel

RSS_URL = "https://feed.xyzfm.space/y9qnpfdrctnx"
WORK_DIR = Path("/tmp/podcast_retry")
CLIP_DIR = Path("/home/ubuntu/podcast_byebye/clips")
TAIL_SECONDS = 30
BYEBYE_PATTERN = re.compile(r"(掰掰|拜拜|bye\s*bye)", re.IGNORECASE)

FAILED_TITLES = [
    "291-委内瑞拉怎么变成了今天这个样子？",
    "283-美国为何把支持以色列作为一种政策立场？",
    "280-莫扎特如何以一己之力重塑古典歌剧审美？",
    "67-为何极端右翼喜欢把尼采视为精神偶像？",
    "51-为什么工作不再给予我们责任感和满足感？",
]


def parse_rss():
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
        dur_str = item.findtext("{http://www.itunes.com/dtds/podcast-1.0.dtd}duration", "00:00:00")
        parts = dur_str.split(":")
        if len(parts) == 3:
            duration = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            duration = int(parts[0]) * 60 + int(parts[1])
        else:
            duration = int(parts[0])
        episodes.append({"title": title, "url": url, "guid": guid, "duration": duration})
    return episodes


def sanitize_filename(title):
    clean = re.sub(r"^\d+-", "", title).strip()
    clean = re.sub(r"[^\w\u4e00-\u9fff\-\s]", "", clean)
    clean = re.sub(r"\s+", "_", clean).strip("_")
    return clean[:80] if clean else "untitled"


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)

    episodes = parse_rss()
    failed = [ep for ep in episodes if ep["title"] in FAILED_TITLES]

    print(f"Found {len(failed)} failed episodes to retry\n")
    print("Loading Whisper model...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    for ep in failed:
        safe_name = sanitize_filename(ep["title"])
        guid_short = ep["guid"][:8]
        m4a_path = WORK_DIR / f"{guid_short}.m4a"
        tail_path = WORK_DIR / f"{guid_short}_tail.m4a"

        print(f"\n[{ep['title']}]")
        print(f"  Duration: {ep['duration']}s")

        # Download
        for attempt in range(3):
            print(f"  Download attempt {attempt + 1}...", end=" ", flush=True)
            r = subprocess.run(["curl", "-L", "-s", "-o", str(m4a_path), ep["url"]], timeout=600, capture_output=True)
            if r.returncode == 0 and m4a_path.exists() and m4a_path.stat().st_size > 100000:
                print(f"OK ({m4a_path.stat().st_size / 1024 / 1024:.1f}MB)")
                break
            print("failed")
        else:
            print("  >>> DOWNLOAD FAILED")
            continue

        # Extract tail
        start = max(0, ep["duration"] - TAIL_SECONDS)
        print(f"  Extract tail from {start}s...", end=" ", flush=True)
        r = subprocess.run(["ffmpeg", "-y", "-i", str(m4a_path), "-ss", str(start), "-t", str(TAIL_SECONDS), "-c", "copy", str(tail_path)], timeout=60, capture_output=True)
        if r.returncode != 0:
            print("copy failed, re-encoding...", end=" ", flush=True)
            r = subprocess.run(["ffmpeg", "-y", "-i", str(m4a_path), "-ss", str(start), "-t", str(TAIL_SECONDS), "-ar", "16000", "-ac", "1", str(tail_path)], timeout=120, capture_output=True)
            if r.returncode != 0:
                print("FAILED")
                continue
        print("OK")

        # Transcribe
        print("  Transcribing...", end=" ", flush=True)
        try:
            segments_gen, info = model.transcribe(str(tail_path), language="zh")
            found = False
            for seg in segments_gen:
                if BYEBYE_PATTERN.search(seg.text):
                    absolute_start = start + seg.start
                    clip_name = f"{guid_short}_{safe_name}_byebye.m4a"
                    clip_path = CLIP_DIR / clip_name
                    print(f"\n  FOUND at {absolute_start:.0f}s → {clip_name}")
                    subprocess.run(["ffmpeg", "-y", "-i", str(m4a_path), "-ss", str(absolute_start), "-c", "copy", str(clip_path)], timeout=60, capture_output=True)
                    found = True
                    break
            if not found:
                print("no byebye found")
        except Exception as e:
            print(f"ERROR: {e}")

        # Cleanup
        for f in [m4a_path, tail_path]:
            f.unlink(missing_ok=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
