#!/usr/bin/env python3
"""
Transcribes any audio files sitting in incoming/ using the Gemini API,
writes results to transcripts/, and removes the processed source file.
Runs inside GitHub Actions - no phone, no browser, no memory limit worries.
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import time

import requests

API_KEY = os.environ["GEMINI_API_KEY"]
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
CHUNK_SECONDS = int(os.environ.get("CHUNK_SECONDS", "600"))  # 10 minutes

INCOMING_DIR = "incoming"
TRANSCRIPTS_DIR = "transcripts"
WORK_DIR = "work"

PROMPT = """Transcribe this audio recording verbatim.
The speakers mix English and Egyptian Arabic in the same conversation (code-switching).
Write each language as it was actually spoken: Arabic words in Arabic script, English words in English - do not translate anything.
Label speakers as "Speaker 1", "Speaker 2", etc., based on distinct voices, and start a new line each time the speaker changes.
Add an approximate timestamp like [00:30] at the start of each new speaker turn if you can tell.
If a word or phrase is unclear, write [inaudible] rather than guessing."""

MIME_MAP = {
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".mp4": "audio/mp4",
    ".aac": "audio/aac", ".ogg": "audio/ogg", ".oga": "audio/ogg", ".flac": "audio/flac",
    ".webm": "audio/webm", ".amr": "audio/amr", ".3gp": "audio/3gpp",
}


def guess_mime(path):
    ext = os.path.splitext(path)[1].lower()
    return MIME_MAP.get(ext, "application/octet-stream")


def mmss(seconds):
    seconds = max(0, round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def shift_timestamps(text, offset_seconds):
    def repl(m):
        h, mi, s = m.group(1), m.group(2), m.group(3)
        secs = (int(h) * 3600 + int(mi) * 60 + int(s)) if s is not None else (int(h) * 60 + int(mi))
        return f"[{mmss(secs + offset_seconds)}]"
    return re.sub(r"\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]", repl, text)


def upload_to_gemini(path, mime_type):
    size = os.path.getsize(path)
    start = requests.post(
        f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={API_KEY}",
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": os.path.basename(path)}},
        timeout=60,
    )
    start.raise_for_status()
    upload_url = start.headers["X-Goog-Upload-URL"]

    with open(path, "rb") as f:
        up = requests.post(
            upload_url,
            headers={
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            data=f,
            timeout=600,
        )
    up.raise_for_status()
    file_info = up.json()["file"]
    return file_info["uri"], file_info["name"]


def wait_for_active(name, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/{name}?key={API_KEY}", timeout=30
        )
        r.raise_for_status()
        state = r.json().get("state")
        if state == "ACTIVE":
            return
        if state == "FAILED":
            raise RuntimeError("Gemini file processing failed")
        time.sleep(3)
    raise RuntimeError("Gemini file processing timed out")


def generate(file_uri, mime_type, prompt):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}",
        json={
            "contents": [{"parts": [
                {"text": prompt},
                {"file_data": {"mime_type": mime_type, "file_uri": file_uri}},
            ]}]
        },
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    parts = data["candidates"][0]["content"]["parts"]
    return "\n".join(p.get("text", "") for p in parts).strip()


def get_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrapped_values=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def split_audio(path, out_dir, chunk_seconds):
    os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".m4a"
    pattern = os.path.join(out_dir, f"chunk_%04d{ext}")
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-f", "segment", "-segment_time", str(chunk_seconds),
         "-c", "copy", "-reset_timestamps", "1", pattern],
        check=True, capture_output=True,
    )
    return sorted(glob.glob(os.path.join(out_dir, f"chunk_*{ext}")))


def transcribe_one(path, mime_type, prompt):
    file_uri, file_name = upload_to_gemini(path, mime_type)
    wait_for_active(file_name)
    return generate(file_uri, mime_type, prompt)


def process_file(path):
    print(f"Processing {path}", flush=True)
    mime_type = guess_mime(path)
    duration = get_duration(path)

    if duration <= CHUNK_SECONDS * 1.1:
        return transcribe_one(path, mime_type, PROMPT)

    name = os.path.splitext(os.path.basename(path))[0]
    work_dir = os.path.join(WORK_DIR, name)
    chunks = split_audio(path, work_dir, CHUNK_SECONDS)

    full_text = []
    offset = 0.0
    for i, chunk_path in enumerate(chunks):
        chunk_duration = get_duration(chunk_path)
        chunk_prompt = PROMPT + (
            f"\n\nThis is part {i + 1} of {len(chunks)} of a longer call "
            f"(covers roughly {mmss(offset)}-{mmss(offset + chunk_duration)} of the original recording). "
            f"Transcribe only this segment, no summary or preamble. "
            f"Speaker numbering does not need to match other parts."
        )
        print(f"  chunk {i + 1}/{len(chunks)}", flush=True)
        text = transcribe_one(chunk_path, guess_mime(chunk_path), chunk_prompt)
        shifted = shift_timestamps(text, offset)
        full_text.append(
            f"\n----- Part {i + 1}/{len(chunks)} ({mmss(offset)}-{mmss(offset + chunk_duration)}) -----\n{shifted}\n"
        )
        offset += chunk_duration

    shutil.rmtree(work_dir, ignore_errors=True)
    return "".join(full_text)


def main():
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    files = sorted(
        p for p in glob.glob(os.path.join(INCOMING_DIR, "*")) if os.path.isfile(p)
    )
    if not files:
        print("No new recordings.")
        return

    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            text = process_file(path)
            out_path = os.path.join(TRANSCRIPTS_DIR, f"{base}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Wrote {out_path}")
            os.remove(path)
        except Exception as e:  # noqa: BLE001 - keep going on other files
            print(f"FAILED on {path}: {e}", file=sys.stderr)
            err_path = os.path.join(TRANSCRIPTS_DIR, f"{base}.ERROR.txt")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(f"Processing failed: {e}\n")


if __name__ == "__main__":
    main()