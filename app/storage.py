from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

KEEP_RECORDINGS = 2  # how many raw recordings to retain for "Redo Last Recording"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip().lower())
    slug = slug.strip("-")
    return slug or "note"


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def save_transcript(
    transcripts_dir: Path,
    *,
    use_case: str,
    transcript: str,
    summary: str,
    backend: str,
    model: str,
    duration_s: float | None,
) -> Path:
    """Write a timestamped transcript+summary file and return its path."""
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = transcripts_dir / f"{ts}_{slugify(use_case)}.txt"

    duration_line = ""
    if duration_s is not None:
        duration_line = f"Duration: {format_duration(duration_s)}\n"

    word_count = len(transcript.split())
    filepath.write_text(
        f"Use Case: {use_case}\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Transcriber: {backend} ({model})\n"
        f"{duration_line}"
        f"Words: {word_count}\n"
        f"{'=' * 40}\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        f"{'=' * 40}\n\n"
        f"SUMMARY:\n{summary}\n",
        encoding="utf-8",
    )
    return filepath


def archive_recording(recordings_dir: Path, audio_path: str, use_case: str) -> Path | None:
    """Move the WAV into recordings/ and prune to the newest KEEP_RECORDINGS."""
    try:
        recordings_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dest = recordings_dir / f"{ts}_{slugify(use_case)}.wav"
        shutil.move(audio_path, dest)
    except Exception:
        return None

    try:
        recordings = sorted(
            recordings_dir.glob("*.wav"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in recordings[KEEP_RECORDINGS:]:
            try:
                old.unlink()
            except Exception:
                pass
    except Exception:
        pass

    return dest


def newest_recording(recordings_dir: Path) -> Path | None:
    if not recordings_dir.exists():
        return None
    wavs = sorted(
        recordings_dir.glob("*.wav"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return wavs[0] if wavs else None
