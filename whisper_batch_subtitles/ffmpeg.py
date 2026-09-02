from __future__ import annotations

import subprocess
from pathlib import Path


def extract_audio(
    media_path: Path,
    audio_path: Path,
    *,
    ffmpeg_threads: int,
    sample_rate: int,
    codec: str,
    overwrite: bool,
    stream_index: int | None = None,
) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-threads",
        str(max(ffmpeg_threads, 1)),
        "-i",
        str(media_path),
    ]
    if stream_index is not None:
        command.extend(["-map", f"0:{stream_index}"])
    command.extend(
        [
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-vn",
        "-acodec",
        codec,
        str(audio_path),
        "-loglevel",
        "error",
        ]
    )
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        detail = stderr or f"exit code {completed.returncode}"
        raise RuntimeError(f"ffmpeg failed for {media_path}: {detail}")


def probe_duration(media_path: Path) -> float | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None

    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None
