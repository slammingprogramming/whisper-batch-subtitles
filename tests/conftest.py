from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

requires_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not available on PATH")


@pytest.fixture(scope="session")
def synthetic_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not available on PATH")
    directory = tmp_path_factory.mktemp("media")
    output_path = directory / "tone.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_path.as_posix(),
            "-loglevel",
            "error",
        ],
        check=True,
    )
    return output_path
