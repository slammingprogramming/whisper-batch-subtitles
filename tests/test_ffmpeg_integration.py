from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from whisper_batch_subtitles.ffmpeg import extract_audio, probe_duration
from whisper_batch_subtitles.media import probe_audio_streams

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

requires_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not available on PATH")


@requires_ffmpeg
def test_probe_audio_streams_finds_generated_tone(synthetic_wav: Path):
    streams = probe_audio_streams(synthetic_wav)
    assert len(streams) == 1
    assert streams[0].channels == 1


@requires_ffmpeg
def test_probe_duration_matches_generated_length(synthetic_wav: Path):
    duration = probe_duration(synthetic_wav)
    assert duration is not None
    assert 0.9 <= duration <= 1.1


@requires_ffmpeg
def test_probe_duration_returns_none_for_missing_file(tmp_path: Path):
    assert probe_duration(tmp_path / "does-not-exist.wav") is None


@requires_ffmpeg
def test_extract_audio_round_trip(tmp_path: Path, synthetic_wav: Path):
    output_path = tmp_path / "extracted.wav"
    extract_audio(
        synthetic_wav,
        output_path,
        ffmpeg_threads=1,
        sample_rate=16000,
        codec="pcm_s16le",
        overwrite=True,
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    streams = probe_audio_streams(output_path)
    assert len(streams) == 1


@requires_ffmpeg
def test_extract_audio_raises_informative_error_for_bad_input(tmp_path: Path):
    bad_input = tmp_path / "not-real-media.mkv"
    bad_input.write_bytes(b"this is not a real media container")
    output_path = tmp_path / "should-not-exist.wav"

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        extract_audio(
            bad_input,
            output_path,
            ffmpeg_threads=1,
            sample_rate=16000,
            codec="pcm_s16le",
            overwrite=True,
        )
