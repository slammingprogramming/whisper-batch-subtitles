from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from whisper_batch_subtitles.benchmark import (
    BenchmarkResult,
    format_benchmark_report,
    generate_benchmark_audio,
    suggest_adjusted_batch_size,
)

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
requires_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not available on PATH")


def make_result(audio_seconds: float, wall_seconds: float, batch_size: int = 8) -> BenchmarkResult:
    return BenchmarkResult(
        model="small",
        device="cuda",
        compute_type="float16",
        batch_size=batch_size,
        audio_seconds=audio_seconds,
        wall_seconds=wall_seconds,
    )


def test_realtime_factor_faster_than_realtime():
    result = make_result(audio_seconds=20.0, wall_seconds=4.0)
    assert result.realtime_factor == 5.0


def test_realtime_factor_slower_than_realtime():
    result = make_result(audio_seconds=20.0, wall_seconds=40.0)
    assert result.realtime_factor == 0.5


def test_realtime_factor_handles_zero_wall_seconds():
    result = make_result(audio_seconds=20.0, wall_seconds=0.0)
    assert result.realtime_factor == 0.0


def test_suggest_adjusted_batch_size_halves_when_struggling():
    result = make_result(audio_seconds=20.0, wall_seconds=40.0, batch_size=8)
    assert suggest_adjusted_batch_size(result, 8) == 4


def test_suggest_adjusted_batch_size_floor_is_one():
    result = make_result(audio_seconds=20.0, wall_seconds=40.0, batch_size=1)
    assert suggest_adjusted_batch_size(result, 1) == 1


def test_suggest_adjusted_batch_size_unchanged_when_keeping_up():
    result = make_result(audio_seconds=20.0, wall_seconds=4.0, batch_size=8)
    assert suggest_adjusted_batch_size(result, 8) == 8


def test_format_benchmark_report_includes_key_numbers():
    result = make_result(audio_seconds=20.0, wall_seconds=4.0, batch_size=8)
    report = format_benchmark_report(result)
    assert "small" in report
    assert "cuda" in report
    assert "5.0x" in report
    assert "faster" in report


def test_format_benchmark_report_includes_suggestion_when_struggling():
    result = make_result(audio_seconds=20.0, wall_seconds=40.0, batch_size=8)
    report = format_benchmark_report(result, current_batch_size=8)
    assert "suggestion" in report
    assert "--batch-size 4" in report


def test_format_benchmark_report_no_suggestion_when_not_provided():
    result = make_result(audio_seconds=20.0, wall_seconds=40.0, batch_size=8)
    report = format_benchmark_report(result)
    assert "suggestion" not in report


@requires_ffmpeg
def test_generate_benchmark_audio_creates_a_wav_file(tmp_path: Path):
    output_path = tmp_path / "benchmark.wav"
    generate_benchmark_audio(output_path, 1.0)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


@requires_ffmpeg
def test_generate_benchmark_audio_raises_clear_error_for_bad_duration(tmp_path: Path):
    output_path = tmp_path / "benchmark.wav"
    with pytest.raises(RuntimeError, match="Failed to generate benchmark audio"):
        generate_benchmark_audio(output_path, -5.0)
