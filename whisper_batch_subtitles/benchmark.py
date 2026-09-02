from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class BenchmarkResult:
    model: str
    device: str
    compute_type: str
    batch_size: int
    audio_seconds: float
    wall_seconds: float

    @property
    def realtime_factor(self) -> float:
        if self.wall_seconds <= 0:
            return 0.0
        return self.audio_seconds / self.wall_seconds


def generate_benchmark_audio(output_path: Path, duration_seconds: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration_seconds}",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
        "-loglevel",
        "error",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"Failed to generate benchmark audio: {stderr or 'unknown ffmpeg error'}")


def run_benchmark(
    *,
    model_name: str,
    device: str,
    compute_type: str,
    batch_size: int = 1,
    audio_duration_seconds: float = 20.0,
) -> BenchmarkResult:
    """Measure actual model throughput on this machine.

    Generates a synthetic tone (not speech -- content doesn't matter for a throughput
    measurement) and transcribes it with VAD disabled, so the model processes the full
    clip regardless of content. This measures raw inference throughput on the current
    hardware, not how much of a real recording would get skipped as silence -- the
    static heuristics in hardware.py::recommend_runtime pick a starting point; this is
    how you check whether that starting point actually holds up on this machine.
    """
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    with tempfile.TemporaryDirectory(prefix="wbs-benchmark-") as tmp_dir:
        audio_path = Path(tmp_dir) / "benchmark.wav"
        generate_benchmark_audio(audio_path, audio_duration_seconds)

        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        pipeline = BatchedInferencePipeline(model) if batch_size > 1 else None

        started_at = time.monotonic()
        if pipeline is not None:
            segments, _info = pipeline.transcribe(str(audio_path), batch_size=batch_size, vad_filter=False)
        else:
            segments, _info = model.transcribe(str(audio_path), vad_filter=False)
        list(segments)  # faster-whisper transcribes lazily -- force full consumption to time it honestly
        wall_seconds = time.monotonic() - started_at

    return BenchmarkResult(
        model=model_name,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        audio_seconds=audio_duration_seconds,
        wall_seconds=wall_seconds,
    )


def suggest_adjusted_batch_size(result: BenchmarkResult, current_batch_size: int) -> int:
    """If the benchmark shows the model struggling to keep up with realtime, suggest
    halving the batch size (down to a floor of 1); otherwise leave it as recommended.
    Deliberately conservative -- a low-confidence heuristic nudge, not an automatic
    override, since a single short synthetic clip is a noisy sample."""
    if result.realtime_factor < 1.0 and current_batch_size > 1:
        return max(current_batch_size // 2, 1)
    return current_batch_size


def format_benchmark_report(result: BenchmarkResult, *, current_batch_size: int | None = None) -> str:
    lines = [
        f"Benchmark ({result.model}, {result.device}, {result.compute_type}, batch_size={result.batch_size}):",
        f"  {result.audio_seconds:.1f}s of synthetic audio processed in {result.wall_seconds:.2f}s wall time",
        f"  realtime factor: {result.realtime_factor:.1f}x "
        f"({'faster' if result.realtime_factor >= 1 else 'slower'} than realtime)",
    ]
    if current_batch_size is not None:
        suggested = suggest_adjusted_batch_size(result, current_batch_size)
        if suggested != current_batch_size:
            lines.append(
                f"  suggestion: this hardware struggled to keep up at batch_size={current_batch_size}; "
                f"try --batch-size {suggested}"
            )
    return "\n".join(lines)
