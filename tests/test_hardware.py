from __future__ import annotations

from whisper_batch_subtitles.hardware import GPUInfo, HardwareProfile, format_hardware_report, recommend_runtime


def profile_with_gpu(memory_mb: int) -> HardwareProfile:
    return HardwareProfile(
        cpu_cores=16,
        ram_gb=64.0,
        gpus=[GPUInfo(index=0, name="Test GPU", memory_mb=memory_mb, backend="cuda")],
        cuda_available=True,
    )


def test_recommend_runtime_scales_model_with_vram():
    assert recommend_runtime(profile_with_gpu(24000)).model == "large-v3"
    assert recommend_runtime(profile_with_gpu(11000)).model == "medium"
    assert recommend_runtime(profile_with_gpu(7000)).model == "small"
    assert recommend_runtime(profile_with_gpu(4000)).model == "base"


def test_recommend_runtime_gpu_uses_cuda_device_and_float16():
    recommendation = recommend_runtime(profile_with_gpu(24000))
    assert recommendation.device == "cuda"
    assert recommendation.compute_type == "float16"


def test_recommend_runtime_cpu_only_fallback():
    profile = HardwareProfile(cpu_cores=8, ram_gb=16.0, gpus=[], cuda_available=False)
    recommendation = recommend_runtime(profile)
    assert recommendation.device == "cpu"
    assert recommendation.compute_type == "int8"
    assert recommendation.model == "small"
    assert recommendation.batch_size == 1


def test_recommend_runtime_one_transcription_worker_per_gpu():
    profile = HardwareProfile(
        cpu_cores=32,
        ram_gb=128.0,
        gpus=[
            GPUInfo(index=0, name="GPU 0", memory_mb=24000, backend="cuda"),
            GPUInfo(index=1, name="GPU 1", memory_mb=24000, backend="cuda"),
            GPUInfo(index=2, name="GPU 2", memory_mb=24000, backend="cuda"),
        ],
        cuda_available=True,
    )
    recommendation = recommend_runtime(profile)
    assert recommendation.transcription_workers == 3


def test_recommend_runtime_worker_counts_are_at_least_one():
    profile = HardwareProfile(cpu_cores=1, ram_gb=4.0, gpus=[], cuda_available=False)
    recommendation = recommend_runtime(profile)
    assert recommendation.ffmpeg_workers >= 1
    assert recommendation.transcription_workers >= 1
    assert recommendation.translation_workers >= 1


def test_format_hardware_report_includes_gpu_and_recommendation_lines():
    profile = profile_with_gpu(8000)
    recommendation = recommend_runtime(profile)
    report = format_hardware_report(profile, recommendation)
    assert "Test GPU" in report
    assert "Recommended runtime:" in report
    assert f"model: {recommendation.model}" in report


def test_format_hardware_report_no_gpus_detected():
    profile = HardwareProfile(cpu_cores=4, ram_gb=None, gpus=[], cuda_available=False)
    recommendation = recommend_runtime(profile)
    report = format_hardware_report(profile, recommendation)
    assert "GPUs: none detected" in report
    assert "RAM: unknown" in report
