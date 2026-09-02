from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from dataclasses import dataclass, field


@dataclass(slots=True)
class GPUInfo:
    index: int
    name: str
    memory_mb: int
    backend: str


@dataclass(slots=True)
class HardwareProfile:
    cpu_cores: int
    ram_gb: float | None
    gpus: list[GPUInfo] = field(default_factory=list)
    cuda_available: bool = False
    rocm_available: bool = False


@dataclass(slots=True)
class RuntimeRecommendation:
    model: str
    device: str
    compute_type: str
    ffmpeg_workers: int
    transcription_workers: int
    translation_workers: int
    ffmpeg_threads_per_worker: int
    batch_size: int
    chunk_length: int
    vad_filter: bool


def detect_hardware() -> HardwareProfile:
    gpus = _detect_nvidia_gpus()
    cuda_available = bool(gpus)
    rocm_gpus = [] if gpus else _detect_rocm_gpus()
    if rocm_gpus:
        gpus = rocm_gpus

    return HardwareProfile(
        cpu_cores=max(os.cpu_count() or 1, 1),
        ram_gb=_detect_total_ram_gb(),
        gpus=gpus,
        cuda_available=cuda_available,
        rocm_available=bool(rocm_gpus),
    )


def recommend_runtime(profile: HardwareProfile) -> RuntimeRecommendation:
    if profile.gpus:
        primary_gpu = max(profile.gpus, key=lambda item: item.memory_mb)
        vram_mb = primary_gpu.memory_mb
        if vram_mb >= 20000:
            model = "large-v3"
            batch_size = 16
        elif vram_mb >= 10000:
            model = "medium"
            batch_size = 12
        elif vram_mb >= 6000:
            model = "small"
            batch_size = 8
        else:
            model = "base"
            batch_size = 4

        ffmpeg_workers = min(max(profile.cpu_cores // 4, 1), 4)
        translation_workers = min(max(profile.cpu_cores // 3, 1), 8)
        # One transcription worker per detected GPU -- Transcriber now routes each worker
        # to a distinct CUDA device_index (see pipeline.py), so this is no longer just a
        # hint: it's the actual worker-to-GPU assignment count.
        transcription_workers = max(len(profile.gpus), 1)
        ffmpeg_threads_per_worker = max(profile.cpu_cores // max(ffmpeg_workers, 1), 1)

        return RuntimeRecommendation(
            model=model,
            device="cuda" if profile.cuda_available else "auto",
            compute_type="float16" if profile.cuda_available else "int8",
            ffmpeg_workers=ffmpeg_workers,
            transcription_workers=transcription_workers,
            translation_workers=translation_workers,
            ffmpeg_threads_per_worker=ffmpeg_threads_per_worker,
            batch_size=batch_size,
            chunk_length=30,
            vad_filter=True,
        )

    cpu_cores = max(profile.cpu_cores, 1)
    ffmpeg_workers = min(max(cpu_cores // 3, 1), 4)
    return RuntimeRecommendation(
        model="small",
        device="cpu",
        compute_type="int8",
        ffmpeg_workers=ffmpeg_workers,
        transcription_workers=min(max(cpu_cores // 4, 1), 2),
        translation_workers=min(max(cpu_cores // 3, 1), 6),
        ffmpeg_threads_per_worker=max(cpu_cores // max(ffmpeg_workers, 1), 1),
        batch_size=1,
        chunk_length=20,
        vad_filter=True,
    )


def format_hardware_report(
    profile: HardwareProfile, recommendation: RuntimeRecommendation
) -> str:
    lines = [
        f"CPU cores: {profile.cpu_cores}",
        f"RAM: {profile.ram_gb:.1f} GB" if profile.ram_gb is not None else "RAM: unknown",
        f"CUDA available: {'yes' if profile.cuda_available else 'no'}",
        f"ROCm available: {'yes' if profile.rocm_available else 'no'}",
    ]

    if profile.gpus:
        lines.append("GPUs:")
        for gpu in profile.gpus:
            lines.append(f"  - [{gpu.backend}] GPU {gpu.index}: {gpu.name} ({gpu.memory_mb} MB)")
    else:
        lines.append("GPUs: none detected")

    lines.extend(
        [
            "",
            "Recommended runtime:",
            f"  model: {recommendation.model}",
            f"  device: {recommendation.device}",
            f"  compute_type: {recommendation.compute_type}",
            f"  ffmpeg_workers: {recommendation.ffmpeg_workers}",
            f"  ffmpeg_threads_per_worker: {recommendation.ffmpeg_threads_per_worker}",
            f"  transcription_workers: {recommendation.transcription_workers}",
            f"  translation_workers: {recommendation.translation_workers}",
            f"  batch_size: {recommendation.batch_size}",
            f"  chunk_length: {recommendation.chunk_length}",
            f"  vad_filter: {recommendation.vad_filter}",
        ]
    )
    return "\n".join(lines)


def _detect_nvidia_gpus() -> list[GPUInfo]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return []
    if completed.returncode != 0:
        return []

    gpus: list[GPUInfo] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpus.append(
                GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_mb=int(parts[2]),
                    backend="cuda",
                )
            )
        except ValueError:
            continue
    return gpus


def _detect_rocm_gpus() -> list[GPUInfo]:
    command = ["rocm-smi", "--showproductname", "--showmeminfo", "vram"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return []
    if completed.returncode != 0:
        return []

    gpus: list[GPUInfo] = []
    current_name = None
    current_index = 0
    for line in completed.stdout.splitlines():
        line = line.strip()
        if "GPU[" in line and "Card series:" in line:
            current_name = line.split("Card series:", 1)[1].strip()
        if "GPU[" in line and "Total Memory (B):" in line and current_name:
            try:
                memory_bytes = int(line.split("Total Memory (B):", 1)[1].strip())
            except ValueError:
                continue
            gpus.append(
                GPUInfo(
                    index=current_index,
                    name=current_name,
                    memory_mb=memory_bytes // (1024 * 1024),
                    backend="rocm",
                )
            )
            current_index += 1
    return gpus


def _detect_total_ram_gb() -> float | None:
    system = platform.system().lower()
    if system == "windows":
        return _detect_windows_ram_gb()

    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            page_count = os.sysconf("SC_PHYS_PAGES")
            return (page_size * page_count) / (1024**3)
        except (OSError, ValueError):
            return None
    return None


def _detect_windows_ram_gb() -> float | None:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    memory_status = MEMORYSTATUSEX()
    memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
        return None
    return memory_status.ullTotalPhys / (1024**3)
