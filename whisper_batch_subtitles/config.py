from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from whisper_batch_subtitles.constants import DEFAULT_CONFIG_FILE, DEFAULT_STATE_DIR
from whisper_batch_subtitles.hardware import HardwareProfile, RuntimeRecommendation


@dataclass(slots=True)
class AppConfig:
    root_dir: Path = Path("process")
    state_dir: Path = Path(DEFAULT_STATE_DIR)
    state_db: Path | None = None
    model: str | None = None
    device: str | None = None
    compute_type: str | None = None
    language: str | None = None
    translate: bool = True
    target_languages: list[str] = field(default_factory=lambda: ["en"])
    diarization_mode: str = "auto"
    speaker_labels: bool = True
    write_role_subtitles: bool = True
    prompt_for_track_roles: bool = True
    pyannote_model: str | None = None
    pyannote_auth_token: str | None = None
    diarization_external_python: str | None = None
    diarization_external_timeout_seconds: int = 1800
    ffmpeg_workers: int | None = None
    dynamic_ffmpeg_workers: bool = False
    ffmpeg_workers_min: int | None = None
    ffmpeg_workers_max: int | None = None
    transcription_workers: int | None = None
    translation_workers: int | None = None
    ffmpeg_threads_per_worker: int | None = None
    batch_size: int | None = None
    chunk_length: int = 30
    queue_size: int = 8
    skip_existing: bool = True
    overwrite: bool = False
    resume: bool = True
    max_retries: int = 2
    vad_filter: bool = True
    beam_size: int = 5
    sample_rate: int = 16000
    audio_codec: str = "pcm_s16le"
    log_level: str = "INFO"
    progress_interval_seconds: float = 10.0
    scan_order: str = "path"
    gpu_device_indices: list[int] = field(default_factory=list)
    subtitle_cleanup_text: bool = True
    subtitle_max_line_chars: int = 42
    subtitle_max_lines: int = 2
    suppress_repeated_segments: bool = True
    duplicate_detection: str = "warn"
    translation_backend: str = "google"
    deepl_api_key: str | None = None
    watch: bool = False
    watch_interval_seconds: float = 300.0
    live_status: bool = False
    tui: bool = False
    text_log_path: Path | None = None
    json_log_path: Path | None = None

    @property
    def cache_dir(self) -> Path:
        return self.state_dir / "cache"

    @property
    def audio_cache_dir(self) -> Path:
        return self.cache_dir / "audio"

    @property
    def transcript_cache_dir(self) -> Path:
        return self.cache_dir / "transcripts"

    @property
    def translation_cache_dir(self) -> Path:
        return self.cache_dir / "translations"

    @property
    def effective_target_languages(self) -> list[str]:
        if not self.translate:
            return []
        return list(dict.fromkeys(language.lower() for language in self.target_languages))

    def ensure_directories(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_cache_dir.mkdir(parents=True, exist_ok=True)
        self.translation_cache_dir.mkdir(parents=True, exist_ok=True)
        if self.state_db is not None:
            self.state_db.parent.mkdir(parents=True, exist_ok=True)
        if self.text_log_path is not None:
            self.text_log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.json_log_path is not None:
            self.json_log_path.parent.mkdir(parents=True, exist_ok=True)


def build_config(
    cli_values: dict[str, Any], recommendation: RuntimeRecommendation | None = None
) -> AppConfig:
    config_path = _resolve_config_path(cli_values.get("config"))
    file_values = _load_config_file(config_path) if config_path is not None else {}
    profile_values = _load_profile_values(cli_values.get("profile"))
    env_values = _load_env_overrides()
    merged = _merge_dicts(_default_config_values(), file_values, profile_values, env_values, cli_values)
    config = _config_from_mapping(merged)
    if recommendation is not None:
        _apply_recommendation(config, recommendation)
    _finalize_config(config)
    return config


def profiles_dir() -> Path:
    return Path("profiles")


def profile_path(name: str) -> Path:
    return profiles_dir() / f"{name}.yaml"


def _load_profile_values(profile_name: Any) -> dict[str, Any]:
    if not profile_name:
        return {}
    path = profile_path(str(profile_name))
    if not path.exists():
        raise ValueError(f"Profile not found: {path}")
    return _load_config_file(path)


def config_to_yaml_text(config: AppConfig) -> str:
    """Serialize an already-built, effective AppConfig back to YAML -- used by
    `--save-profile` to snapshot the resolved runtime settings as a reusable profile.

    Deliberately excludes secrets (`pyannote_auth_token`, `deepl_api_key`) and
    installation-specific paths (`state_dir`, log paths) -- a profile is meant to be a
    portable, shareable "job shape," not a full config dump. Set secrets via the main
    config file or environment variables instead."""
    ordered_values = {
        "root_dir": str(config.root_dir),
        "model": config.model,
        "device": config.device,
        "compute_type": config.compute_type,
        "language": config.language,
        "translate": config.translate,
        "target_languages": config.target_languages,
        "diarization_mode": config.diarization_mode,
        "speaker_labels": config.speaker_labels,
        "write_role_subtitles": config.write_role_subtitles,
        "prompt_for_track_roles": config.prompt_for_track_roles,
        "pyannote_model": config.pyannote_model,
        "diarization_external_python": config.diarization_external_python,
        "diarization_external_timeout_seconds": config.diarization_external_timeout_seconds,
        "gpu_device_indices": config.gpu_device_indices,
        "ffmpeg_workers": config.ffmpeg_workers,
        "dynamic_ffmpeg_workers": config.dynamic_ffmpeg_workers,
        "ffmpeg_workers_min": config.ffmpeg_workers_min,
        "ffmpeg_workers_max": config.ffmpeg_workers_max,
        "transcription_workers": config.transcription_workers,
        "translation_workers": config.translation_workers,
        "ffmpeg_threads_per_worker": config.ffmpeg_threads_per_worker,
        "batch_size": config.batch_size,
        "chunk_length": config.chunk_length,
        "queue_size": config.queue_size,
        "skip_existing": config.skip_existing,
        "resume": config.resume,
        "max_retries": config.max_retries,
        "vad_filter": config.vad_filter,
        "beam_size": config.beam_size,
        "sample_rate": config.sample_rate,
        "audio_codec": config.audio_codec,
        "subtitle_cleanup_text": config.subtitle_cleanup_text,
        "subtitle_max_line_chars": config.subtitle_max_line_chars,
        "subtitle_max_lines": config.subtitle_max_lines,
        "suppress_repeated_segments": config.suppress_repeated_segments,
        "duplicate_detection": config.duplicate_detection,
        "translation_backend": config.translation_backend,
        "log_level": config.log_level,
        "progress_interval_seconds": config.progress_interval_seconds,
        "scan_order": config.scan_order,
        "live_status": config.live_status,
        "tui": config.tui,
        "watch": config.watch,
        "watch_interval_seconds": config.watch_interval_seconds,
    }
    return yaml.safe_dump(ordered_values, sort_keys=False, default_flow_style=False, allow_unicode=False)


def default_config_path() -> Path:
    return Path(DEFAULT_CONFIG_FILE)


def sample_config_text(
    recommendation: RuntimeRecommendation | None = None,
    profile: HardwareProfile | None = None,
    overrides: dict[str, Any] | None = None,
) -> str:
    config_values = _default_config_values()
    if recommendation is not None:
        config_values.update(
            {
                "model": recommendation.model,
                "device": recommendation.device,
                "compute_type": recommendation.compute_type,
                "ffmpeg_workers": recommendation.ffmpeg_workers,
                "transcription_workers": recommendation.transcription_workers,
                "translation_workers": recommendation.translation_workers,
                "ffmpeg_threads_per_worker": recommendation.ffmpeg_threads_per_worker,
                "batch_size": recommendation.batch_size,
                "chunk_length": recommendation.chunk_length,
                "vad_filter": recommendation.vad_filter,
            }
        )
    else:
        config_values.update(
            {
                "model": None,
                "device": None,
                "compute_type": None,
                "ffmpeg_workers": None,
                "transcription_workers": None,
                "translation_workers": None,
                "ffmpeg_threads_per_worker": None,
                "batch_size": None,
            }
        )
    if profile is not None and profile.gpus and config_values.get("device") == "cuda":
        config_values["gpu_device_indices"] = [gpu.index for gpu in profile.gpus]
    if overrides:
        config_values.update(overrides)

    ordered_values = {
        "root_dir": config_values["root_dir"],
        "state_dir": config_values["state_dir"],
        "model": config_values["model"],
        "device": config_values["device"],
        "compute_type": config_values["compute_type"],
        "language": None,
        "translate": config_values["translate"],
        "target_languages": config_values["target_languages"],
        "diarization_mode": config_values["diarization_mode"],
        "speaker_labels": config_values["speaker_labels"],
        "write_role_subtitles": config_values["write_role_subtitles"],
        "prompt_for_track_roles": config_values["prompt_for_track_roles"],
        "pyannote_model": config_values["pyannote_model"],
        "pyannote_auth_token": config_values["pyannote_auth_token"],
        "diarization_external_python": config_values["diarization_external_python"],
        "diarization_external_timeout_seconds": config_values["diarization_external_timeout_seconds"],
        "gpu_device_indices": config_values["gpu_device_indices"],
        "ffmpeg_workers": config_values["ffmpeg_workers"],
        "dynamic_ffmpeg_workers": config_values["dynamic_ffmpeg_workers"],
        "ffmpeg_workers_min": config_values["ffmpeg_workers_min"],
        "ffmpeg_workers_max": config_values["ffmpeg_workers_max"],
        "transcription_workers": config_values["transcription_workers"],
        "translation_workers": config_values["translation_workers"],
        "ffmpeg_threads_per_worker": config_values["ffmpeg_threads_per_worker"],
        "batch_size": config_values["batch_size"],
        "chunk_length": config_values["chunk_length"],
        "queue_size": config_values["queue_size"],
        "skip_existing": config_values["skip_existing"],
        "overwrite": config_values["overwrite"],
        "resume": config_values["resume"],
        "max_retries": config_values["max_retries"],
        "vad_filter": config_values["vad_filter"],
        "beam_size": config_values["beam_size"],
        "sample_rate": config_values["sample_rate"],
        "audio_codec": config_values["audio_codec"],
        "subtitle_cleanup_text": config_values["subtitle_cleanup_text"],
        "subtitle_max_line_chars": config_values["subtitle_max_line_chars"],
        "subtitle_max_lines": config_values["subtitle_max_lines"],
        "suppress_repeated_segments": config_values["suppress_repeated_segments"],
        "duplicate_detection": config_values["duplicate_detection"],
        "translation_backend": config_values["translation_backend"],
        "deepl_api_key": config_values["deepl_api_key"],
        "log_level": config_values["log_level"],
        "progress_interval_seconds": config_values["progress_interval_seconds"],
        "scan_order": config_values["scan_order"],
        "watch": config_values["watch"],
        "watch_interval_seconds": config_values["watch_interval_seconds"],
        "live_status": config_values["live_status"],
        "tui": config_values["tui"],
    }

    header_lines = [
        "# Whisper Batch Subtitles configuration",
        "# Generated by `python -m whisper_batch_subtitles init-config`.",
    ]
    if profile is not None:
        gpu_summary = ", ".join(
            f"{gpu.name} ({gpu.memory_mb} MB)" for gpu in profile.gpus
        ) or "none"
        ram_text = f"{profile.ram_gb:.1f} GB" if profile.ram_gb is not None else "unknown"
        header_lines.extend(
            [
                "# Detected hardware:",
                f"#   CPU cores: {profile.cpu_cores}",
                f"#   RAM: {ram_text}",
                f"#   GPUs: {gpu_summary}",
            ]
        )
    if recommendation is not None:
        header_lines.extend(
            [
                "# Recommended runtime used for tuned defaults:",
                f"#   model: {recommendation.model}",
                f"#   device: {recommendation.device}",
                f"#   compute_type: {recommendation.compute_type}",
                f"#   ffmpeg_workers: {recommendation.ffmpeg_workers}",
                f"#   transcription_workers: {recommendation.transcription_workers}",
                f"#   translation_workers: {recommendation.translation_workers}",
                f"#   ffmpeg_threads_per_worker: {recommendation.ffmpeg_threads_per_worker}",
                f"#   batch_size: {recommendation.batch_size}",
                f"#   chunk_length: {recommendation.chunk_length}",
                f"#   vad_filter: {recommendation.vad_filter}",
            ]
        )

    yaml_text = yaml.safe_dump(
        ordered_values,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )
    return "\n".join(header_lines) + "\n\n" + yaml_text


def _default_config_values() -> dict[str, Any]:
    return {
        "root_dir": "process",
        "state_dir": DEFAULT_STATE_DIR,
        "translate": True,
        "target_languages": ["en"],
        "diarization_mode": "auto",
        "speaker_labels": True,
        "write_role_subtitles": True,
        "prompt_for_track_roles": True,
        "pyannote_model": None,
        "pyannote_auth_token": None,
        "diarization_external_python": None,
        "diarization_external_timeout_seconds": 1800,
        "chunk_length": 30,
        "queue_size": 8,
        "skip_existing": True,
        "overwrite": False,
        "resume": True,
        "max_retries": 2,
        "vad_filter": True,
        "beam_size": 5,
        "sample_rate": 16000,
        "audio_codec": "pcm_s16le",
        "log_level": "INFO",
        "progress_interval_seconds": 10.0,
        "scan_order": "path",
        "gpu_device_indices": [],
        "dynamic_ffmpeg_workers": False,
        "ffmpeg_workers_min": None,
        "ffmpeg_workers_max": None,
        "subtitle_cleanup_text": True,
        "subtitle_max_line_chars": 42,
        "subtitle_max_lines": 2,
        "suppress_repeated_segments": True,
        "duplicate_detection": "warn",
        "translation_backend": "google",
        "deepl_api_key": None,
        "watch": False,
        "watch_interval_seconds": 300.0,
        "live_status": False,
        "tui": False,
    }


def _resolve_config_path(raw_path: Any) -> Path | None:
    if raw_path in (None, ""):
        candidate = default_config_path()
        return candidate if candidate.exists() else None
    return Path(str(raw_path)).expanduser()


def _load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return loaded


def _load_env_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith("WBS_"):
            continue
        field_name = key[4:].lower()
        if field_name in {"target_languages"}:
            overrides[field_name] = [part.strip() for part in value.split(",") if part.strip()]
        elif field_name in {"gpu_device_indices"}:
            overrides[field_name] = _normalize_int_list(value)
        elif field_name in {
            "translate",
            "skip_existing",
            "overwrite",
            "resume",
            "vad_filter",
            "speaker_labels",
            "write_role_subtitles",
            "prompt_for_track_roles",
            "subtitle_cleanup_text",
            "suppress_repeated_segments",
            "watch",
            "live_status",
            "tui",
            "dynamic_ffmpeg_workers",
        }:
            overrides[field_name] = _parse_bool(value)
        elif field_name in {
            "ffmpeg_workers",
            "ffmpeg_workers_min",
            "ffmpeg_workers_max",
            "transcription_workers",
            "translation_workers",
            "ffmpeg_threads_per_worker",
            "batch_size",
            "chunk_length",
            "queue_size",
            "max_retries",
            "beam_size",
            "sample_rate",
            "diarization_external_timeout_seconds",
            "subtitle_max_line_chars",
            "subtitle_max_lines",
        }:
            overrides[field_name] = int(value)
        elif field_name in {"progress_interval_seconds", "watch_interval_seconds"}:
            overrides[field_name] = float(value)
        else:
            overrides[field_name] = value
    return overrides


def _merge_dicts(*mappings: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if value is None:
                continue
            merged[key] = value
    return merged


def _config_from_mapping(mapping: dict[str, Any]) -> AppConfig:
    return AppConfig(
        root_dir=Path(str(mapping["root_dir"])).expanduser(),
        state_dir=Path(str(mapping["state_dir"])).expanduser(),
        state_db=_to_optional_path(mapping.get("state_db")),
        model=_to_optional_str(mapping.get("model")),
        device=_to_optional_str(mapping.get("device")),
        compute_type=_to_optional_str(mapping.get("compute_type")),
        language=_to_optional_str(mapping.get("language")),
        translate=bool(mapping.get("translate", True)),
        target_languages=_normalize_language_list(mapping.get("target_languages", ["en"])),
        diarization_mode=str(mapping.get("diarization_mode", "auto")).lower(),
        speaker_labels=bool(mapping.get("speaker_labels", True)),
        write_role_subtitles=bool(mapping.get("write_role_subtitles", True)),
        prompt_for_track_roles=bool(mapping.get("prompt_for_track_roles", True)),
        pyannote_model=_to_optional_str(mapping.get("pyannote_model")),
        pyannote_auth_token=_to_optional_str(mapping.get("pyannote_auth_token")),
        diarization_external_python=_to_optional_str(mapping.get("diarization_external_python")),
        diarization_external_timeout_seconds=int(mapping.get("diarization_external_timeout_seconds", 1800)),
        ffmpeg_workers=_to_optional_int(mapping.get("ffmpeg_workers")),
        dynamic_ffmpeg_workers=bool(mapping.get("dynamic_ffmpeg_workers", False)),
        ffmpeg_workers_min=_to_optional_int(mapping.get("ffmpeg_workers_min")),
        ffmpeg_workers_max=_to_optional_int(mapping.get("ffmpeg_workers_max")),
        transcription_workers=_to_optional_int(mapping.get("transcription_workers")),
        translation_workers=_to_optional_int(mapping.get("translation_workers")),
        ffmpeg_threads_per_worker=_to_optional_int(mapping.get("ffmpeg_threads_per_worker")),
        batch_size=_to_optional_int(mapping.get("batch_size")),
        chunk_length=int(mapping.get("chunk_length", 30)),
        queue_size=int(mapping.get("queue_size", 8)),
        skip_existing=bool(mapping.get("skip_existing", True)),
        overwrite=bool(mapping.get("overwrite", False)),
        resume=bool(mapping.get("resume", True)),
        max_retries=int(mapping.get("max_retries", 2)),
        vad_filter=bool(mapping.get("vad_filter", True)),
        beam_size=int(mapping.get("beam_size", 5)),
        sample_rate=int(mapping.get("sample_rate", 16000)),
        audio_codec=str(mapping.get("audio_codec", "pcm_s16le")),
        log_level=str(mapping.get("log_level", "INFO")).upper(),
        progress_interval_seconds=float(mapping.get("progress_interval_seconds", 10.0)),
        scan_order=str(mapping.get("scan_order", "path")).lower(),
        gpu_device_indices=_normalize_int_list(mapping.get("gpu_device_indices", [])),
        subtitle_cleanup_text=bool(mapping.get("subtitle_cleanup_text", True)),
        subtitle_max_line_chars=int(mapping.get("subtitle_max_line_chars", 42)),
        subtitle_max_lines=int(mapping.get("subtitle_max_lines", 2)),
        suppress_repeated_segments=bool(mapping.get("suppress_repeated_segments", True)),
        duplicate_detection=str(mapping.get("duplicate_detection", "warn")).lower(),
        translation_backend=str(mapping.get("translation_backend", "google")).lower(),
        deepl_api_key=_to_optional_str(mapping.get("deepl_api_key")),
        watch=bool(mapping.get("watch", False)),
        watch_interval_seconds=float(mapping.get("watch_interval_seconds", 300.0)),
        live_status=bool(mapping.get("live_status", False)),
        tui=bool(mapping.get("tui", False)),
        text_log_path=_to_optional_path(mapping.get("text_log_path")),
        json_log_path=_to_optional_path(mapping.get("json_log_path")),
    )


def _apply_recommendation(config: AppConfig, recommendation: RuntimeRecommendation) -> None:
    if config.model is None:
        config.model = recommendation.model
    if config.device is None:
        config.device = recommendation.device
    if config.compute_type is None:
        config.compute_type = recommendation.compute_type
    if config.ffmpeg_workers is None:
        config.ffmpeg_workers = recommendation.ffmpeg_workers
    if config.transcription_workers is None:
        config.transcription_workers = recommendation.transcription_workers
    if config.translation_workers is None:
        config.translation_workers = recommendation.translation_workers
    if config.ffmpeg_threads_per_worker is None:
        config.ffmpeg_threads_per_worker = recommendation.ffmpeg_threads_per_worker
    if config.batch_size is None:
        config.batch_size = recommendation.batch_size
    if config.chunk_length <= 0:
        config.chunk_length = recommendation.chunk_length


def _finalize_config(config: AppConfig) -> None:
    if config.state_db is None:
        config.state_db = config.state_dir / "state.sqlite3"
    if config.text_log_path is None:
        config.text_log_path = config.state_dir / "logs" / "runtime.log"
    if config.json_log_path is None:
        config.json_log_path = config.state_dir / "logs" / "events.jsonl"
    if config.overwrite:
        config.skip_existing = False
    if not config.translate:
        config.target_languages = []
    if not config.target_languages and config.translate:
        config.target_languages = ["en"]
    config.root_dir = config.root_dir.expanduser()
    config.state_dir = config.state_dir.expanduser()


def _normalize_int_list(raw_value: Any) -> list[int]:
    if isinstance(raw_value, str):
        parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    elif isinstance(raw_value, (list, tuple, set)):
        parts = [str(part).strip() for part in raw_value if str(part).strip()]
    else:
        return []
    return [int(part) for part in parts]


def _normalize_language_list(raw_value: Any) -> list[str]:
    if isinstance(raw_value, str):
        values = [part.strip() for part in raw_value.split(",") if part.strip()]
    elif isinstance(raw_value, (list, tuple, set)):
        values = [str(part).strip() for part in raw_value if str(part).strip()]
    else:
        values = ["en"]
    return [value.lower() for value in values]


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_optional_int(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    return int(value)


def _to_optional_str(value: Any) -> str | None:
    if value in (None, "", "null"):
        return None
    return str(value)


def _to_optional_path(value: Any) -> Path | None:
    if value in (None, "", "null"):
        return None
    return Path(str(value)).expanduser()
