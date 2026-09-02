from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AudioStreamInfo:
    index: int
    codec_name: str | None = None
    channels: int | None = None
    channel_layout: str | None = None
    language: str | None = None
    title: str | None = None
    disposition_default: bool = False


@dataclass(slots=True)
class TrackAssignment:
    stream_index: int
    role: str
    label: str
    stream_title: str | None = None


@dataclass(slots=True)
class Job:
    media_path: Path
    relative_path: str
    fingerprint: str
    file_size: int
    mtime_ns: int
    source_srt_path: Path
    translated_srt_paths: dict[str, Path]
    audio_cache_path: Path
    transcript_cache_path: Path
    translation_cache_dir: Path
    status: str = "queued"
    language: str | None = None
    duration_seconds: float | None = None
    content_signature: str | None = None
    audio_streams: list[AudioStreamInfo] = field(default_factory=list)
    track_assignments: list[TrackAssignment] = field(default_factory=list)
    transcript_segments: list[dict[str, Any]] = field(default_factory=list)
    track_transcripts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    translations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    stage_attempts: dict[str, int] = field(default_factory=dict)

    def translation_cache_path(self, target_language: str) -> Path:
        return self.translation_cache_dir / f"{target_language}.json"

    def track_audio_cache_path(self, role: str, stream_index: int) -> Path:
        return self.audio_cache_path.with_name(
            f"{self.audio_cache_path.stem}.{role}.s{stream_index}{self.audio_cache_path.suffix}"
        )

    def track_transcript_cache_path(self, role: str, stream_index: int) -> Path:
        return self.transcript_cache_path.with_name(
            f"{self.transcript_cache_path.stem}.{role}.s{stream_index}{self.transcript_cache_path.suffix}"
        )

    def diarization_cache_path(self, variant: str = "default") -> Path:
        safe_variant = re.sub(r"[^A-Za-z0-9_.-]+", "_", variant).strip("_") or "default"
        return self.transcript_cache_path.with_name(
            f"{self.transcript_cache_path.stem}.diarization.{safe_variant}{self.transcript_cache_path.suffix}"
        )

    def role_srt_path(self, role: str) -> Path:
        return self.source_srt_path.with_suffix(f".{role}.srt")
