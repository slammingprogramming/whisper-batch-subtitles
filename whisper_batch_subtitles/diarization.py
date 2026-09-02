from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any


class Diarizer:
    def annotate(
        self,
        audio_path: Path,
        segments: list[dict[str, Any]],
        *,
        cache_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        return segments


class PyannoteDiarizer(Diarizer):
    def __init__(
        self,
        *,
        model_name: str,
        auth_token: str | None,
        device: str,
        logger: logging.Logger,
    ) -> None:
        self.logger = logger
        try:
            from pyannote.audio import Pipeline  # type: ignore
            import torch  # type: ignore
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pyannote.audio is not installed. Install the optional diarization dependency to use this backend."
            ) from error

        if not auth_token:
            raise RuntimeError("PYANNOTE_AUTH_TOKEN is required for the pyannote diarization backend.")

        self._torch = torch
        self._lock = threading.Lock()
        self.pipeline = Pipeline.from_pretrained(model_name, use_auth_token=auth_token)
        if device == "cuda":
            self.pipeline.to(torch.device("cuda"))

    def annotate(
        self,
        audio_path: Path,
        segments: list[dict[str, Any]],
        *,
        cache_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        turns = self._run_or_load_turns(audio_path, cache_path=cache_path)
        return _label_segments_from_turns(segments, turns)

    def _run_or_load_turns(self, audio_path: Path, *, cache_path: Path | None) -> list[dict[str, Any]]:
        if cache_path is not None and cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return list(payload.get("turns", []))

        with self._lock:
            diarization = self.pipeline(str(audio_path))
        speaker_names: dict[str, str] = {}
        turns: list[dict[str, Any]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            label = speaker_names.setdefault(speaker, f"Speaker {len(speaker_names) + 1}")
            turns.append(
                {
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": speaker,
                    "speaker_label": label,
                }
            )

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({"audio_path": str(audio_path), "turns": turns}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return turns


class ExternalCommandDiarizer(Diarizer):
    def __init__(
        self,
        *,
        python_executable: str,
        model_name: str | None,
        auth_token: str | None,
        device: str,
        timeout_seconds: int,
        logger: logging.Logger,
    ) -> None:
        self.python_executable = python_executable
        self.model_name = model_name or "pyannote/speaker-diarization-3.1"
        self.auth_token = auth_token
        self.device = device
        self.timeout_seconds = timeout_seconds
        self.logger = logger

    def annotate(
        self,
        audio_path: Path,
        segments: list[dict[str, Any]],
        *,
        cache_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        if cache_path is None:
            raise RuntimeError("External diarization requires a cache path for JSON handoff.")
        turns = self._run_or_load_turns(audio_path, cache_path)
        return _label_segments_from_turns(segments, turns)

    def _run_or_load_turns(self, audio_path: Path, cache_path: Path) -> list[dict[str, Any]]:
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return list(payload.get("turns", []))

        command = [
            self.python_executable,
            "-m",
            "whisper_batch_subtitles.diarization_helper",
            "--audio-path",
            str(audio_path),
            "--output-path",
            str(cache_path),
            "--model",
            self.model_name,
            "--device",
            self.device,
        ]
        if self.auth_token:
            command.extend(["--auth-token", self.auth_token])

        self.logger.info("Launching external diarization helper: %s", " ".join(command[:4]) + " ...")
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(self.timeout_seconds, 1),
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            detail = stderr or stdout or f"exit code {completed.returncode}"
            raise RuntimeError(f"External diarization helper failed: {detail}")

        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return list(payload.get("turns", []))


def create_diarizer(
    *,
    mode: str,
    device: str,
    model_name: str | None,
    auth_token: str | None,
    external_python: str | None,
    external_timeout_seconds: int,
    logger: logging.Logger,
) -> Diarizer:
    normalized_mode = mode.lower()
    if normalized_mode == "off":
        return Diarizer()

    effective_token = auth_token or os.environ.get("PYANNOTE_AUTH_TOKEN")
    if normalized_mode == "external" or (
        normalized_mode == "auto" and external_python not in (None, "", "null")
    ):
        if not external_python:
            logger.warning("External diarization requested but no diarization_external_python was configured.")
            return Diarizer()
        return ExternalCommandDiarizer(
            python_executable=external_python,
            model_name=model_name,
            auth_token=effective_token,
            device=device,
            timeout_seconds=external_timeout_seconds,
            logger=logger,
        )

    if normalized_mode in {"auto", "pyannote"}:
        effective_model = model_name or "pyannote/speaker-diarization-3.1"
        try:
            return PyannoteDiarizer(
                model_name=effective_model,
                auth_token=effective_token,
                device=device,
                logger=logger,
            )
        except RuntimeError as error:
            logger.warning("Diarization backend unavailable, continuing without single-track diarization: %s", error)
            return Diarizer()

    return Diarizer()


def _label_segments_from_turns(
    segments: list[dict[str, Any]], turns: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    labeled_segments: list[dict[str, Any]] = []
    for segment in segments:
        best_turn = None
        best_overlap = 0.0
        for turn in turns:
            overlap = _segment_overlap(
                float(segment["start"]),
                float(segment["end"]),
                float(turn["start"]),
                float(turn["end"]),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = turn
        labeled_segment = dict(segment)
        if best_turn is not None:
            labeled_segment["speaker"] = str(best_turn["speaker"])
            labeled_segment["speaker_label"] = str(best_turn["speaker_label"])
        labeled_segments.append(labeled_segment)
    return labeled_segments


def _segment_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))
