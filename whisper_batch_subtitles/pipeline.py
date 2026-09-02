from __future__ import annotations

import builtins
import hashlib
import json
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from faster_whisper import BatchedInferencePipeline, WhisperModel

from whisper_batch_subtitles.config import AppConfig
from whisper_batch_subtitles.constants import SUPPORTED_MEDIA
from whisper_batch_subtitles.diarization import create_diarizer
from whisper_batch_subtitles.ffmpeg import extract_audio, probe_duration
from whisper_batch_subtitles.media import (
    ROLE_LABELS,
    TRACK_ROLES,
    build_stream_signature,
    compute_content_signature,
    describe_stream,
    guess_track_assignments,
    merge_role_segments,
    normalize_track_assignments,
    probe_audio_streams,
    serialize_track_assignments,
)
from whisper_batch_subtitles.models import Job, TrackAssignment
from whisper_batch_subtitles.scaling import DynamicWorkerPool, decide_scale_action
from whisper_batch_subtitles.state import StateStore
from whisper_batch_subtitles.subtitles import normalize_segments, suppress_repeated_segments, write_srt
from whisper_batch_subtitles.translators import create_translation_engine

STOP = object()

_PERMANENT_ERROR_PATTERNS = (
    "invalid data found",
    "moov atom not found",
    "could not find codec parameters",
    "does not contain any stream",
    "no such file or directory",
    "all audio tracks are assigned role 'ignore'",
)


def classify_stage_error(error: Exception) -> str:
    """Classify a stage-worker exception as "permanent" (retrying won't help -- a
    corrupt file, a missing stream, our own "nothing to do" guard) or "transient"
    (worth the existing retry-with-backoff behavior: network blips, transient locks,
    momentary GPU contention)."""
    message = str(error).lower()
    for pattern in _PERMANENT_ERROR_PATTERNS:
        if pattern in message:
            return "permanent"
    return "transient"


@dataclass(slots=True)
class PipelineMetrics:
    discovered: int = 0
    queued: int = 0
    skipped: int = 0
    extracted: int = 0
    transcribed: int = 0
    translated: int = 0
    completed: int = 0
    failed: int = 0
    start_time: float = field(default_factory=time.monotonic)
    current_stage_files: dict[str, str] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(self, field_name: str, amount: int = 1) -> None:
        with self.lock:
            setattr(self, field_name, getattr(self, field_name) + amount)

    def set_current(self, stage: str, relative_path: str | None) -> None:
        with self.lock:
            if relative_path is None:
                self.current_stage_files.pop(stage, None)
            else:
                self.current_stage_files[stage] = relative_path

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            data = {
                "discovered": self.discovered,
                "queued": self.queued,
                "skipped": self.skipped,
                "extracted": self.extracted,
                "transcribed": self.transcribed,
                "translated": self.translated,
                "completed": self.completed,
                "failed": self.failed,
                "current_stage_files": dict(self.current_stage_files),
            }
        data["elapsed_seconds"] = max(time.monotonic() - self.start_time, 0.001)
        return data


def format_progress_line(snapshot: dict[str, Any], queue_parts: list[str]) -> str:
    completed = snapshot["completed"]
    failed = snapshot["failed"]
    queued = snapshot["queued"]
    skipped = snapshot["skipped"]
    elapsed_hours = snapshot["elapsed_seconds"] / 3600
    throughput = completed / elapsed_hours if elapsed_hours > 0 else 0.0
    remaining = max(queued - completed - failed, 0)
    eta_hours = (remaining / throughput) if throughput > 0 else None
    eta_text = f"{eta_hours:.2f}h" if eta_hours is not None else "unknown"
    return (
        f"discovered={snapshot['discovered']} queued={queued} completed={completed} "
        f"skipped={skipped} failed={failed} throughput={throughput:.2f} files/hour eta={eta_text} "
        f"queues[{', '.join(queue_parts)}]"
    )


class ProgressReporter(threading.Thread):
    def __init__(
        self,
        *,
        metrics: PipelineMetrics,
        extract_queue: queue.Queue[Job | object],
        transcribe_queue: queue.Queue[Job | object],
        translate_queue: queue.Queue[Job | object] | None,
        write_queue: queue.Queue[Job | object],
        interval_seconds: float,
        logger: logging.Logger,
        live_status: bool = False,
        tui: bool = False,
    ) -> None:
        super().__init__(daemon=True)
        self.metrics = metrics
        self.extract_queue = extract_queue
        self.transcribe_queue = transcribe_queue
        self.translate_queue = translate_queue
        self.write_queue = write_queue
        self.interval_seconds = interval_seconds
        self.logger = logger
        self.live_status = live_status
        self.tui = tui
        self.stop_event = threading.Event()

    def _queue_depths(self) -> dict[str, int]:
        depths = {
            "extract": self.extract_queue.qsize(),
            "transcribe": self.transcribe_queue.qsize(),
        }
        if self.translate_queue is not None:
            depths["translate"] = self.translate_queue.qsize()
        depths["write"] = self.write_queue.qsize()
        return depths

    def run(self) -> None:
        if self.tui:
            self._run_tui()
            return

        wrote_live_line = False
        while not self.stop_event.wait(self.interval_seconds):
            snapshot = self.metrics.snapshot()
            queue_parts = [f"{stage}={depth}" for stage, depth in self._queue_depths().items()]
            line = format_progress_line(snapshot, queue_parts)

            if self.live_status:
                # \x1b[K clears any leftover characters from a longer previous line.
                # Requires a VT100-capable terminal (modern Windows Terminal/PowerShell
                # qualify); on a terminal without VT support this just prints the escape
                # code literally, which is why it's opt-in rather than the default.
                sys.stdout.write("\r" + line + "\x1b[K")
                sys.stdout.flush()
                wrote_live_line = True
            else:
                self.logger.info("Progress | %s", line)

        if wrote_live_line:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _run_tui(self) -> None:
        from rich.live import Live

        from whisper_batch_subtitles.tui import build_dashboard

        def render() -> Any:
            snapshot = self.metrics.snapshot()
            return build_dashboard(snapshot, self._queue_depths(), snapshot["current_stage_files"])

        refresh_seconds = min(self.interval_seconds, 1.0)
        with Live(render(), refresh_per_second=4) as live:
            while not self.stop_event.wait(refresh_seconds):
                live.update(render())


class Transcriber:
    def __init__(self, config: AppConfig, *, device_index: int | None = None) -> None:
        self.config = config
        model_kwargs: dict[str, Any] = {
            "device": config.device or "cpu",
            "compute_type": config.compute_type or "int8",
        }
        if device_index is not None:
            model_kwargs["device_index"] = device_index
        self.model = WhisperModel(config.model or "small", **model_kwargs)
        self.pipeline = BatchedInferencePipeline(self.model) if (config.batch_size or 1) > 1 else None

    def transcribe(
        self, audio_path: Path, *, language: str | None
    ) -> tuple[list[dict[str, Any]], str | None]:
        common_kwargs = {
            "language": language,
            "beam_size": self.config.beam_size,
            "vad_filter": self.config.vad_filter,
            "chunk_length": self.config.chunk_length,
        }
        if self.pipeline is not None:
            segments_iter, info = self.pipeline.transcribe(
                str(audio_path),
                batch_size=max(self.config.batch_size or 1, 1),
                **common_kwargs,
            )
        else:
            segments_iter, info = self.model.transcribe(str(audio_path), **common_kwargs)

        segments = [
            {"start": float(segment.start), "end": float(segment.end), "text": segment.text}
            for segment in segments_iter
        ]
        return normalize_segments(segments), language or getattr(info, "language", None)


class PipelineRunner:
    def __init__(self, config: AppConfig, state: StateStore) -> None:
        self.config = config
        self.state = state
        self.logger = logging.getLogger("whisper_batch_subtitles.pipeline")
        self.metrics = PipelineMetrics()
        self.extract_queue: queue.Queue[Job | object] = queue.Queue(maxsize=config.queue_size)
        self.transcribe_queue: queue.Queue[Job | object] = queue.Queue(maxsize=config.queue_size)
        self.translate_queue: queue.Queue[Job | object] | None = (
            queue.Queue(maxsize=config.queue_size) if config.effective_target_languages else None
        )
        if self.translate_queue is not None:
            # Fail fast on translation backend misconfiguration (e.g. missing DeepL key)
            # before any pipeline stage starts, rather than crashing a worker thread mid-run.
            create_translation_engine(config)
        self.write_queue: queue.Queue[Job | object] = queue.Queue(maxsize=config.queue_size)
        self.diarizer = create_diarizer(
            mode=config.diarization_mode,
            device=config.device or "cpu",
            model_name=config.pyannote_model,
            auth_token=config.pyannote_auth_token,
            external_python=config.diarization_external_python,
            external_timeout_seconds=config.diarization_external_timeout_seconds,
            logger=self.logger,
        )
        self.diarization_variant = (
            f"{config.diarization_mode}-"
            f"{config.pyannote_model or config.diarization_external_python or 'default'}"
        )

    def run(self) -> int:
        self.config.ensure_directories()
        pending_jobs = self._discover_jobs()
        if not pending_jobs:
            self.logger.info("No queued work after discovery.")
            return 0

        reporter = ProgressReporter(
            metrics=self.metrics,
            extract_queue=self.extract_queue,
            transcribe_queue=self.transcribe_queue,
            translate_queue=self.translate_queue,
            write_queue=self.write_queue,
            interval_seconds=self.config.progress_interval_seconds,
            logger=self.logger,
            live_status=self.config.live_status and not self.config.tui,
            tui=self.config.tui,
        )
        reporter.start()

        extraction_pool: DynamicWorkerPool | None = None
        extraction_scaler_stop: threading.Event | None = None
        extraction_scaler_thread: threading.Thread | None = None
        extraction_workers: list[threading.Thread] = []

        if self.config.dynamic_ffmpeg_workers:
            base_count = max(self.config.ffmpeg_workers or 1, 1)
            min_workers = max(self.config.ffmpeg_workers_min or 1, 1)
            max_workers = max(self.config.ffmpeg_workers_max or base_count, min_workers)
            extraction_pool = DynamicWorkerPool(
                target=self._extraction_worker,
                name_prefix="extract",
                min_workers=min_workers,
                max_workers=max_workers,
            )
            extraction_pool.start_initial(base_count)
            extraction_scaler_stop = threading.Event()
            extraction_scaler_thread = threading.Thread(
                target=self._dynamic_extraction_scaler,
                args=(extraction_pool, extraction_scaler_stop),
                name="extract-scaler",
                daemon=True,
            )
            extraction_scaler_thread.start()
        else:
            extraction_workers = self._start_workers(
                count=max(self.config.ffmpeg_workers or 1, 1),
                target=self._extraction_worker,
                name_prefix="extract",
            )

        transcription_workers = self._start_workers(
            count=max(self.config.transcription_workers or 1, 1),
            target=self._transcription_worker,
            name_prefix="transcribe",
            pass_index=True,
        )
        translation_workers = self._start_workers(
            count=max(self.config.translation_workers or 1, 1) if self.translate_queue is not None else 0,
            target=self._translation_worker,
            name_prefix="translate",
        )
        writer_workers = self._start_workers(count=1, target=self._writer_worker, name_prefix="write")

        for job, stage_name in pending_jobs:
            if stage_name == "transcribe":
                self.transcribe_queue.put(job)
            else:
                self.extract_queue.put(job)

        if extraction_pool is not None:
            assert extraction_scaler_stop is not None and extraction_scaler_thread is not None
            extraction_scaler_stop.set()
            extraction_scaler_thread.join(timeout=2)
            extraction_pool.join_all(self.extract_queue, STOP)
        else:
            for _ in extraction_workers:
                self.extract_queue.put(STOP)
            for worker in extraction_workers:
                worker.join()

        for _ in transcription_workers:
            self.transcribe_queue.put(STOP)
        for worker in transcription_workers:
            worker.join()

        if self.translate_queue is not None:
            for _ in translation_workers:
                self.translate_queue.put(STOP)
            for worker in translation_workers:
                worker.join()

        self.write_queue.put(STOP)
        for worker in writer_workers:
            worker.join()

        reporter.stop_event.set()
        reporter.join(timeout=1)

        snapshot = self.metrics.snapshot()
        self.logger.info(
            "Run finished | discovered=%s queued=%s completed=%s skipped=%s failed=%s",
            snapshot["discovered"],
            snapshot["queued"],
            snapshot["completed"],
            snapshot["skipped"],
            snapshot["failed"],
        )
        return snapshot["failed"]

    def _start_workers(
        self, *, count: int, target: Any, name_prefix: str, pass_index: bool = False
    ) -> list[threading.Thread]:
        workers: list[threading.Thread] = []
        for index in range(count):
            args = (index,) if pass_index else ()
            worker = threading.Thread(target=target, args=args, name=f"{name_prefix}-{index}", daemon=True)
            worker.start()
            workers.append(worker)
        return workers

    def _device_index_for_worker(self, worker_index: int) -> int | None:
        indices = self.config.gpu_device_indices
        if not indices or (self.config.device or "").lower() != "cuda":
            return None
        return indices[worker_index % len(indices)]

    def _discover_jobs(self) -> list[tuple[Job, str]]:
        media_paths = self._collect_media_paths()
        self.logger.info("Discovered %s candidate media files under %s", len(media_paths), self.config.root_dir)
        pending_jobs: list[tuple[Job, str]] = []

        for media_path in media_paths:
            try:
                job = self._build_job(media_path)
            except Exception as error:
                self.logger.error("Skipping %s: failed to inspect during discovery (%s)", media_path, error)
                self.metrics.increment("discovered")
                self.metrics.increment("failed")
                continue
            existing = self.state.upsert_discovered_job(
                job,
                model_name=self.config.model or "small",
                device=self.config.device or "cpu",
                compute_type=self.config.compute_type or "int8",
            )
            self.metrics.increment("discovered")

            if job.content_signature and self.config.duplicate_detection != "off":
                duplicate_row = self.state.find_duplicate(job.content_signature, job.media_path)
                if duplicate_row is not None:
                    self.logger.warning(
                        "%s appears to be a duplicate of already-processed %s (content signature match)",
                        job.relative_path,
                        duplicate_row["relative_path"],
                    )
                    if self.config.duplicate_detection == "skip":
                        self.metrics.increment("skipped")
                        self.state.mark_skipped(job, f"duplicate of {duplicate_row['relative_path']}")
                        continue

            if self._should_skip_job(job, existing):
                self.metrics.increment("skipped")
                self.state.mark_skipped(job, "existing outputs already satisfy current run")
                continue

            self.metrics.increment("queued")
            if self.config.resume and not self.config.overwrite and self._has_resume_artifacts(job):
                pending_jobs.append((job, "transcribe"))
            else:
                pending_jobs.append((job, "extract"))

        return pending_jobs

    def _collect_media_paths(self) -> list[Path]:
        media_paths = [
            path for path in self.config.root_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA
        ]
        if self.config.scan_order == "newest":
            media_paths.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
        elif self.config.scan_order == "oldest":
            media_paths.sort(key=lambda path: path.stat().st_mtime_ns)
        elif self.config.scan_order == "largest":
            media_paths.sort(key=lambda path: path.stat().st_size, reverse=True)
        elif self.config.scan_order == "smallest":
            media_paths.sort(key=lambda path: path.stat().st_size)
        else:
            media_paths.sort()
        return media_paths

    def _build_job(self, media_path: Path) -> Job:
        stat = media_path.stat()
        relative_path = str(media_path.relative_to(self.config.root_dir))
        fingerprint = hashlib.sha1(
            f"{media_path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()
        source_srt_path = media_path.with_suffix(".srt")
        translated_srt_paths = {
            language: media_path.with_suffix(f".{language}.srt") for language in self.config.effective_target_languages
        }
        audio_streams = probe_audio_streams(media_path)
        track_assignments = self._resolve_track_assignments(media_path, audio_streams)
        content_signature = None
        if self.config.duplicate_detection != "off" and stat.st_size > 0:
            content_signature = compute_content_signature(media_path, stat.st_size)
        return Job(
            media_path=media_path,
            relative_path=relative_path,
            fingerprint=fingerprint,
            file_size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            source_srt_path=source_srt_path,
            translated_srt_paths=translated_srt_paths,
            audio_cache_path=self.config.audio_cache_dir / f"{fingerprint}.wav",
            transcript_cache_path=self.config.transcript_cache_dir / f"{fingerprint}.json",
            translation_cache_dir=self.config.translation_cache_dir / fingerprint,
            audio_streams=audio_streams,
            track_assignments=track_assignments,
            content_signature=content_signature,
        )

    def _resolve_track_assignments(
        self, media_path: Path, audio_streams: list[Any]
    ) -> list[TrackAssignment]:
        if not audio_streams:
            return []
        if len(audio_streams) == 1:
            return guess_track_assignments(audio_streams)

        signature = build_stream_signature(audio_streams)
        stored = self.state.load_track_profile(media_path.parent, signature, self.config.root_dir)
        if stored is not None:
            assignments = normalize_track_assignments(audio_streams, stored)
            if assignments:
                self.logger.info("Applied saved track profile for %s", media_path.parent)
                return assignments

        guessed = guess_track_assignments(audio_streams)
        if self.config.prompt_for_track_roles and sys.stdin.isatty():
            assignments = self._prompt_for_track_assignments(media_path, audio_streams, guessed)
        else:
            assignments = guessed
            self.logger.info(
                "No saved profile for multi-track media %s; using heuristic track assignments.",
                media_path.name,
            )

        self.state.save_track_profile(media_path.parent, signature, serialize_track_assignments(assignments))
        return assignments

    def _prompt_for_track_assignments(
        self, media_path: Path, audio_streams: list[Any], guessed: list[TrackAssignment]
    ) -> list[TrackAssignment]:
        guessed_by_index = {assignment.stream_index: assignment for assignment in guessed}
        self.logger.warning("Detected multiple audio tracks for %s", media_path)
        print()
        print(f"Multiple audio tracks detected for {media_path.name}")
        print(f"Saved answers will be reused for {media_path.parent} when the same layout is seen again.")
        print("Available roles: me, others, mixed, system, ignore")
        print()

        assignments: list[TrackAssignment] = []
        for stream in audio_streams:
            guessed_role = guessed_by_index.get(stream.index, TrackAssignment(stream.index, "ignore", "Ignore")).role
            while True:
                print(f"  {describe_stream(stream)}")
                raw = builtins.input(f"    role [{guessed_role}]: ").strip().lower()
                role = raw or guessed_role
                if role in TRACK_ROLES:
                    assignments.append(
                        TrackAssignment(
                            stream_index=stream.index,
                            role=role,
                            label=ROLE_LABELS[role],
                            stream_title=stream.title,
                        )
                    )
                    break
                print("    Invalid role. Choose one of: me, others, mixed, system, ignore")
        print()
        return assignments

    def _should_skip_job(self, job: Job, existing: Any) -> bool:
        if self.config.overwrite:
            return False

        record_language = str(existing["language"]).lower() if existing and existing["language"] else None
        if existing and existing["fingerprint"] == job.fingerprint and existing["status"] == "completed":
            if self._outputs_complete(job, record_language):
                return True

        if self.config.skip_existing and job.source_srt_path.exists():
            if not self.config.effective_target_languages:
                return True
            if record_language is not None and self._outputs_complete(job, record_language):
                return True
            if any(path.exists() for path in job.translated_srt_paths.values()):
                return True
        return False

    def _outputs_complete(self, job: Job, detected_language: str | None) -> bool:
        if not job.source_srt_path.exists():
            return False
        for language, path in job.translated_srt_paths.items():
            if detected_language is not None and language == detected_language:
                continue
            if not path.exists():
                return False
        if self.config.write_role_subtitles and len(job.audio_streams) > 1:
            for assignment in job.track_assignments:
                if assignment.role == "ignore":
                    continue
                if not job.role_srt_path(assignment.role).exists():
                    return False
        return True

    def _has_resume_artifacts(self, job: Job) -> bool:
        if job.transcript_cache_path.exists() or job.audio_cache_path.exists():
            return True
        for assignment in job.track_assignments:
            if assignment.role == "ignore":
                continue
            if job.track_audio_cache_path(assignment.role, assignment.stream_index).exists():
                return True
            if job.track_transcript_cache_path(assignment.role, assignment.stream_index).exists():
                return True
        return False

    def _active_assignments(self, job: Job) -> list[TrackAssignment]:
        return [assignment for assignment in job.track_assignments if assignment.role != "ignore"]

    def _dynamic_extraction_scaler(self, pool: DynamicWorkerPool, stop_event: threading.Event) -> None:
        # Deliberately only the extraction stage: ffmpeg subprocess workers are cheap
        # to spin up/down. Transcription workers stay statically sized since each one
        # loads a real model onto a specific GPU device_index at startup.
        #
        # Self-correcting by design: if two shrink ticks fire back-to-back before a
        # worker has actually retired (a real but narrow race, since decide_scale_action
        # only requests a shrink when the queue is already empty), the pool could
        # transiently dip below min_workers. The next tick's `live_workers < min_workers`
        # check unconditionally grows it back, so this heals within one check interval
        # rather than needing perfect atomic accounting here.
        check_interval = max(min(self.config.progress_interval_seconds, 5.0), 1.0)
        while not stop_event.wait(check_interval):
            action = decide_scale_action(
                queue_depth=self.extract_queue.qsize(),
                downstream_depth=self.transcribe_queue.qsize(),
                live_workers=pool.live_count(),
                min_workers=pool.min_workers,
                max_workers=pool.max_workers,
            )
            if action == "grow":
                if pool.grow():
                    self.logger.info("Dynamic scaling: grew extraction workers to %s", pool.live_count())
            elif action == "shrink":
                self.extract_queue.put(STOP)
                self.logger.info(
                    "Dynamic scaling: requested extraction worker retirement (currently %s)", pool.live_count()
                )

    def _extraction_worker(self) -> None:
        while True:
            item = self.extract_queue.get()
            try:
                if item is STOP:
                    return
                job = item
                self.metrics.set_current("extract", job.relative_path)
                self.state.mark_running(job, "extract")

                active_assignments = self._active_assignments(job)
                if len(job.audio_streams) > 1 and not active_assignments:
                    raise RuntimeError(
                        "All audio tracks are assigned role 'ignore'; nothing to transcribe. "
                        "Fix the saved track profile or re-run inspect-tracks."
                    )
                if len(job.audio_streams) <= 1:
                    if not (self.config.resume and job.audio_cache_path.exists() and not self.config.overwrite):
                        extract_audio(
                            job.media_path,
                            job.audio_cache_path,
                            ffmpeg_threads=max(self.config.ffmpeg_threads_per_worker or 1, 1),
                            sample_rate=self.config.sample_rate,
                            codec=self.config.audio_codec,
                            overwrite=True,
                        )
                else:
                    for assignment in active_assignments:
                        track_audio_path = job.track_audio_cache_path(assignment.role, assignment.stream_index)
                        if self.config.resume and track_audio_path.exists() and not self.config.overwrite:
                            continue
                        extract_audio(
                            job.media_path,
                            track_audio_path,
                            ffmpeg_threads=max(self.config.ffmpeg_threads_per_worker or 1, 1),
                            sample_rate=self.config.sample_rate,
                            codec=self.config.audio_codec,
                            overwrite=True,
                            stream_index=assignment.stream_index,
                        )

                if job.duration_seconds is None:
                    job.duration_seconds = probe_duration(job.media_path)
                self.metrics.increment("extracted")
                self.transcribe_queue.put(job)
            except Exception as error:
                self._handle_stage_error(job, "extract", error, self.extract_queue)
            finally:
                self.metrics.set_current("extract", None)
                self.extract_queue.task_done()

    def _transcription_worker(self, worker_index: int = 0) -> None:
        transcriber = Transcriber(self.config, device_index=self._device_index_for_worker(worker_index))
        while True:
            item = self.transcribe_queue.get()
            try:
                if item is STOP:
                    return
                job = item
                self.metrics.set_current("transcribe", job.relative_path)
                self.state.mark_running(job, "transcribe")

                cached_transcript = None
                if self.config.resume and job.transcript_cache_path.exists() and not self.config.overwrite:
                    cached_transcript = self._load_json(job.transcript_cache_path)

                if cached_transcript is not None:
                    self._restore_cached_transcript(job, cached_transcript)
                else:
                    self._transcribe_job(job, transcriber)
                    self._save_json(
                        job.transcript_cache_path,
                        {
                            "fingerprint": job.fingerprint,
                            "language": job.language,
                            "segments": job.transcript_segments,
                            "track_transcripts": job.track_transcripts,
                            "track_assignments": serialize_track_assignments(job.track_assignments),
                        },
                    )

                if job.duration_seconds is None:
                    job.duration_seconds = probe_duration(job.media_path)

                self.metrics.increment("transcribed")
                if self.translate_queue is None:
                    self.write_queue.put(job)
                else:
                    self.translate_queue.put(job)
            except Exception as error:
                self._handle_stage_error(job, "transcribe", error, self.transcribe_queue)
            finally:
                self.metrics.set_current("transcribe", None)
                self.transcribe_queue.task_done()

    def _transcribe_job(self, job: Job, transcriber: Transcriber) -> None:
        active_assignments = self._active_assignments(job)
        if len(job.audio_streams) <= 1:
            segments, detected_language = transcriber.transcribe(job.audio_cache_path, language=self.config.language)
            if self.config.diarization_mode != "off":
                segments = self.diarizer.annotate(
                    job.audio_cache_path,
                    segments,
                    cache_path=job.diarization_cache_path(self.diarization_variant),
                )
            if not self.config.speaker_labels:
                segments = self._strip_speaker_metadata(segments)
            job.language = (detected_language or self.config.language or "unknown").lower()
            job.transcript_segments = normalize_segments(segments)
            job.track_transcripts = {}
            return

        merged_role_segments: dict[str, list[dict[str, Any]]] = {}
        detected_languages: list[str] = []
        for assignment in active_assignments:
            track_audio_path = job.track_audio_cache_path(assignment.role, assignment.stream_index)
            cache_path = job.track_transcript_cache_path(assignment.role, assignment.stream_index)
            cached_track = None
            if self.config.resume and cache_path.exists() and not self.config.overwrite:
                cached_track = self._load_json(cache_path)

            if cached_track is not None:
                track_segments = normalize_segments(cached_track.get("segments", []))
                detected_language = str(cached_track.get("language") or "")
            else:
                track_segments, detected_language = transcriber.transcribe(
                    track_audio_path,
                    language=self.config.language,
                )
                self._save_json(
                    cache_path,
                    {
                        "fingerprint": job.fingerprint,
                        "role": assignment.role,
                        "stream_index": assignment.stream_index,
                        "language": detected_language or self.config.language,
                        "segments": track_segments,
                    },
                )

            labeled_segments = []
            for segment in track_segments:
                labeled_segment = dict(segment)
                if self.config.speaker_labels:
                    labeled_segment["speaker"] = assignment.role
                    labeled_segment["speaker_label"] = assignment.label
                labeled_segments.append(labeled_segment)
            merged_role_segments.setdefault(assignment.role, []).extend(normalize_segments(labeled_segments))
            if detected_language:
                detected_languages.append(str(detected_language).lower())

        for role_segments in merged_role_segments.values():
            role_segments.sort(key=lambda segment: (float(segment["start"]), float(segment["end"])))

        job.track_transcripts = merged_role_segments
        job.transcript_segments = normalize_segments(
            merge_role_segments(
                merged_role_segments,
                labels={assignment.role: assignment.label for assignment in active_assignments},
            )
        )
        job.language = detected_languages[0] if detected_languages else (self.config.language or "unknown").lower()

    def _translation_worker(self) -> None:
        if self.translate_queue is None:
            return

        translator = create_translation_engine(self.config)
        while True:
            item = self.translate_queue.get()
            try:
                if item is STOP:
                    return
                job = item
                self.metrics.set_current("translate", job.relative_path)
                self.state.mark_running(job, "translate")

                if not job.transcript_segments:
                    cached_transcript = self._load_json(job.transcript_cache_path)
                    if cached_transcript is None:
                        raise RuntimeError("Missing transcript cache before translation stage")
                    self._restore_cached_transcript(job, cached_transcript)

                for target_language in self.config.effective_target_languages:
                    if job.language and target_language == job.language.lower():
                        continue
                    cache_path = job.translation_cache_path(target_language)
                    if self.config.resume and cache_path.exists() and not self.config.overwrite:
                        cached_translation = self._load_json(cache_path)
                        if cached_translation is not None:
                            job.translations[target_language] = normalize_segments(cached_translation.get("segments", []))
                            continue

                    translated_segments = translator.translate_segments(job.transcript_segments, target_language)
                    job.translations[target_language] = normalize_segments(translated_segments)
                    self._save_json(
                        cache_path,
                        {
                            "fingerprint": job.fingerprint,
                            "language": job.language,
                            "target_language": target_language,
                            "segments": job.translations[target_language],
                        },
                    )

                self.metrics.increment("translated")
                self.write_queue.put(job)
            except Exception as error:
                self._handle_stage_error(job, "translate", error, self.translate_queue)
            finally:
                self.metrics.set_current("translate", None)
                self.translate_queue.task_done()

    def _writer_worker(self) -> None:
        while True:
            item = self.write_queue.get()
            started_at = time.monotonic()
            try:
                if item is STOP:
                    return
                job = item
                self.metrics.set_current("write", job.relative_path)
                self.state.mark_running(job, "write")

                if not job.transcript_segments:
                    cached_transcript = self._load_json(job.transcript_cache_path)
                    if cached_transcript is None:
                        raise RuntimeError("Missing transcript cache before write stage")
                    self._restore_cached_transcript(job, cached_transcript)

                self._write_srt(job.transcript_segments, job.source_srt_path)

                if self.config.write_role_subtitles and job.track_transcripts:
                    for role, segments in job.track_transcripts.items():
                        self._write_srt(segments, job.role_srt_path(role))

                for target_language, output_path in job.translated_srt_paths.items():
                    if job.language and target_language == job.language.lower():
                        continue
                    translated_segments = job.translations.get(target_language)
                    if translated_segments is None:
                        cached_translation = self._load_json(job.translation_cache_path(target_language))
                        if cached_translation is None:
                            raise RuntimeError(
                                f"Missing translation cache for target language {target_language}"
                            )
                        translated_segments = normalize_segments(cached_translation.get("segments", []))
                        job.translations[target_language] = translated_segments
                    self._write_srt(translated_segments, output_path)

                runtime_seconds = time.monotonic() - started_at
                self.state.mark_completed(job, runtime_seconds=runtime_seconds)
                self.metrics.increment("completed")
            except Exception as error:
                self._handle_stage_error(job, "write", error, self.write_queue)
            finally:
                self.metrics.set_current("write", None)
                self.write_queue.task_done()

    def _write_srt(self, segments: list[dict[str, Any]], output_path: Path) -> None:
        # Single choke point for every SRT write, so suppression/formatting behave the
        # same whether segments came from a fresh run or a cache/resume restore.
        if self.config.suppress_repeated_segments:
            segments = suppress_repeated_segments(segments)
        write_srt(
            segments,
            output_path,
            cleanup_text=self.config.subtitle_cleanup_text,
            max_line_chars=self.config.subtitle_max_line_chars,
            max_lines=self.config.subtitle_max_lines,
        )

    def _restore_cached_transcript(self, job: Job, cached_transcript: dict[str, Any]) -> None:
        job.language = cached_transcript.get("language")
        job.transcript_segments = normalize_segments(cached_transcript.get("segments", []))
        raw_track_transcripts = cached_transcript.get("track_transcripts") or {}
        job.track_transcripts = {
            str(role): normalize_segments(segments) for role, segments in raw_track_transcripts.items()
        }
        raw_assignments = cached_transcript.get("track_assignments")
        if raw_assignments:
            job.track_assignments = normalize_track_assignments(job.audio_streams, raw_assignments)

    def _strip_speaker_metadata(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stripped_segments: list[dict[str, Any]] = []
        for segment in segments:
            stripped_segment = dict(segment)
            stripped_segment.pop("speaker", None)
            stripped_segment.pop("speaker_label", None)
            stripped_segments.append(stripped_segment)
        return stripped_segments

    def _handle_stage_error(
        self, job: Job, stage: str, error: Exception, stage_queue: queue.Queue[Job | object]
    ) -> None:
        attempts = job.stage_attempts.get(stage, 0) + 1
        job.stage_attempts[stage] = attempts
        error_message = f"{type(error).__name__}: {error}"
        classification = classify_stage_error(error)

        if classification == "transient" and attempts <= self.config.max_retries:
            self.logger.warning(
                "Retrying %s stage for %s (%s/%s): %s",
                stage,
                job.relative_path,
                attempts,
                self.config.max_retries,
                error_message,
            )
            time.sleep(min(attempts, 3))
            stage_queue.put(job)
            return

        self.logger.exception(
            "Stage %s failed for %s (%s, attempt %s/%s)",
            stage,
            job.relative_path,
            classification,
            attempts,
            self.config.max_retries,
        )
        self.metrics.increment("failed")
        self.state.mark_failure(job, stage, error_message, attempts)

    def _load_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
