from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from whisper_batch_subtitles.models import Job


class StateStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(database_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize_schema()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def load_track_profile(
        self, directory_path: Path, stream_signature: str, root_dir: Path
    ) -> list[dict[str, Any]] | None:
        search_paths = [directory_path, *directory_path.parents]
        root_resolved = root_dir.resolve()
        with self._lock:
            for candidate in search_paths:
                try:
                    candidate.resolve().relative_to(root_resolved)
                except ValueError:
                    continue
                row = self._connection.execute(
                    """
                    SELECT assignments_json
                    FROM track_profiles
                    WHERE directory_path = ? AND stream_signature = ?
                    """,
                    (str(candidate), stream_signature),
                ).fetchone()
                if row is not None:
                    return json.loads(str(row["assignments_json"]))
        return None

    def save_track_profile(
        self,
        directory_path: Path,
        stream_signature: str,
        assignments: list[dict[str, Any]],
    ) -> None:
        now = _utc_now()
        assignments_json = json.dumps(assignments, ensure_ascii=True)
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO track_profiles (directory_path, stream_signature, assignments_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(directory_path, stream_signature) DO UPDATE SET
                    assignments_json = excluded.assignments_json,
                    updated_at = excluded.updated_at
                """,
                (str(directory_path), stream_signature, assignments_json, now, now),
            )
            self._connection.commit()

    def fetch_job(self, media_path: Path) -> sqlite3.Row | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM media_jobs WHERE media_path = ?",
                (str(media_path),),
            ).fetchone()
        return row

    def upsert_discovered_job(
        self, job: Job, *, model_name: str, device: str, compute_type: str
    ) -> sqlite3.Row | None:
        existing = self.fetch_job(job.media_path)
        now = _utc_now()
        translated_paths_json = json.dumps(
            {language: str(path) for language, path in job.translated_srt_paths.items()},
            ensure_ascii=True,
        )

        with self._lock:
            if existing is None:
                self._connection.execute(
                    """
                    INSERT INTO media_jobs (
                        media_path, relative_path, fingerprint, file_size, mtime_ns,
                        status, stage, attempts, last_error, duration_seconds, language,
                        model_name, device, compute_type, source_srt_path, translated_srt_paths,
                        audio_cache_path, transcript_cache_path, translation_cache_dir,
                        discovered_at, started_at, completed_at, updated_at,
                        last_runtime_seconds, segments_count, content_signature
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(job.media_path),
                        job.relative_path,
                        job.fingerprint,
                        job.file_size,
                        job.mtime_ns,
                        "queued",
                        "discover",
                        0,
                        None,
                        None,
                        None,
                        model_name,
                        device,
                        compute_type,
                        str(job.source_srt_path),
                        translated_paths_json,
                        str(job.audio_cache_path),
                        str(job.transcript_cache_path),
                        str(job.translation_cache_dir),
                        now,
                        None,
                        None,
                        now,
                        None,
                        0,
                        job.content_signature,
                    ),
                )
            else:
                attempts = 0 if existing["fingerprint"] != job.fingerprint else int(existing["attempts"])
                status = existing["status"] if existing["fingerprint"] == job.fingerprint else "queued"
                stage = existing["stage"] if existing["fingerprint"] == job.fingerprint else "discover"
                last_error = None if existing["fingerprint"] != job.fingerprint else existing["last_error"]
                self._connection.execute(
                    """
                    UPDATE media_jobs
                    SET relative_path = ?, fingerprint = ?, file_size = ?, mtime_ns = ?, status = ?, stage = ?,
                        attempts = ?, last_error = ?, model_name = ?, device = ?, compute_type = ?,
                        source_srt_path = ?, translated_srt_paths = ?, audio_cache_path = ?,
                        transcript_cache_path = ?, translation_cache_dir = ?, updated_at = ?, content_signature = ?
                    WHERE media_path = ?
                    """,
                    (
                        job.relative_path,
                        job.fingerprint,
                        job.file_size,
                        job.mtime_ns,
                        status,
                        stage,
                        attempts,
                        last_error,
                        model_name,
                        device,
                        compute_type,
                        str(job.source_srt_path),
                        translated_paths_json,
                        str(job.audio_cache_path),
                        str(job.transcript_cache_path),
                        str(job.translation_cache_dir),
                        now,
                        job.content_signature,
                        str(job.media_path),
                    ),
                )

            self._connection.commit()
        return existing

    def mark_skipped(self, job: Job, reason: str) -> None:
        self._update_status(
            job,
            status="skipped",
            stage="skip",
            last_error=reason,
        )

    def mark_running(self, job: Job, stage: str) -> None:
        now = _utc_now()
        with self._lock:
            self._connection.execute(
                """
                UPDATE media_jobs
                SET status = ?, stage = ?, started_at = COALESCE(started_at, ?), updated_at = ?
                WHERE media_path = ?
                """,
                ("running", stage, now, now, str(job.media_path)),
            )
            self._connection.commit()

    def mark_failure(self, job: Job, stage: str, error: str, attempts: int) -> None:
        self._update_status(
            job,
            status="failed",
            stage=stage,
            attempts=attempts,
            last_error=error,
        )

    def mark_completed(self, job: Job, runtime_seconds: float) -> None:
        now = _utc_now()
        translated_paths_json = json.dumps(
            {language: str(path) for language, path in job.translated_srt_paths.items()},
            ensure_ascii=True,
        )
        with self._lock:
            self._connection.execute(
                """
                UPDATE media_jobs
                SET status = ?, stage = ?, completed_at = ?, updated_at = ?, duration_seconds = ?,
                    language = ?, source_srt_path = ?, translated_srt_paths = ?,
                    audio_cache_path = ?, transcript_cache_path = ?, translation_cache_dir = ?,
                    last_runtime_seconds = ?, segments_count = ?, last_error = NULL
                WHERE media_path = ?
                """,
                (
                    "completed",
                    "done",
                    now,
                    now,
                    job.duration_seconds,
                    job.language,
                    str(job.source_srt_path),
                    translated_paths_json,
                    str(job.audio_cache_path),
                    str(job.transcript_cache_path),
                    str(job.translation_cache_dir),
                    runtime_seconds,
                    len(job.transcript_segments),
                    str(job.media_path),
                ),
            )
            self._connection.commit()

    def find_duplicate(self, content_signature: str, exclude_media_path: Path) -> sqlite3.Row | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM media_jobs
                WHERE content_signature = ? AND media_path != ? AND status = 'completed'
                LIMIT 1
                """,
                (content_signature, str(exclude_media_path)),
            ).fetchone()
        return row

    def status_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT status, COUNT(*) AS count FROM media_jobs GROUP BY status"
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def _update_status(
        self,
        job: Job,
        *,
        status: str,
        stage: str,
        attempts: int | None = None,
        last_error: str | None = None,
    ) -> None:
        now = _utc_now()
        with self._lock:
            self._connection.execute(
                """
                UPDATE media_jobs
                SET status = ?, stage = ?, attempts = COALESCE(?, attempts), last_error = ?, updated_at = ?
                WHERE media_path = ?
                """,
                (status, stage, attempts, last_error, now, str(job.media_path)),
            )
            self._connection.commit()

    def _initialize_schema(self) -> None:
        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS media_jobs (
                    media_path TEXT PRIMARY KEY,
                    relative_path TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    duration_seconds REAL,
                    language TEXT,
                    model_name TEXT NOT NULL,
                    device TEXT NOT NULL,
                    compute_type TEXT NOT NULL,
                    source_srt_path TEXT NOT NULL,
                    translated_srt_paths TEXT NOT NULL,
                    audio_cache_path TEXT NOT NULL,
                    transcript_cache_path TEXT NOT NULL,
                    translation_cache_dir TEXT NOT NULL,
                    discovered_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    updated_at TEXT NOT NULL,
                    last_runtime_seconds REAL,
                    segments_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._migrate_add_column("media_jobs", "content_signature", "TEXT")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS track_profiles (
                    directory_path TEXT NOT NULL,
                    stream_signature TEXT NOT NULL,
                    assignments_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (directory_path, stream_signature)
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_jobs_content_signature ON media_jobs(content_signature)"
            )
            self._connection.commit()

    def _migrate_add_column(self, table: str, column: str, column_type: str) -> None:
        existing_columns = {
            str(row["name"]) for row in self._connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing_columns:
            self._connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
