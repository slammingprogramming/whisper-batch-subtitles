from __future__ import annotations

from pathlib import Path

from whisper_batch_subtitles.models import Job
from whisper_batch_subtitles.state import StateStore


def make_job(tmp_path: Path, name: str = "video.mkv", fingerprint: str = "fp1") -> Job:
    return Job(
        media_path=tmp_path / name,
        relative_path=name,
        fingerprint=fingerprint,
        file_size=100,
        mtime_ns=1,
        source_srt_path=tmp_path / f"{Path(name).stem}.srt",
        translated_srt_paths={},
        audio_cache_path=tmp_path / "cache" / f"{fingerprint}.wav",
        transcript_cache_path=tmp_path / "cache" / f"{fingerprint}.json",
        translation_cache_dir=tmp_path / "cache" / fingerprint,
    )


def test_upsert_discovered_job_inserts_new_row(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    try:
        job = make_job(tmp_path)
        existing = store.upsert_discovered_job(job, model_name="small", device="cpu", compute_type="int8")
        assert existing is None
        row = store.fetch_job(job.media_path)
        assert row["status"] == "queued"
        assert row["fingerprint"] == "fp1"
    finally:
        store.close()


def test_upsert_same_fingerprint_preserves_status_and_attempts(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    try:
        job = make_job(tmp_path)
        store.upsert_discovered_job(job, model_name="small", device="cpu", compute_type="int8")
        store.mark_failure(job, "extract", "boom", attempts=2)

        existing = store.upsert_discovered_job(job, model_name="small", device="cpu", compute_type="int8")
        assert existing["status"] == "failed"
        assert existing["attempts"] == 2

        row = store.fetch_job(job.media_path)
        assert row["status"] == "failed"
        assert row["attempts"] == 2
    finally:
        store.close()


def test_upsert_changed_fingerprint_resets_status(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    try:
        job = make_job(tmp_path, fingerprint="fp1")
        store.upsert_discovered_job(job, model_name="small", device="cpu", compute_type="int8")
        store.mark_failure(job, "extract", "boom", attempts=3)

        changed_job = make_job(tmp_path, fingerprint="fp2")
        existing = store.upsert_discovered_job(changed_job, model_name="small", device="cpu", compute_type="int8")
        assert existing["fingerprint"] == "fp1"

        row = store.fetch_job(changed_job.media_path)
        assert row["fingerprint"] == "fp2"
        assert row["status"] == "queued"
        assert row["attempts"] == 0
        assert row["last_error"] is None
    finally:
        store.close()


def test_mark_completed_records_language_and_duration(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    try:
        job = make_job(tmp_path)
        store.upsert_discovered_job(job, model_name="small", device="cpu", compute_type="int8")
        job.language = "en"
        job.duration_seconds = 12.5
        job.transcript_segments = [{"start": 0.0, "end": 1.0, "text": "hi"}]
        store.mark_completed(job, runtime_seconds=3.2)

        row = store.fetch_job(job.media_path)
        assert row["status"] == "completed"
        assert row["language"] == "en"
        assert row["duration_seconds"] == 12.5
        assert row["segments_count"] == 1
        assert row["last_error"] is None
    finally:
        store.close()


def test_status_counts_aggregates_by_status(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    try:
        job_a = make_job(tmp_path, name="a.mkv", fingerprint="a")
        job_b = make_job(tmp_path, name="b.mkv", fingerprint="b")
        store.upsert_discovered_job(job_a, model_name="small", device="cpu", compute_type="int8")
        store.upsert_discovered_job(job_b, model_name="small", device="cpu", compute_type="int8")
        store.mark_skipped(job_b, "already present")

        counts = store.status_counts()
        assert counts["queued"] == 1
        assert counts["skipped"] == 1
    finally:
        store.close()


def test_track_profile_round_trip(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    try:
        root_dir = tmp_path / "process"
        directory = root_dir / "season1"
        directory.mkdir(parents=True)
        assignments = [{"stream_index": 0, "role": "me", "label": "Me", "stream_title": None}]

        assert store.load_track_profile(directory, "sig-a", root_dir) is None
        store.save_track_profile(directory, "sig-a", assignments)

        loaded = store.load_track_profile(directory, "sig-a", root_dir)
        assert loaded == assignments

        # A different signature in the same directory is a distinct profile.
        assert store.load_track_profile(directory, "sig-b", root_dir) is None
    finally:
        store.close()
