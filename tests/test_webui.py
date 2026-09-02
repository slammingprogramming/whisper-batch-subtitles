from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

from whisper_batch_subtitles.models import Job
from whisper_batch_subtitles.state import StateStore
from whisper_batch_subtitles.webui import fetch_dashboard_data, render_dashboard_html, run_dashboard_server


def make_job(tmp_path: Path, name: str, fingerprint: str) -> Job:
    return Job(
        media_path=tmp_path / name,
        relative_path=name,
        fingerprint=fingerprint,
        file_size=100,
        mtime_ns=1,
        source_srt_path=tmp_path / f"{name}.srt",
        translated_srt_paths={},
        audio_cache_path=tmp_path / "cache" / f"{fingerprint}.wav",
        transcript_cache_path=tmp_path / "cache" / f"{fingerprint}.json",
        translation_cache_dir=tmp_path / "cache" / fingerprint,
    )


def test_fetch_dashboard_data_missing_database_is_unavailable(tmp_path):
    data = fetch_dashboard_data(tmp_path / "does-not-exist.sqlite3")
    assert data["available"] is False
    assert "error" in data


def test_fetch_dashboard_data_returns_status_counts_and_recent_jobs(tmp_path):
    db_path = tmp_path / "state.sqlite3"
    store = StateStore(db_path)
    try:
        job_a = make_job(tmp_path, "a.mkv", "fp-a")
        job_b = make_job(tmp_path, "b.mkv", "fp-b")
        store.upsert_discovered_job(job_a, model_name="small", device="cpu", compute_type="int8")
        store.upsert_discovered_job(job_b, model_name="small", device="cpu", compute_type="int8")
        store.mark_skipped(job_b, "already present")
    finally:
        store.close()

    data = fetch_dashboard_data(db_path)
    assert data["available"] is True
    assert data["status_counts"]["queued"] == 1
    assert data["status_counts"]["skipped"] == 1
    relative_paths = {job["relative_path"] for job in data["recent_jobs"]}
    assert relative_paths == {"a.mkv", "b.mkv"}


def test_render_dashboard_html_unavailable_shows_error():
    html_text = render_dashboard_html({"available": False, "error": "database is locked"})
    assert "temporarily unavailable" in html_text
    assert "database is locked" in html_text


def test_render_dashboard_html_escapes_untrusted_content():
    data = {
        "available": True,
        "status_counts": {"queued": 1},
        "recent_jobs": [
            {
                "relative_path": "<script>alert(1)</script>.mkv",
                "status": "failed",
                "stage": "extract",
                "last_error": "<img src=x onerror=alert(1)>",
                "updated_at": "2026-01-01",
            }
        ],
    }
    html_text = render_dashboard_html(data)
    assert "<script>alert(1)</script>" not in html_text
    assert "&lt;script&gt;" in html_text
    assert "<img src=x onerror=alert(1)>" not in html_text


def test_render_dashboard_html_shows_status_counts_and_total():
    data = {
        "available": True,
        "status_counts": {"queued": 2, "completed": 3},
        "recent_jobs": [],
    }
    html_text = render_dashboard_html(data)
    assert "queued" in html_text
    assert ">2<" in html_text
    assert "5 total tracked jobs" in html_text


def test_dashboard_server_serves_html_and_json(tmp_path):
    db_path = tmp_path / "state.sqlite3"
    store = StateStore(db_path)
    try:
        job = make_job(tmp_path, "video.mkv", "fp1")
        store.upsert_discovered_job(job, model_name="small", device="cpu", compute_type="int8")
    finally:
        store.close()

    server = run_dashboard_server(db_path, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            assert response.status == 200
            assert "text/html" in response.headers.get("Content-Type", "")
            body = response.read().decode("utf-8")
            assert "video.mkv" in body

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5) as response:
            assert response.status == 200
            assert "application/json" in response.headers.get("Content-Type", "")
            payload = json.loads(response.read().decode("utf-8"))
            assert payload["available"] is True
            assert payload["status_counts"]["queued"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
