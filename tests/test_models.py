from __future__ import annotations

from pathlib import Path

from whisper_batch_subtitles.models import Job


def make_job(tmp_path: Path) -> Job:
    return Job(
        media_path=tmp_path / "video.mkv",
        relative_path="video.mkv",
        fingerprint="abc123",
        file_size=100,
        mtime_ns=1,
        source_srt_path=tmp_path / "video.srt",
        translated_srt_paths={"es": tmp_path / "video.es.srt"},
        audio_cache_path=tmp_path / "cache" / "abc123.wav",
        transcript_cache_path=tmp_path / "cache" / "abc123.json",
        translation_cache_dir=tmp_path / "cache" / "abc123",
    )


def test_translation_cache_path(tmp_path):
    job = make_job(tmp_path)
    assert job.translation_cache_path("es") == job.translation_cache_dir / "es.json"


def test_track_audio_and_transcript_cache_paths(tmp_path):
    job = make_job(tmp_path)
    assert job.track_audio_cache_path("me", 0) == tmp_path / "cache" / "abc123.me.s0.wav"
    assert job.track_transcript_cache_path("others", 1) == tmp_path / "cache" / "abc123.others.s1.json"


def test_role_srt_path(tmp_path):
    job = make_job(tmp_path)
    assert job.role_srt_path("me") == tmp_path / "video.me.srt"


def test_diarization_cache_path_default_variant(tmp_path):
    job = make_job(tmp_path)
    assert job.diarization_cache_path() == tmp_path / "cache" / "abc123.diarization.default.json"


def test_diarization_cache_path_distinguishes_variants(tmp_path):
    job = make_job(tmp_path)
    path_a = job.diarization_cache_path("pyannote-pyannote/speaker-diarization-3.1")
    path_b = job.diarization_cache_path("pyannote-some-other-model")
    external_path = job.diarization_cache_path("external-C:/envs/diarize/python.exe")

    # Different diarization backends/models must not collide on the same cache file,
    # otherwise switching models silently reuses stale speaker turns.
    assert path_a != path_b
    assert path_a != external_path


def test_diarization_cache_path_sanitizes_unsafe_characters(tmp_path):
    job = make_job(tmp_path)
    path = job.diarization_cache_path("external-C:/envs/diarize env/python.exe")
    assert path.parent == job.transcript_cache_path.parent
    assert " " not in path.name
    assert ":" not in path.name


def test_diarization_cache_path_blank_variant_falls_back_to_default(tmp_path):
    job = make_job(tmp_path)
    assert job.diarization_cache_path("") == job.diarization_cache_path("default")
