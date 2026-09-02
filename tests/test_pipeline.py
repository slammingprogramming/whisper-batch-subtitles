from __future__ import annotations

from pathlib import Path

import pytest

import whisper_batch_subtitles.pipeline as pipeline_module
from whisper_batch_subtitles.config import AppConfig
from whisper_batch_subtitles.models import AudioStreamInfo, Job, TrackAssignment
from whisper_batch_subtitles.pipeline import STOP, PipelineRunner, Transcriber, classify_stage_error, format_progress_line
from whisper_batch_subtitles.state import StateStore


def make_config(tmp_path: Path) -> AppConfig:
    config = AppConfig(
        root_dir=tmp_path / "process",
        state_dir=tmp_path / "state",
        diarization_mode="off",
        translate=False,
        max_retries=0,
        ffmpeg_workers=1,
        transcription_workers=1,
        translation_workers=1,
    )
    config.state_db = config.state_dir / "state.sqlite3"
    config.text_log_path = config.state_dir / "logs" / "runtime.log"
    config.json_log_path = config.state_dir / "logs" / "events.jsonl"
    config.root_dir.mkdir(parents=True, exist_ok=True)
    return config


@pytest.fixture
def runner(tmp_path):
    config = make_config(tmp_path)
    config.ensure_directories()
    state = StateStore(config.state_db)
    pipeline_runner = PipelineRunner(config, state)
    yield pipeline_runner
    state.close()


def test_discover_jobs_skips_file_that_fails_to_inspect_instead_of_crashing(runner, monkeypatch):
    # Regression test: previously, one file that ffprobe couldn't read would
    # raise out of _discover_jobs and abort the entire batch run, losing every
    # other already-discovered job in the library.
    good_path = runner.config.root_dir / "good.mp4"
    bad_path = runner.config.root_dir / "bad.mp4"
    good_path.write_bytes(b"placeholder")
    bad_path.write_bytes(b"placeholder")

    def fake_probe(media_path):
        if media_path.name == "bad.mp4":
            raise RuntimeError("ffprobe failed for corrupt file")
        return []

    monkeypatch.setattr(pipeline_module, "probe_audio_streams", fake_probe)

    pending_jobs = runner._discover_jobs()

    relative_paths = {job.relative_path for job, _stage in pending_jobs}
    assert relative_paths == {"good.mp4"}
    snapshot = runner.metrics.snapshot()
    assert snapshot["failed"] == 1
    assert snapshot["discovered"] == 2


def make_multi_track_job(tmp_path: Path) -> Job:
    streams = [AudioStreamInfo(index=0), AudioStreamInfo(index=1)]
    assignments = [
        TrackAssignment(stream_index=0, role="ignore", label="Ignore"),
        TrackAssignment(stream_index=1, role="ignore", label="Ignore"),
    ]
    return Job(
        media_path=tmp_path / "video.mkv",
        relative_path="video.mkv",
        fingerprint="fp",
        file_size=1,
        mtime_ns=1,
        source_srt_path=tmp_path / "video.srt",
        translated_srt_paths={},
        audio_cache_path=tmp_path / "cache" / "fp.wav",
        transcript_cache_path=tmp_path / "cache" / "fp.json",
        translation_cache_dir=tmp_path / "cache" / "fp",
        audio_streams=streams,
        track_assignments=assignments,
    )


def test_extraction_worker_fails_loudly_when_all_tracks_ignored(runner, tmp_path):
    # Regression test: a saved/manual track profile that marks every stream
    # "ignore" used to silently fall through both extraction and transcription
    # loops (zero iterations) and complete with an empty subtitle file.
    job = make_multi_track_job(tmp_path)
    runner.extract_queue.put(job)
    runner.extract_queue.put(STOP)

    runner._extraction_worker()

    assert runner.metrics.snapshot()["failed"] == 1
    assert runner.transcribe_queue.empty()


def test_format_progress_line_includes_counts_and_queues():
    snapshot = {
        "discovered": 10,
        "queued": 8,
        "completed": 3,
        "skipped": 2,
        "failed": 1,
        "elapsed_seconds": 3600.0,
    }
    line = format_progress_line(snapshot, ["extract=1", "transcribe=2"])
    assert "discovered=10" in line
    assert "completed=3" in line
    assert "failed=1" in line
    assert "throughput=3.00 files/hour" in line
    assert "queues[extract=1, transcribe=2]" in line


def test_format_progress_line_unknown_eta_before_any_completion():
    snapshot = {
        "discovered": 5,
        "queued": 5,
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "elapsed_seconds": 10.0,
    }
    line = format_progress_line(snapshot, [])
    assert "eta=unknown" in line


def test_classify_stage_error_recognizes_permanent_patterns():
    assert classify_stage_error(RuntimeError("Invalid data found when processing input")) == "permanent"
    assert classify_stage_error(RuntimeError("moov atom not found")) == "permanent"
    assert classify_stage_error(FileNotFoundError("No such file or directory")) == "permanent"


def test_classify_stage_error_defaults_to_transient():
    assert classify_stage_error(RuntimeError("temporary network hiccup")) == "transient"
    assert classify_stage_error(TimeoutError("timed out")) == "transient"


def test_handle_stage_error_does_not_retry_permanent_errors(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config.max_retries = 5  # plenty of retry budget available -- must not be used
    config.ensure_directories()
    state = StateStore(config.state_db)
    try:
        runner = PipelineRunner(config, state)
        job = make_multi_track_job(tmp_path)
        error = RuntimeError("moov atom not found")
        runner._handle_stage_error(job, "extract", error, runner.extract_queue)
        assert runner.metrics.snapshot()["failed"] == 1
        assert runner.extract_queue.empty()
    finally:
        state.close()


def test_handle_stage_error_retries_transient_errors(tmp_path):
    config = make_config(tmp_path)
    config.max_retries = 2
    config.ensure_directories()
    state = StateStore(config.state_db)
    try:
        runner = PipelineRunner(config, state)
        job = make_multi_track_job(tmp_path)
        error = RuntimeError("connection reset")
        runner._handle_stage_error(job, "extract", error, runner.extract_queue)
        assert runner.metrics.snapshot()["failed"] == 0
        assert runner.extract_queue.qsize() == 1
    finally:
        state.close()


def test_device_index_for_worker_round_robins_across_configured_gpus(tmp_path):
    config = make_config(tmp_path)
    config.device = "cuda"
    config.gpu_device_indices = [2, 5]
    config.ensure_directories()
    state = StateStore(config.state_db)
    try:
        runner = PipelineRunner(config, state)
        assert runner._device_index_for_worker(0) == 2
        assert runner._device_index_for_worker(1) == 5
        assert runner._device_index_for_worker(2) == 2
        assert runner._device_index_for_worker(3) == 5
    finally:
        state.close()


def test_device_index_for_worker_none_when_not_cuda(tmp_path):
    config = make_config(tmp_path)
    config.device = "cpu"
    config.gpu_device_indices = [0, 1]
    config.ensure_directories()
    state = StateStore(config.state_db)
    try:
        runner = PipelineRunner(config, state)
        assert runner._device_index_for_worker(0) is None
    finally:
        state.close()


def test_device_index_for_worker_none_when_no_indices_configured(tmp_path):
    config = make_config(tmp_path)
    config.device = "cuda"
    config.gpu_device_indices = []
    config.ensure_directories()
    state = StateStore(config.state_db)
    try:
        runner = PipelineRunner(config, state)
        assert runner._device_index_for_worker(0) is None
    finally:
        state.close()


class FakeWhisperModel:
    last_kwargs: dict | None = None

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        FakeWhisperModel.last_kwargs = kwargs


def test_transcriber_passes_device_index_to_whisper_model(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_module, "WhisperModel", FakeWhisperModel)
    config = make_config(tmp_path)
    config.batch_size = 1
    Transcriber(config, device_index=3)
    assert FakeWhisperModel.last_kwargs["device_index"] == 3


def test_transcriber_omits_device_index_when_not_given(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_module, "WhisperModel", FakeWhisperModel)
    config = make_config(tmp_path)
    config.batch_size = 1
    Transcriber(config)
    assert "device_index" not in FakeWhisperModel.last_kwargs


def test_content_signature_not_computed_when_duplicate_detection_off(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "probe_audio_streams", lambda path: [])
    config = make_config(tmp_path)
    config.duplicate_detection = "off"
    config.ensure_directories()
    state = StateStore(config.state_db)
    try:
        runner = PipelineRunner(config, state)
        media_path = config.root_dir / "video.mp4"
        media_path.write_bytes(b"content")
        job = runner._build_job(media_path)
        assert job.content_signature is None
    finally:
        state.close()


def test_discover_jobs_duplicate_skip_mode_skips_matching_content(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "probe_audio_streams", lambda path: [])
    config = make_config(tmp_path)
    config.duplicate_detection = "skip"
    config.ensure_directories()
    state = StateStore(config.state_db)
    try:
        runner = PipelineRunner(config, state)
        original = config.root_dir / "original.mp4"
        original.write_bytes(b"identical content payload")
        pending = runner._discover_jobs()
        job, _stage = pending[0]
        state.mark_completed(job, runtime_seconds=1.0)

        duplicate_path = config.root_dir / "copy.mp4"
        duplicate_path.write_bytes(b"identical content payload")
        pending_again = runner._discover_jobs()

        relative_paths = {j.relative_path for j, _stage in pending_again}
        assert "copy.mp4" not in relative_paths
    finally:
        state.close()


def test_discover_jobs_duplicate_warn_mode_still_queues_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "probe_audio_streams", lambda path: [])
    config = make_config(tmp_path)
    config.duplicate_detection = "warn"
    config.ensure_directories()
    state = StateStore(config.state_db)
    try:
        runner = PipelineRunner(config, state)
        original = config.root_dir / "original.mp4"
        original.write_bytes(b"identical content payload")
        pending = runner._discover_jobs()
        job, _stage = pending[0]
        state.mark_completed(job, runtime_seconds=1.0)

        duplicate_path = config.root_dir / "copy.mp4"
        duplicate_path.write_bytes(b"identical content payload")
        pending_again = runner._discover_jobs()

        relative_paths = {j.relative_path for j, _stage in pending_again}
        assert "copy.mp4" in relative_paths
    finally:
        state.close()


def test_write_srt_choke_point_applies_suppression_and_formatting(tmp_path):
    config = make_config(tmp_path)
    config.suppress_repeated_segments = True
    config.subtitle_cleanup_text = True
    config.ensure_directories()
    state = StateStore(config.state_db)
    try:
        runner = PipelineRunner(config, state)
        segments = [
            {"start": 0.0, "end": 1.0, "text": "hi"},
            {"start": 1.0, "end": 2.0, "text": "hi"},
            {"start": 2.0, "end": 3.0, "text": "hi"},
            {"start": 3.0, "end": 4.0, "text": "actual content"},
        ]
        output_path = tmp_path / "out.srt"
        runner._write_srt(segments, output_path)
        content = output_path.read_text(encoding="utf-8")
        # the 3 repeated "hi" segments collapse into one cue, and cleanup capitalizes it
        assert content.count("Hi") == 1
        assert "Actual content" in content
    finally:
        state.close()


def test_dynamic_ffmpeg_workers_end_to_end_completes_without_hanging(tmp_path, monkeypatch):
    # Integration test for the dynamic-scaling wiring inside the real run() orchestration
    # -- not just the isolated DynamicWorkerPool/decide_scale_action mechanics, which
    # already have thorough unit coverage in tests/test_scaling.py. This is checking that
    # start_initial/shutdown/join_all interoperate correctly with the rest of the
    # pipeline (extract -> transcribe -> write) without hanging or losing jobs.
    class FakeSegment:
        def __init__(self, start: float, end: float, text: str) -> None:
            self.start = start
            self.end = end
            self.text = text

    class FakeInfo:
        language = "en"

    class FakeWhisperModelForRun:
        def __init__(self, model_name: str, **kwargs) -> None:
            pass

        def transcribe(self, audio_path: str, **kwargs):
            return [FakeSegment(0.0, 1.0, "hello world")], FakeInfo()

    def fake_extract_audio(media_path: Path, audio_path: Path, **kwargs) -> None:
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"fake-audio")

    monkeypatch.setattr(pipeline_module, "probe_audio_streams", lambda path: [])
    monkeypatch.setattr(pipeline_module, "probe_duration", lambda path: 1.0)
    monkeypatch.setattr(pipeline_module, "WhisperModel", FakeWhisperModelForRun)
    monkeypatch.setattr(pipeline_module, "extract_audio", fake_extract_audio)

    config = make_config(tmp_path)
    config.dynamic_ffmpeg_workers = True
    config.ffmpeg_workers = 1
    config.ffmpeg_workers_min = 1
    config.ffmpeg_workers_max = 3
    config.ensure_directories()

    for index in range(6):
        (config.root_dir / f"video{index}.mp4").write_bytes(b"placeholder")

    state = StateStore(config.state_db)
    try:
        runner = PipelineRunner(config, state)
        failures = runner.run()
        assert failures == 0
        counts = state.status_counts()
        assert counts.get("completed") == 6
        for index in range(6):
            assert (config.root_dir / f"video{index}.srt").exists()
    finally:
        state.close()


def test_diarization_variant_differs_by_model(tmp_path, monkeypatch):
    # No auth token -> PyannoteDiarizer construction fails fast and falls back
    # to a no-op Diarizer without ever hitting the network. Guard against a
    # stray PYANNOTE_AUTH_TOKEN in the environment triggering a real download.
    monkeypatch.delenv("PYANNOTE_AUTH_TOKEN", raising=False)
    config = make_config(tmp_path)
    config.diarization_mode = "pyannote"
    config.pyannote_model = "pyannote/speaker-diarization-3.1"
    config.pyannote_auth_token = None
    config.ensure_directories()
    state = StateStore(config.state_db)
    try:
        runner = PipelineRunner(config, state)
        assert "pyannote/speaker-diarization-3.1" in runner.diarization_variant
        assert runner.diarization_variant.startswith("pyannote-")
    finally:
        state.close()
