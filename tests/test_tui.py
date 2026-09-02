from __future__ import annotations

from whisper_batch_subtitles.tui import build_dashboard, is_available, render_dashboard_to_text


def make_snapshot(**overrides):
    snapshot = {
        "discovered": 10,
        "queued": 8,
        "completed": 3,
        "skipped": 2,
        "failed": 1,
        "elapsed_seconds": 3600.0,
        "current_stage_files": {},
    }
    snapshot.update(overrides)
    return snapshot


def test_is_available_reflects_real_rich_installation():
    # rich is a dev-time dependency of this test file itself (via the tui extra),
    # so on a machine where it's installed this must report True.
    assert is_available() is True


def test_build_dashboard_returns_a_renderable_group():
    dashboard = build_dashboard(make_snapshot(), {"extract": 1, "transcribe": 2}, {})
    # Group is duck-typed as a renderable; just confirm it doesn't blow up and has
    # the expected rich renderable protocol (__rich_console__).
    assert hasattr(dashboard, "__rich_console__")


def test_render_dashboard_to_text_includes_counts():
    text = render_dashboard_to_text(make_snapshot(), {"extract": 1, "transcribe": 2}, {})
    assert "Discovered" in text
    assert "10" in text
    assert "Completed" in text
    assert "3" in text
    assert "Failed" in text
    assert "1" in text


def test_render_dashboard_to_text_includes_queue_depths():
    text = render_dashboard_to_text(make_snapshot(), {"extract": 5, "transcribe": 0, "write": 2}, {})
    assert "extract" in text
    assert "transcribe" in text
    assert "write" in text


def test_render_dashboard_to_text_shows_current_files():
    text = render_dashboard_to_text(
        make_snapshot(),
        {"extract": 0},
        {"transcribe": "episode-01.mkv", "write": "episode-02.mkv"},
    )
    assert "episode-01.mkv" in text
    assert "episode-02.mkv" in text


def test_render_dashboard_to_text_idle_when_no_current_files():
    text = render_dashboard_to_text(make_snapshot(), {"extract": 0}, {})
    assert "idle" in text


def test_render_dashboard_to_text_shows_throughput_and_eta():
    snapshot = make_snapshot(completed=36, elapsed_seconds=3600.0)  # 36 files/hour
    text = render_dashboard_to_text(snapshot, {}, {})
    assert "36.00 files/hour" in text
    assert "ETA" in text
