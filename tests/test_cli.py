from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

import whisper_batch_subtitles.cli as cli_module
from whisper_batch_subtitles.cli import (
    _COMMANDS,
    _namespace_to_config_values,
    _normalize_argv,
    _resolve_wizard_preset_choice,
    _run_init_config,
    _run_watch_loop,
    _run_wizard,
    build_parser,
    main,
    wizard_overrides,
)
from whisper_batch_subtitles.presets import preset_names

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not available on PATH")


def test_normalize_argv_defaults_to_run_when_empty():
    assert _normalize_argv([]) == ["run"]


def test_normalize_argv_prepends_run_for_bare_flags():
    assert _normalize_argv(["--root-dir", "process"]) == ["run", "--root-dir", "process"]


def test_normalize_argv_leaves_known_commands_alone():
    for command in _COMMANDS:
        assert _normalize_argv([command]) == [command]


def test_normalize_argv_leaves_help_alone():
    assert _normalize_argv(["-h"]) == ["-h"]
    assert _normalize_argv(["--help"]) == ["--help"]


def test_namespace_to_config_values_drops_none_and_command():
    import argparse

    namespace = argparse.Namespace(command="run", model="small", device=None, translate=None)
    values = _namespace_to_config_values(namespace)
    assert values == {"model": "small"}
    assert "command" not in values
    assert "device" not in values


def test_build_parser_exposes_all_subcommands():
    parser = build_parser()
    args = parser.parse_args(["wizard"])
    assert args.command == "wizard"
    args = parser.parse_args(["hardware"])
    assert args.command == "hardware"
    args = parser.parse_args(["serve"])
    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 8765


def test_build_parser_hardware_accepts_benchmark_flags():
    parser = build_parser()
    args = parser.parse_args(
        ["hardware", "--benchmark", "--benchmark-model", "tiny", "--benchmark-duration-seconds", "5"]
    )
    assert args.benchmark is True
    assert args.benchmark_model == "tiny"
    assert args.benchmark_duration_seconds == 5.0


def test_build_parser_serve_accepts_host_and_port():
    parser = build_parser()
    args = parser.parse_args(["serve", "--host", "0.0.0.0", "--port", "9000"])
    assert args.host == "0.0.0.0"
    assert args.port == 9000


def test_build_parser_run_accepts_preset_and_gpu_devices():
    parser = build_parser()
    args = parser.parse_args(["run", "--preset", "low-vram", "--gpu-device", "0", "--gpu-device", "1"])
    assert args.preset == "low-vram"
    assert args.gpu_device_indices == [0, 1]


def test_build_parser_run_rejects_unknown_preset():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--preset", "not-a-real-preset"])


def test_build_parser_run_accepts_watch_flags():
    parser = build_parser()
    args = parser.parse_args(["run", "--watch", "--watch-interval-seconds", "45"])
    assert args.watch is True
    assert args.watch_interval_seconds == 45.0


def test_build_parser_run_accepts_subtitle_and_duplicate_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--subtitle-max-line-chars",
            "30",
            "--no-subtitle-cleanup-text",
            "--duplicate-detection",
            "skip",
            "--translation-backend",
            "deepl",
        ]
    )
    assert args.subtitle_max_line_chars == 30
    assert args.subtitle_cleanup_text is False
    assert args.duplicate_detection == "skip"
    assert args.translation_backend == "deepl"


def test_resolve_wizard_preset_choice_empty_uses_default():
    names = preset_names()
    assert _resolve_wizard_preset_choice("", names, 2) == names[1]


def test_resolve_wizard_preset_choice_numeric():
    names = preset_names()
    assert _resolve_wizard_preset_choice("1", names, 2) == names[0]


def test_resolve_wizard_preset_choice_by_name():
    names = preset_names()
    assert _resolve_wizard_preset_choice("fastest", names, 2) == "fastest"


def test_resolve_wizard_preset_choice_invalid_falls_back_to_default():
    names = preset_names()
    assert _resolve_wizard_preset_choice("nonsense", names, 2) == names[1]
    assert _resolve_wizard_preset_choice("999", names, 2) == names[1]


def test_wizard_overrides_includes_preset_and_translation_settings():
    overrides = wizard_overrides("low-vram", translate=True, target_languages=["es", "fr"])
    assert overrides["model"] == "base"
    assert overrides["translate"] is True
    assert overrides["target_languages"] == ["es", "fr"]


def test_wizard_overrides_no_target_languages_when_translate_false():
    overrides = wizard_overrides("balanced", translate=False, target_languages=[])
    assert overrides["translate"] is False
    assert "target_languages" not in overrides


def _make_input_fn(answers: list[str]):
    iterator = iter(answers)

    def input_fn(_prompt: str) -> str:
        return next(iterator)

    return input_fn


def test_run_wizard_writes_config_from_scripted_answers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_fn = _make_input_fn(["low-vram", "y", "es,fr", "my-config.yaml"])
    result = _run_wizard(input_fn=input_fn)
    assert result == 0
    output = (tmp_path / "my-config.yaml").read_text(encoding="utf-8")
    assert "model: base" in output
    assert "translate: true" in output
    assert "- es" in output
    assert "- fr" in output


def test_run_wizard_default_output_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_fn = _make_input_fn(["", "n", "", ""])
    result = _run_wizard(input_fn=input_fn)
    assert result == 0
    assert (tmp_path / "whisper-batch-subtitles.yaml").exists()
    output = (tmp_path / "whisper-batch-subtitles.yaml").read_text(encoding="utf-8")
    assert "translate: false" in output


def test_run_init_config_with_preset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _run_init_config("cpu-only.yaml", preset_name="cpu-only")
    assert result == 0
    output = (tmp_path / "cpu-only.yaml").read_text(encoding="utf-8")
    assert "device: cpu" in output
    assert "batch_size: 1" in output


def test_run_watch_loop_calls_run_once_until_interrupted():
    calls = {"count": 0}

    def run_once() -> int:
        calls["count"] += 1
        if calls["count"] >= 3:
            raise KeyboardInterrupt
        return 0

    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    result = _run_watch_loop(
        run_once=run_once,
        interval_seconds=5.0,
        logger=logging.getLogger("test-watch"),
        sleep_fn=fake_sleep,
    )
    assert result == 0
    assert calls["count"] == 3
    # the 3rd run_once() raises before reaching its sleep_fn call, so only 2 sleeps happen
    assert sleeps == [5.0, 5.0]


def test_run_watch_loop_interrupt_during_sleep_also_stops_cleanly():
    def run_once() -> int:
        return 0

    def fake_sleep(seconds: float) -> None:
        raise KeyboardInterrupt

    result = _run_watch_loop(
        run_once=run_once,
        interval_seconds=1.0,
        logger=logging.getLogger("test-watch"),
        sleep_fn=fake_sleep,
    )
    assert result == 0


@requires_ffmpeg
def test_main_save_profile_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "process").mkdir()
    result = main(["run", "--root-dir", "process", "--preset", "fastest", "--save-profile", "podcast"])
    assert result == 0
    saved = (tmp_path / "profiles" / "podcast.yaml").read_text(encoding="utf-8")
    assert "beam_size: 1" in saved
    assert "translate: false" in saved


@requires_ffmpeg
def test_main_run_with_profile_round_trips(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "process").mkdir()
    main(["run", "--root-dir", "process", "--preset", "fastest", "--save-profile", "podcast"])
    result = main(["run", "--root-dir", "process", "--profile", "podcast", "--save-profile", "podcast-roundtrip"])
    assert result == 0
    original = (tmp_path / "profiles" / "podcast.yaml").read_text(encoding="utf-8")
    roundtrip = (tmp_path / "profiles" / "podcast-roundtrip.yaml").read_text(encoding="utf-8")
    assert original == roundtrip


@requires_ffmpeg
def test_main_run_with_missing_profile_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "process").mkdir()
    with pytest.raises(ValueError, match="Profile not found"):
        main(["run", "--root-dir", "process", "--profile", "does-not-exist"])
