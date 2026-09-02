from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from whisper_batch_subtitles.config import (
    AppConfig,
    _normalize_int_list,
    _normalize_language_list,
    _parse_bool,
    build_config,
    config_to_yaml_text,
    default_config_path,
    profile_path,
    sample_config_text,
)
from whisper_batch_subtitles.hardware import GPUInfo, HardwareProfile, recommend_runtime


def test_precedence_cli_overrides_env_overrides_file_overrides_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("model: medium\ntarget_languages: [fr]\n", encoding="utf-8")
    monkeypatch.setenv("WBS_TARGET_LANGUAGES", "de,it")

    config = build_config({"config": str(config_file), "target_languages": ["ja"]})

    assert config.model == "medium"  # untouched by env/cli -> file value survives
    assert config.target_languages == ["ja"]  # cli beats env and file


def test_env_overrides_file_when_cli_is_silent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("target_languages: [fr]\n", encoding="utf-8")
    monkeypatch.setenv("WBS_TARGET_LANGUAGES", "de,it")

    config = build_config({"config": str(config_file)})

    assert config.target_languages == ["de", "it"]


def test_env_overrides_parse_bool_int_and_float(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WBS_TRANSLATE", "false")
    monkeypatch.setenv("WBS_FFMPEG_WORKERS", "7")
    monkeypatch.setenv("WBS_PROGRESS_INTERVAL_SECONDS", "2.5")

    config = build_config({})

    assert config.translate is False
    assert config.ffmpeg_workers == 7
    assert config.progress_interval_seconds == 2.5


def test_hardware_recommendation_only_fills_unset_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = HardwareProfile(cpu_cores=8, ram_gb=32.0, gpus=[], cuda_available=False)
    recommendation = recommend_runtime(profile)

    config = build_config({"model": "large-v3"}, recommendation=recommendation)

    assert config.model == "large-v3"  # explicit value beats hardware recommendation
    assert config.device == recommendation.device  # unset field filled by recommendation


def test_overwrite_forces_skip_existing_false(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = build_config({"overwrite": True})
    assert config.skip_existing is False


def test_translate_false_clears_target_languages(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = build_config({"translate": False, "target_languages": ["es"]})
    assert config.target_languages == []
    assert config.effective_target_languages == []


def test_translate_true_with_empty_targets_defaults_to_en(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = build_config({"translate": True, "target_languages": []})
    assert config.target_languages == ["en"]


def test_finalize_config_derives_state_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = build_config({"state_dir": str(tmp_path / "state")})
    assert config.state_db == tmp_path / "state" / "state.sqlite3"
    assert config.text_log_path == tmp_path / "state" / "logs" / "runtime.log"
    assert config.json_log_path == tmp_path / "state" / "logs" / "events.jsonl"


def test_effective_target_languages_dedupes_case_insensitively(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = build_config({"target_languages": ["EN", "en", "Es"]})
    assert config.effective_target_languages == ["en", "es"]


def test_default_config_path():
    assert default_config_path() == Path("whisper-batch-subtitles.yaml")


def test_normalize_language_list_variants():
    assert _normalize_language_list("en, es , fr") == ["en", "es", "fr"]
    assert _normalize_language_list(["EN", "Es"]) == ["en", "es"]
    assert _normalize_language_list(None) == ["en"]


def test_parse_bool_variants():
    assert _parse_bool("true") is True
    assert _parse_bool("Yes") is True
    assert _parse_bool("1") is True
    assert _parse_bool("off") is False
    assert _parse_bool("") is False


def test_sample_config_text_is_valid_yaml_with_expected_keys():
    text = sample_config_text()
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    loaded = yaml.safe_load(body)
    assert loaded["root_dir"] == "process"
    assert loaded["target_languages"] == ["en"]
    assert "model" in loaded


def test_normalize_int_list_variants():
    assert _normalize_int_list("0, 1, 2") == [0, 1, 2]
    assert _normalize_int_list([0, "1"]) == [0, 1]
    assert _normalize_int_list(None) == []
    assert _normalize_int_list("") == []


def test_profile_path_uses_profiles_directory():
    assert profile_path("podcast") == Path("profiles") / "podcast.yaml"


def test_load_profile_values_merges_into_build_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "podcast.yaml").write_text("model: medium\nbeam_size: 8\n", encoding="utf-8")

    config = build_config({"profile": "podcast"})
    assert config.model == "medium"
    assert config.beam_size == 8


def test_load_profile_values_missing_profile_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Profile not found"):
        build_config({"profile": "does-not-exist"})


def test_profile_layer_is_overridden_by_cli_and_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "podcast.yaml").write_text("model: medium\n", encoding="utf-8")
    monkeypatch.setenv("WBS_MODEL", "large-v3")

    # env beats the profile layer
    config = build_config({"profile": "podcast"})
    assert config.model == "large-v3"

    # cli beats both
    config = build_config({"profile": "podcast", "model": "small"})
    assert config.model == "small"


def test_config_to_yaml_text_round_trips_through_build_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = build_config({"root_dir": str(tmp_path / "process"), "beam_size": 7})
    text = config_to_yaml_text(config)
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "saved.yaml").write_text(text, encoding="utf-8")

    reloaded = build_config({"profile": "saved"})
    assert reloaded.beam_size == 7


def test_config_to_yaml_text_excludes_secrets():
    config = AppConfig(pyannote_auth_token="super-secret-token", deepl_api_key="super-secret-key")
    text = config_to_yaml_text(config)
    assert "super-secret-token" not in text
    assert "super-secret-key" not in text
    assert "pyannote_auth_token" not in text
    assert "deepl_api_key" not in text


def test_config_to_yaml_text_excludes_installation_specific_paths():
    config = AppConfig(state_dir=Path("/some/machine/specific/path"))
    text = config_to_yaml_text(config)
    assert "state_dir" not in text
    assert "machine/specific" not in text


def test_sample_config_text_includes_hardware_comments_when_profile_given():
    profile = HardwareProfile(
        cpu_cores=8,
        ram_gb=32.0,
        gpus=[GPUInfo(index=0, name="Test GPU", memory_mb=8192, backend="cuda")],
        cuda_available=True,
    )
    recommendation = recommend_runtime(profile)
    text = sample_config_text(recommendation=recommendation, profile=profile)
    assert "Test GPU" in text
    assert f"model: {recommendation.model}" in text
