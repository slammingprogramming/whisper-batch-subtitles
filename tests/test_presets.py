from __future__ import annotations

import pytest

from whisper_batch_subtitles.presets import PRESETS, preset_names, preset_overrides


def test_preset_names_are_sorted_and_nonempty():
    names = preset_names()
    assert names == sorted(names)
    assert set(names) == set(PRESETS)
    assert len(names) >= 5


def test_preset_overrides_none_returns_empty_dict():
    assert preset_overrides(None) == {}
    assert preset_overrides("") == {}


def test_preset_overrides_returns_a_copy_not_the_original():
    overrides = preset_overrides("fastest")
    overrides["beam_size"] = 999
    assert PRESETS["fastest"]["beam_size"] != 999


def test_preset_overrides_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        preset_overrides("does-not-exist")


def test_low_vram_preset_uses_small_model_and_batch_size_one():
    overrides = preset_overrides("low-vram")
    assert overrides["model"] == "base"
    assert overrides["batch_size"] == 1


def test_cpu_only_preset_forces_cpu_device():
    overrides = preset_overrides("cpu-only")
    assert overrides["device"] == "cpu"


def test_balanced_preset_has_no_overrides():
    assert preset_overrides("balanced") == {}
