from __future__ import annotations

from typing import Any

# Named config overrides selectable via `run --preset <name>` or the `wizard` command.
# Applied on top of the hardware-detected recommendation, but anything the user also
# passes explicitly on the command line still wins over the preset's value.
PRESETS: dict[str, dict[str, Any]] = {
    "fastest": {
        "beam_size": 1,
        "diarization_mode": "off",
        "translate": False,
        "suppress_repeated_segments": True,
    },
    "balanced": {},
    "archive-quality": {
        "beam_size": 8,
        "vad_filter": False,
        "skip_existing": True,
        "subtitle_cleanup_text": True,
    },
    "low-vram": {
        "model": "base",
        "batch_size": 1,
        "compute_type": "int8",
    },
    "cpu-only": {
        "device": "cpu",
        "compute_type": "int8",
        "batch_size": 1,
        "diarization_mode": "off",
    },
}


def preset_names() -> list[str]:
    return sorted(PRESETS)


def preset_overrides(name: str | None) -> dict[str, Any]:
    if not name:
        return {}
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name!r} (choose from: {', '.join(preset_names())})")
    return dict(PRESETS[name])
