from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any

_MULTISPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,!?;:])")
_REPEATED_PUNCT_RE = re.compile(r"([!?])\1{2,}")


def format_timestamp(seconds: float) -> str:
    milliseconds = int((seconds % 1) * 1000)
    whole_seconds = int(seconds)
    hours, whole_seconds = divmod(whole_seconds, 3600)
    minutes, whole_seconds = divmod(whole_seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def normalize_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for segment in segments:
        normalized_segment = {
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": str(segment["text"]).strip(),
        }
        if "speaker" in segment:
            normalized_segment["speaker"] = str(segment["speaker"])
        if "speaker_label" in segment:
            normalized_segment["speaker_label"] = str(segment["speaker_label"])
        normalized.append(normalized_segment)
    return normalized


def clean_subtitle_text(text: str) -> str:
    """Mechanical cleanup: collapse whitespace, fix spacing before punctuation, cap
    repeated `!!!`/`???` runs at two characters, and capitalize the first letter.
    Deliberately conservative -- no NLP, nothing that could plausibly change meaning."""
    text = _MULTISPACE_RE.sub(" ", text).strip()
    if not text:
        return text
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1\1", text)
    if text[0].isalpha():
        text = text[0].upper() + text[1:]
    return text


def wrap_subtitle_text(text: str, *, max_chars_per_line: int, max_lines: int) -> str:
    """Balanced line wrapping for subtitle display. Never drops text: if wrapping would
    need more than `max_lines` lines, the overflow is joined onto the last line instead
    of being truncated."""
    if max_chars_per_line <= 0 or not text:
        return text
    lines = textwrap.wrap(text, width=max_chars_per_line, break_long_words=False, break_on_hyphens=False)
    if not lines:
        return text
    if max_lines > 0 and len(lines) > max_lines:
        head = lines[: max_lines - 1]
        tail = " ".join(lines[max_lines - 1 :])
        lines = [*head, tail]
    return "\n".join(lines)


def suppress_repeated_segments(segments: list[dict[str, Any]], *, min_repeats: int = 3) -> list[dict[str, Any]]:
    """Collapse runs of `min_repeats`+ consecutive segments with identical text into a
    single segment spanning the run. Targets a known Whisper failure mode (looping on
    the same phrase during silence/noise) rather than doing general hallucination
    detection."""
    if not segments:
        return segments

    collapsed: list[dict[str, Any]] = []
    run: list[dict[str, Any]] = []

    def flush_run() -> None:
        if not run:
            return
        if len(run) >= min_repeats:
            merged = dict(run[0])
            merged["end"] = run[-1]["end"]
            collapsed.append(merged)
        else:
            collapsed.extend(run)

    for segment in segments:
        if run and _normalized_text(segment) == _normalized_text(run[-1]) and _normalized_text(segment):
            run.append(segment)
        else:
            flush_run()
            run = [segment]
    flush_run()
    return collapsed


def _normalized_text(segment: dict[str, Any]) -> str:
    return " ".join(str(segment.get("text", "")).strip().lower().split())


def write_srt(
    segments: list[dict[str, Any]],
    output_path: Path,
    *,
    cleanup_text: bool = True,
    max_line_chars: int = 42,
    max_lines: int = 2,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, segment in enumerate(normalize_segments(segments), start=1):
            handle.write(f"{index}\n")
            handle.write(
                f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            )
            handle.write(f"{_format_segment_text(segment, cleanup_text=cleanup_text, max_line_chars=max_line_chars, max_lines=max_lines)}\n\n")


def _format_segment_text(segment: dict[str, Any], *, cleanup_text: bool, max_line_chars: int, max_lines: int) -> str:
    text = str(segment["text"]).strip()
    if cleanup_text:
        text = clean_subtitle_text(text)
    if max_line_chars > 0:
        text = wrap_subtitle_text(text, max_chars_per_line=max_line_chars, max_lines=max_lines)
    speaker_label = segment.get("speaker_label")
    if speaker_label:
        return f"[{speaker_label}] {text}"
    return text
