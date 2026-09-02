from __future__ import annotations

from pathlib import Path

from whisper_batch_subtitles.subtitles import (
    clean_subtitle_text,
    format_timestamp,
    normalize_segments,
    suppress_repeated_segments,
    wrap_subtitle_text,
    write_srt,
)


def test_format_timestamp_basic():
    assert format_timestamp(0.0) == "00:00:00,000"
    assert format_timestamp(1.5) == "00:00:01,500"
    assert format_timestamp(61.25) == "00:01:01,250"
    assert format_timestamp(3661.75) == "01:01:01,750"


def test_format_timestamp_milliseconds_never_overflow_to_1000():
    # Floating point artifacts (e.g. 1.9999999999999998) must not produce
    # a millisecond component of 1000, which would render an invalid SRT cue.
    value = format_timestamp(1.9999999999999998)
    hours, minutes, rest = value.split(":")
    seconds, milliseconds = rest.split(",")
    assert int(milliseconds) < 1000


def test_normalize_segments_strips_text_and_keeps_speaker_fields():
    raw = [
        {"start": 1, "end": 2, "text": "  hello  ", "speaker": "me", "speaker_label": "Me"},
        {"start": "3.5", "end": "4.0", "text": "world"},
    ]
    normalized = normalize_segments(raw)
    assert normalized[0] == {"start": 1.0, "end": 2.0, "text": "hello", "speaker": "me", "speaker_label": "Me"}
    assert normalized[1] == {"start": 3.5, "end": 4.0, "text": "world"}
    assert "speaker" not in normalized[1]


def test_clean_subtitle_text_collapses_whitespace_and_capitalizes():
    assert clean_subtitle_text("  hello   world  ") == "Hello world"


def test_clean_subtitle_text_fixes_spacing_before_punctuation():
    assert clean_subtitle_text("hello , world !") == "Hello, world!"


def test_clean_subtitle_text_caps_repeated_punctuation():
    assert clean_subtitle_text("wait!!!!") == "Wait!!"
    assert clean_subtitle_text("really????") == "Really??"


def test_clean_subtitle_text_empty_string_stays_empty():
    assert clean_subtitle_text("   ") == ""


def test_clean_subtitle_text_does_not_uppercase_non_alpha_start():
    assert clean_subtitle_text("[music] okay") == "[music] okay"


def test_wrap_subtitle_text_short_text_unchanged():
    assert wrap_subtitle_text("hi there", max_chars_per_line=42, max_lines=2) == "hi there"


def test_wrap_subtitle_text_wraps_long_line():
    text = "this is a long sentence that should wrap onto a second line for readability"
    wrapped = wrap_subtitle_text(text, max_chars_per_line=20, max_lines=2)
    lines = wrapped.split("\n")
    assert len(lines) == 2
    assert len(lines[0]) <= 20
    assert sorted(wrapped.split()) == sorted(text.split())


def test_wrap_subtitle_text_never_drops_content_beyond_max_lines():
    text = " ".join(["word"] * 30)
    wrapped = wrap_subtitle_text(text, max_chars_per_line=10, max_lines=2)
    lines = wrapped.split("\n")
    assert len(lines) == 2
    # every original word must survive somewhere in the wrapped output
    assert sorted(wrapped.split()) == sorted(text.split())


def test_wrap_subtitle_text_disabled_with_zero_width():
    text = "x" * 100
    assert wrap_subtitle_text(text, max_chars_per_line=0, max_lines=2) == text


def test_suppress_repeated_segments_collapses_long_run():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "the the the"},
        {"start": 1.0, "end": 2.0, "text": "the the the"},
        {"start": 2.0, "end": 3.0, "text": "the the the"},
        {"start": 3.0, "end": 4.0, "text": "the the the"},
        {"start": 4.0, "end": 5.0, "text": "actual content"},
    ]
    result = suppress_repeated_segments(segments, min_repeats=3)
    assert len(result) == 2
    assert result[0]["text"] == "the the the"
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 4.0
    assert result[1]["text"] == "actual content"


def test_suppress_repeated_segments_leaves_short_runs_alone():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 1.0, "end": 2.0, "text": "hello"},
        {"start": 2.0, "end": 3.0, "text": "world"},
    ]
    result = suppress_repeated_segments(segments, min_repeats=3)
    assert len(result) == 3


def test_suppress_repeated_segments_is_case_and_whitespace_insensitive():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello."},
        {"start": 1.0, "end": 2.0, "text": "  hello.  "},
        {"start": 2.0, "end": 3.0, "text": "HELLO."},
    ]
    result = suppress_repeated_segments(segments, min_repeats=3)
    assert len(result) == 1


def test_suppress_repeated_segments_empty_input():
    assert suppress_repeated_segments([]) == []


def test_write_srt_produces_valid_sequential_cues(tmp_path: Path):
    output_path = tmp_path / "out.srt"
    segments = [
        {"start": 0.0, "end": 1.0, "text": "first"},
        {"start": 1.0, "end": 2.5, "text": "second", "speaker_label": "Alice"},
    ]
    write_srt(segments, output_path)
    content = output_path.read_text(encoding="utf-8")

    # default cleanup capitalizes the first letter of each cue
    assert content.startswith("1\n00:00:00,000 --> 00:00:01,000\nFirst\n\n")
    assert "2\n00:00:01,000 --> 00:00:02,500\n[Alice] Second\n\n" in content


def test_write_srt_can_disable_cleanup_and_wrapping(tmp_path: Path):
    output_path = tmp_path / "out.srt"
    segments = [{"start": 0.0, "end": 1.0, "text": "  raw   text  "}]
    write_srt(segments, output_path, cleanup_text=False, max_line_chars=0)
    content = output_path.read_text(encoding="utf-8")
    # normalize_segments always strips outer whitespace, but internal spacing and
    # capitalization are left untouched when cleanup_text=False.
    assert "\nraw   text\n\n" in content


def test_write_srt_creates_parent_directories(tmp_path: Path):
    output_path = tmp_path / "nested" / "dir" / "out.srt"
    write_srt([{"start": 0.0, "end": 1.0, "text": "hi"}], output_path)
    assert output_path.exists()
