from __future__ import annotations

from whisper_batch_subtitles.media import (
    build_stream_signature,
    describe_stream,
    guess_track_assignments,
    merge_role_segments,
    normalize_track_assignments,
    serialize_track_assignments,
)
from whisper_batch_subtitles.models import AudioStreamInfo, TrackAssignment


def stream(index, title=None, language=None, channel_layout=None, codec_name="aac", default=False):
    return AudioStreamInfo(
        index=index,
        codec_name=codec_name,
        channels=2,
        channel_layout=channel_layout,
        language=language,
        title=title,
        disposition_default=default,
    )


def test_guess_track_assignments_empty():
    assert guess_track_assignments([]) == []


def test_guess_track_assignments_single_stream_is_always_mixed():
    [assignment] = guess_track_assignments([stream(0, title="Anything")])
    assert assignment.role == "mixed"
    assert assignment.stream_index == 0


def test_guess_track_assignments_uses_keywords():
    streams = [
        stream(0, title="Microphone"),
        stream(1, title="Discord Call"),
        stream(2, title="Game Audio"),
    ]
    assignments = {a.stream_index: a.role for a in guess_track_assignments(streams)}
    assert assignments[0] == "me"
    assert assignments[1] == "others"
    assert assignments[2] == "system"


def test_guess_track_assignments_second_mixed_candidate_becomes_ignore():
    # Two streams that would both guess "mixed" (via default disposition,
    # no keyword match) -- only the first should keep the role.
    streams = [stream(0, default=True), stream(1, default=True)]
    assignments = guess_track_assignments(streams)
    roles = [a.role for a in assignments]
    assert roles.count("mixed") == 1
    assert roles.count("ignore") == 1


def test_guess_track_assignments_never_returns_all_ignore():
    # No keywords, no default disposition -- everything would guess "ignore",
    # but the function must promote at least one stream to "mixed".
    streams = [stream(0), stream(1)]
    assignments = guess_track_assignments(streams)
    assert any(a.role == "mixed" for a in assignments)


def test_normalize_track_assignments_filters_unknown_stream_and_role():
    streams = [stream(0), stream(1)]
    raw = [
        {"stream_index": 0, "role": "me", "label": "Me"},
        {"stream_index": 99, "role": "me", "label": "Me"},  # unknown stream, dropped
        {"stream_index": 1, "role": "bogus", "label": "???"},  # unknown role, dropped
    ]
    assignments = normalize_track_assignments(streams, raw)
    assert len(assignments) == 1
    assert assignments[0].stream_index == 0
    assert assignments[0].role == "me"


def test_serialize_round_trips_through_normalize():
    streams = [stream(0, title="Mic"), stream(1, title="Chat")]
    original = guess_track_assignments(streams)
    serialized = serialize_track_assignments(original)
    restored = normalize_track_assignments(streams, serialized)
    assert [a.role for a in restored] == [a.role for a in original]


def test_build_stream_signature_stable_and_order_sensitive():
    streams_a = [stream(0, title="Mic"), stream(1, title="Chat")]
    streams_b = [stream(0, title="Mic"), stream(1, title="Chat")]
    assert build_stream_signature(streams_a) == build_stream_signature(streams_b)

    streams_c = [stream(0, title="Different")]
    assert build_stream_signature(streams_a) != build_stream_signature(streams_c)


def test_describe_stream_handles_missing_metadata():
    text = describe_stream(AudioStreamInfo(index=3))
    assert "stream=3" in text
    assert "title='untitled'" in text
    assert "lang=unknown" in text


def test_merge_role_segments_sorts_by_start_time():
    role_segments = {
        "others": [{"start": 5.0, "end": 6.0, "text": "later"}],
        "me": [{"start": 1.0, "end": 2.0, "text": "earlier"}],
    }
    merged = merge_role_segments(role_segments, labels={"me": "Me", "others": "Others"})
    assert [segment["text"] for segment in merged] == ["earlier", "later"]
    assert merged[0]["speaker"] == "me"
    assert merged[0]["speaker_label"] == "Me"
