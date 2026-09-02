from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from whisper_batch_subtitles.models import AudioStreamInfo, TrackAssignment

TRACK_ROLES = ("me", "others", "mixed", "system", "ignore")

ROLE_LABELS = {
    "me": "Me",
    "others": "Others",
    "mixed": "Mixed",
    "system": "System",
    "ignore": "Ignore",
}

ROLE_KEYWORDS = {
    "me": ("mic", "microphone", "voice", "commentary", "narration", "host", "local", "self"),
    "others": ("chat", "discord", "call", "meeting", "team", "party", "guest", "remote", "class"),
    "system": ("game", "desktop", "system", "music", "bgm", "capture", "console"),
    "mixed": ("mix", "master", "main", "stereo", "program"),
}


def compute_content_signature(media_path: Path, file_size: int, *, sample_bytes: int = 1_048_576) -> str:
    """Cheap heuristic duplicate-detection fingerprint.

    Hashes the file size plus the first and last `sample_bytes` of content instead of the
    whole file, so scanning a large library stays fast. This is a heuristic, not a
    cryptographic guarantee -- two different files could theoretically collide -- but it's
    enough to flag "you probably already have this" for real-world duplicate downloads/copies.
    """
    digest = hashlib.sha1()
    digest.update(str(file_size).encode("utf-8"))
    with media_path.open("rb") as handle:
        head = handle.read(sample_bytes)
        digest.update(head)
        if file_size > sample_bytes:
            handle.seek(max(file_size - sample_bytes, 0))
            tail = handle.read(sample_bytes)
            digest.update(tail)
    return digest.hexdigest()


def probe_audio_streams(media_path: Path) -> list[AudioStreamInfo]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_streams",
        "-of",
        "json",
        str(media_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {media_path}: {completed.stderr.strip()}")

    payload = json.loads(completed.stdout or "{}")
    streams: list[AudioStreamInfo] = []
    for stream in payload.get("streams", []):
        tags = stream.get("tags") or {}
        disposition = stream.get("disposition") or {}
        streams.append(
            AudioStreamInfo(
                index=int(stream["index"]),
                codec_name=stream.get("codec_name"),
                channels=_safe_int(stream.get("channels")),
                channel_layout=stream.get("channel_layout"),
                language=tags.get("language"),
                title=tags.get("title"),
                disposition_default=bool(disposition.get("default")),
            )
        )
    return streams


def build_stream_signature(streams: list[AudioStreamInfo]) -> str:
    signature_payload = [
        {
            "index": stream.index,
            "codec_name": stream.codec_name,
            "channels": stream.channels,
            "channel_layout": stream.channel_layout,
            "language": stream.language,
            "title": stream.title,
            "default": stream.disposition_default,
        }
        for stream in streams
    ]
    return json.dumps(signature_payload, ensure_ascii=True, sort_keys=True)


def describe_stream(stream: AudioStreamInfo) -> str:
    title = stream.title or "untitled"
    language = stream.language or "unknown"
    channels = stream.channels if stream.channels is not None else "?"
    layout = stream.channel_layout or "unknown-layout"
    default_flag = " default" if stream.disposition_default else ""
    return (
        f"stream={stream.index} title={title!r} lang={language} "
        f"channels={channels} layout={layout}{default_flag}"
    )


def guess_track_assignments(streams: list[AudioStreamInfo]) -> list[TrackAssignment]:
    if not streams:
        return []
    if len(streams) == 1:
        only = streams[0]
        return [
            TrackAssignment(
                stream_index=only.index,
                role="mixed",
                label=ROLE_LABELS["mixed"],
                stream_title=only.title,
            )
        ]

    assignments: list[TrackAssignment] = []
    used_roles: set[str] = set()
    for stream in streams:
        role = _guess_role_for_stream(stream)
        if role == "mixed" and "mixed" in used_roles:
            role = "ignore"
        used_roles.add(role)
        assignments.append(
            TrackAssignment(
                stream_index=stream.index,
                role=role,
                label=ROLE_LABELS[role],
                stream_title=stream.title,
            )
        )

    if all(assignment.role == "ignore" for assignment in assignments):
        assignments[0].role = "mixed"
        assignments[0].label = ROLE_LABELS["mixed"]
    return assignments


def normalize_track_assignments(
    streams: list[AudioStreamInfo], raw_assignments: list[dict[str, str | int]]
) -> list[TrackAssignment]:
    streams_by_index = {stream.index: stream for stream in streams}
    assignments: list[TrackAssignment] = []
    for raw_assignment in raw_assignments:
        stream_index = int(raw_assignment["stream_index"])
        role = str(raw_assignment["role"]).lower()
        if stream_index not in streams_by_index:
            continue
        if role not in TRACK_ROLES:
            continue
        label = str(raw_assignment.get("label") or ROLE_LABELS.get(role, role.title()))
        stream = streams_by_index[stream_index]
        assignments.append(
            TrackAssignment(
                stream_index=stream_index,
                role=role,
                label=label,
                stream_title=stream.title,
            )
        )
    return assignments


def serialize_track_assignments(assignments: list[TrackAssignment]) -> list[dict[str, str | int]]:
    return [
        {
            "stream_index": assignment.stream_index,
            "role": assignment.role,
            "label": assignment.label,
            "stream_title": assignment.stream_title,
        }
        for assignment in assignments
    ]


def merge_role_segments(
    role_segments: dict[str, list[dict[str, object]]],
    labels: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    labels = labels or {}
    for role, segments in role_segments.items():
        label = labels.get(role, ROLE_LABELS.get(role, role.title()))
        for segment in segments:
            merged.append(
                {
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "text": str(segment["text"]).strip(),
                    "speaker": role,
                    "speaker_label": label,
                }
            )
    merged.sort(key=lambda segment: (float(segment["start"]), float(segment["end"])))
    return merged


def _guess_role_for_stream(stream: AudioStreamInfo) -> str:
    haystack = " ".join(
        part.lower()
        for part in (
            stream.title or "",
            stream.language or "",
            stream.channel_layout or "",
            stream.codec_name or "",
        )
        if part
    )
    for role, keywords in ROLE_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return role
    if stream.disposition_default:
        return "mixed"
    return "ignore"


def _safe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
