from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pyannote diarization and write speaker turns to JSON.")
    parser.add_argument("--audio-path", required=True, type=str, help="Input WAV/audio path")
    parser.add_argument("--output-path", required=True, type=str, help="Output JSON path")
    parser.add_argument("--model", required=True, type=str, help="pyannote model name")
    parser.add_argument("--device", default="cpu", type=str, help="Diarization device")
    parser.add_argument("--auth-token", default=None, type=str, help="Hugging Face auth token for pyannote")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        from pyannote.audio import Pipeline  # type: ignore
        import torch  # type: ignore
    except ImportError as error:
        raise SystemExit(
            "pyannote.audio is not installed in this environment. Install the diarization environment dependencies first."
        ) from error

    auth_token = args.auth_token or os.environ.get("PYANNOTE_AUTH_TOKEN")
    if not auth_token:
        raise SystemExit("PYANNOTE_AUTH_TOKEN is required for the pyannote diarization helper.")

    pipeline = Pipeline.from_pretrained(args.model, use_auth_token=auth_token)
    if args.device == "cuda":
        pipeline.to(torch.device("cuda"))

    diarization = pipeline(args.audio_path)
    speaker_names: dict[str, str] = {}
    turns: list[dict[str, object]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        label = speaker_names.setdefault(speaker, f"Speaker {len(speaker_names) + 1}")
        turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker,
                "speaker_label": label,
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "audio_path": str(Path(args.audio_path)),
                "model": args.model,
                "device": args.device,
                "turns": turns,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
