from __future__ import annotations

import argparse
import builtins
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

from whisper_batch_subtitles.config import build_config, config_to_yaml_text, profile_path, sample_config_text
from whisper_batch_subtitles.hardware import (
    detect_hardware,
    format_hardware_report,
    recommend_runtime,
)
from whisper_batch_subtitles.logging_utils import setup_logging
from whisper_batch_subtitles.media import (
    build_stream_signature,
    describe_stream,
    guess_track_assignments,
    probe_audio_streams,
    serialize_track_assignments,
)
from whisper_batch_subtitles.pipeline import PipelineRunner
from whisper_batch_subtitles.presets import preset_names, preset_overrides
from whisper_batch_subtitles.state import StateStore

_COMMANDS = {"run", "hardware", "init-config", "inspect-tracks", "wizard", "serve"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="High-throughput, resumable transcription pipeline powered by faster-whisper."
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the transcription pipeline")
    _add_run_arguments(run_parser)

    hardware_parser = subparsers.add_parser("hardware", help="Show detected hardware and runtime recommendations")
    hardware_parser.add_argument("--config", type=str, default=None, help="Optional config file path")
    hardware_parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a short synthetic transcription benchmark using the recommended model/device/compute "
        "settings and report actual observed throughput (downloads the model on first use)",
    )
    hardware_parser.add_argument(
        "--benchmark-model",
        type=str,
        default=None,
        help="Override the model used for the benchmark (default: the recommended model)",
    )
    hardware_parser.add_argument(
        "--benchmark-duration-seconds",
        type=float,
        default=20.0,
        help="Length of the synthetic benchmark clip",
    )

    init_parser = subparsers.add_parser("init-config", help="Write a starter YAML config file")
    init_parser.add_argument(
        "--output",
        type=str,
        default="whisper-batch-subtitles.yaml",
        help="Where to write the starter config",
    )
    init_parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=preset_names(),
        help="Apply a named preset on top of the hardware-detected defaults",
    )

    inspect_parser = subparsers.add_parser("inspect-tracks", help="Inspect audio tracks and saved role mappings")
    inspect_parser.add_argument("media_path", type=str, help="Media file to inspect")
    inspect_parser.add_argument("--config", type=str, default=None, help="Optional config file path")

    subparsers.add_parser("wizard", help="Interactively choose a preset and write a starter config")

    serve_parser = subparsers.add_parser(
        "serve", help="Serve a read-only local web dashboard of pipeline status (no authentication)"
    )
    serve_parser.add_argument("--config", type=str, default=None, help="Optional config file path (to locate the state DB)")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the dashboard server to")
    serve_parser.add_argument("--port", type=int, default=8765, help="Port to bind the dashboard server to")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    normalized_argv = _normalize_argv(list(sys.argv[1:]) if argv is None else argv)
    args = parser.parse_args(normalized_argv)

    if args.command == "hardware":
        return _run_hardware_command(args)
    if args.command == "init-config":
        return _run_init_config(args.output, preset_name=args.preset)
    if args.command == "inspect-tracks":
        return _run_inspect_tracks(args)
    if args.command == "wizard":
        return _run_wizard()
    if args.command == "serve":
        return _run_serve(args)
    return _run_pipeline_command(args)


def _normalize_argv(argv: list[str] | None) -> list[str]:
    tokens = list(argv or [])
    if not tokens:
        return ["run"]
    if tokens[0] in {"-h", "--help"}:
        return tokens
    if tokens[0] not in _COMMANDS:
        return ["run", *tokens]
    return tokens


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="YAML config file path")
    parser.add_argument("--profile", type=str, default=None, help="Load a named profile from profiles/<name>.yaml")
    parser.add_argument(
        "--save-profile",
        type=str,
        default=None,
        help="Write the effective resolved config to profiles/<name>.yaml and exit, without running the pipeline",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=preset_names(),
        help="Apply a named preset on top of the hardware-detected defaults",
    )
    parser.add_argument("--root-dir", type=str, default=None, help="Root folder to scan recursively")
    parser.add_argument("--state-dir", type=str, default=None, help="Directory for caches, logs, and SQLite state")
    parser.add_argument("--state-db", type=str, default=None, help="Explicit SQLite database path")
    parser.add_argument("--model", type=str, default=None, help="Whisper model name")
    parser.add_argument("--device", type=str, default=None, help="Whisper device, such as cuda or cpu")
    parser.add_argument("--compute-type", type=str, default=None, help="Whisper compute type, such as float16 or int8")
    parser.add_argument("--language", type=str, default=None, help="Force a known source language")
    parser.add_argument(
        "--gpu-device",
        dest="gpu_device_indices",
        action="append",
        type=int,
        default=None,
        help="CUDA device index to route a transcription worker to; repeat for multiple GPUs "
        "(default: auto-detected from all visible GPUs)",
    )
    parser.add_argument("--translate", dest="translate", action="store_true", help="Enable translation outputs")
    parser.add_argument("--no-translate", dest="translate", action="store_false", help="Disable translation outputs")
    parser.set_defaults(translate=None)
    parser.add_argument(
        "--target-language",
        dest="target_languages",
        action="append",
        default=None,
        help="Translation target language; repeat for multiple outputs",
    )
    parser.add_argument(
        "--translation-backend",
        type=str,
        default=None,
        choices=["google", "deepl"],
        help="Translation backend to use",
    )
    parser.add_argument(
        "--deepl-api-key",
        type=str,
        default=None,
        help="API key for the DeepL translation backend (or set DEEPL_API_KEY)",
    )
    parser.add_argument(
        "--diarization-mode",
        type=str,
        default=None,
        choices=["off", "auto", "pyannote", "external"],
        help="Speaker diarization mode for single-track audio",
    )
    parser.add_argument("--speaker-labels", dest="speaker_labels", action="store_true", help="Write speaker labels when available")
    parser.add_argument("--no-speaker-labels", dest="speaker_labels", action="store_false", help="Disable speaker labels in subtitle text")
    parser.set_defaults(speaker_labels=None)
    parser.add_argument(
        "--write-role-subtitles",
        dest="write_role_subtitles",
        action="store_true",
        help="Write per-role SRT files for multi-track recordings",
    )
    parser.add_argument(
        "--no-write-role-subtitles",
        dest="write_role_subtitles",
        action="store_false",
        help="Disable per-role SRT outputs",
    )
    parser.set_defaults(write_role_subtitles=None)
    parser.add_argument(
        "--prompt-for-track-roles",
        dest="prompt_for_track_roles",
        action="store_true",
        help="Prompt once for unknown multi-track layouts and remember the result",
    )
    parser.add_argument(
        "--no-prompt-for-track-roles",
        dest="prompt_for_track_roles",
        action="store_false",
        help="Do not prompt for multi-track layouts; rely on saved profiles and heuristics",
    )
    parser.set_defaults(prompt_for_track_roles=None)
    parser.add_argument("--pyannote-model", type=str, default=None, help="Optional pyannote model name for diarization")
    parser.add_argument("--pyannote-auth-token", type=str, default=None, help="Optional pyannote auth token")
    parser.add_argument(
        "--diarization-external-python",
        type=str,
        default=None,
        help="Python executable for a separate diarization environment",
    )
    parser.add_argument(
        "--diarization-external-timeout-seconds",
        type=int,
        default=None,
        help="Timeout for external diarization helper calls",
    )
    parser.add_argument("--ffmpeg-workers", type=int, default=None, help="Parallel FFmpeg extraction workers")
    parser.add_argument(
        "--dynamic-ffmpeg-workers",
        dest="dynamic_ffmpeg_workers",
        action="store_true",
        help="Auto-scale the number of FFmpeg extraction workers within "
        "[--ffmpeg-workers-min, --ffmpeg-workers-max] based on observed queue backlog",
    )
    parser.add_argument(
        "--no-dynamic-ffmpeg-workers", dest="dynamic_ffmpeg_workers", action="store_false", help="Use a fixed extraction worker count"
    )
    parser.set_defaults(dynamic_ffmpeg_workers=None)
    parser.add_argument(
        "--ffmpeg-workers-min", type=int, default=None, help="Minimum extraction workers when dynamic scaling is on"
    )
    parser.add_argument(
        "--ffmpeg-workers-max", type=int, default=None, help="Maximum extraction workers when dynamic scaling is on"
    )
    parser.add_argument("--transcription-workers", type=int, default=None, help="Parallel transcription workers")
    parser.add_argument("--translation-workers", type=int, default=None, help="Parallel translation workers")
    parser.add_argument(
        "--ffmpeg-threads-per-worker",
        type=int,
        default=None,
        help="FFmpeg threads assigned to each extraction worker",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batched inference size for faster-whisper")
    parser.add_argument("--chunk-length", type=int, default=None, help="Chunk length passed into faster-whisper")
    parser.add_argument("--queue-size", type=int, default=None, help="Maximum in-flight jobs per stage queue")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", help="Skip outputs already present")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", help="Do not skip existing outputs")
    parser.set_defaults(skip_existing=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs and ignore existing artifacts")
    parser.add_argument("--resume", dest="resume", action="store_true", help="Resume from cached audio/transcripts")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore caches and resume state")
    parser.set_defaults(resume=None)
    parser.add_argument("--max-retries", type=int, default=None, help="Retries per stage before marking failed")
    parser.add_argument("--vad-filter", dest="vad_filter", action="store_true", help="Enable VAD filtering")
    parser.add_argument("--no-vad-filter", dest="vad_filter", action="store_false", help="Disable VAD filtering")
    parser.set_defaults(vad_filter=None)
    parser.add_argument("--beam-size", type=int, default=None, help="Whisper beam size")
    parser.add_argument("--sample-rate", type=int, default=None, help="Audio extraction sample rate")
    parser.add_argument("--audio-codec", type=str, default=None, help="FFmpeg audio codec for cached extraction")
    parser.add_argument(
        "--subtitle-max-line-chars",
        type=int,
        default=None,
        help="Wrap subtitle lines at this many characters (0 disables wrapping)",
    )
    parser.add_argument(
        "--subtitle-max-lines",
        type=int,
        default=None,
        help="Maximum lines per subtitle cue when wrapping",
    )
    parser.add_argument(
        "--subtitle-cleanup-text",
        dest="subtitle_cleanup_text",
        action="store_true",
        help="Clean up subtitle text (whitespace, punctuation spacing, capitalization)",
    )
    parser.add_argument(
        "--no-subtitle-cleanup-text",
        dest="subtitle_cleanup_text",
        action="store_false",
        help="Write subtitle text exactly as transcribed/translated",
    )
    parser.set_defaults(subtitle_cleanup_text=None)
    parser.add_argument(
        "--suppress-repeated-segments",
        dest="suppress_repeated_segments",
        action="store_true",
        help="Collapse runs of 3+ identical consecutive segments (a known Whisper looping failure mode)",
    )
    parser.add_argument(
        "--no-suppress-repeated-segments",
        dest="suppress_repeated_segments",
        action="store_false",
        help="Disable repeated-segment suppression",
    )
    parser.set_defaults(suppress_repeated_segments=None)
    parser.add_argument(
        "--duplicate-detection",
        type=str,
        default=None,
        choices=["off", "warn", "skip"],
        help="How to handle files whose content matches an already-completed file elsewhere",
    )
    parser.add_argument("--log-level", type=str, default=None, help="Logging level")
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=None,
        help="How often to emit queue and throughput progress lines",
    )
    parser.add_argument(
        "--live-status",
        dest="live_status",
        action="store_true",
        help="Redraw a single self-updating status line instead of scrolling progress log lines "
        "(requires a VT100-capable terminal)",
    )
    parser.add_argument("--no-live-status", dest="live_status", action="store_false", help="Disable live status line")
    parser.set_defaults(live_status=None)
    parser.add_argument(
        "--tui",
        dest="tui",
        action="store_true",
        help="Show a live multi-panel terminal dashboard (counts, queue depth, current file per stage) "
        "instead of scrolling log lines. Requires the 'rich' package (pip install -e .[tui]); "
        "takes precedence over --live-status if both are set",
    )
    parser.add_argument("--no-tui", dest="tui", action="store_false", help="Disable the TUI dashboard")
    parser.set_defaults(tui=None)
    parser.add_argument(
        "--scan-order",
        type=str,
        default=None,
        choices=["path", "newest", "oldest", "largest", "smallest"],
        help="Discovery ordering strategy",
    )
    parser.add_argument(
        "--watch",
        dest="watch",
        action="store_true",
        help="After a scan finishes, wait and rescan the folder repeatedly instead of exiting",
    )
    parser.add_argument("--no-watch", dest="watch", action="store_false", help="Disable watch mode")
    parser.set_defaults(watch=None)
    parser.add_argument(
        "--watch-interval-seconds",
        type=float,
        default=None,
        help="Seconds to wait between rescans in watch mode",
    )


def _run_hardware_command(args: argparse.Namespace) -> int:
    profile = detect_hardware()
    recommendation = recommend_runtime(profile)
    print(format_hardware_report(profile, recommendation))
    if getattr(args, "benchmark", False):
        from whisper_batch_subtitles.benchmark import format_benchmark_report, run_benchmark

        print()
        print("Running benchmark (this downloads the model on first use)...")
        result = run_benchmark(
            model_name=args.benchmark_model or recommendation.model,
            device=recommendation.device,
            compute_type=recommendation.compute_type,
            batch_size=recommendation.batch_size,
            audio_duration_seconds=args.benchmark_duration_seconds,
        )
        print(format_benchmark_report(result, current_batch_size=recommendation.batch_size))
    return 0


def _run_init_config(output_path: str, *, preset_name: str | None = None) -> int:
    profile = detect_hardware()
    recommendation = recommend_runtime(profile)
    overrides = preset_overrides(preset_name)
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        sample_config_text(recommendation=recommendation, profile=profile, overrides=overrides),
        encoding="utf-8",
    )
    print(f"Wrote starter config to {path}")
    return 0


def _resolve_wizard_preset_choice(raw_choice: str, names: list[str], default_index: int) -> str:
    if not raw_choice:
        return names[default_index - 1]
    if raw_choice.isdigit():
        index = int(raw_choice)
        if 1 <= index <= len(names):
            return names[index - 1]
    if raw_choice in names:
        return raw_choice
    return names[default_index - 1]


def wizard_overrides(preset_name: str, *, translate: bool, target_languages: list[str]) -> dict[str, Any]:
    overrides = preset_overrides(preset_name)
    overrides["translate"] = translate
    if target_languages:
        overrides["target_languages"] = target_languages
    return overrides


def _run_wizard(*, input_fn: Callable[[str], str] = builtins.input) -> int:
    profile = detect_hardware()
    recommendation = recommend_runtime(profile)
    print(format_hardware_report(profile, recommendation))
    print()

    names = preset_names()
    print("Choose a preset:")
    for index, name in enumerate(names, start=1):
        print(f"  {index}. {name}")
    default_index = names.index("balanced") + 1 if "balanced" in names else 1
    raw_choice = input_fn(f"Preset [{default_index}]: ").strip()
    preset_name = _resolve_wizard_preset_choice(raw_choice, names, default_index)

    raw_translate = input_fn("Enable translation? [Y/n]: ").strip().lower()
    translate = raw_translate not in {"n", "no"}
    target_languages: list[str] = []
    if translate:
        raw_languages = input_fn("Target languages, comma-separated [en]: ").strip()
        target_languages = [part.strip() for part in raw_languages.split(",") if part.strip()] or ["en"]

    raw_output = input_fn("Config output path [whisper-batch-subtitles.yaml]: ").strip()
    output_path = raw_output or "whisper-batch-subtitles.yaml"

    overrides = wizard_overrides(preset_name, translate=translate, target_languages=target_languages)
    text = sample_config_text(recommendation=recommendation, profile=profile, overrides=overrides)
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Wrote config to {path} (preset: {preset_name})")
    return 0


def _run_inspect_tracks(args: argparse.Namespace) -> int:
    media_path = Path(args.media_path).expanduser()
    if not media_path.exists():
        raise SystemExit(f"Media path does not exist: {media_path}")

    config = build_config({"config": args.config} if args.config is not None else {})
    state = StateStore(config.state_db)
    try:
        streams = probe_audio_streams(media_path)
        if not streams:
            print(f"No audio streams found in {media_path}")
            return 0

        signature = build_stream_signature(streams)
        guessed = guess_track_assignments(streams)
        stored = state.load_track_profile(media_path.parent, signature, config.root_dir)

        print(f"Media: {media_path}")
        print("Audio streams:")
        for stream in streams:
            print(f"  - {describe_stream(stream)}")
        print()
        print("Guessed role mapping:")
        for assignment in guessed:
            print(f"  - stream {assignment.stream_index}: {assignment.role} ({assignment.label})")
        print()
        if stored:
            print("Saved role mapping:")
            for assignment in stored:
                print(
                    f"  - stream {assignment['stream_index']}: {assignment['role']} "
                    f"({assignment.get('label') or assignment['role']})"
                )
        else:
            print("Saved role mapping: none")
            print("This layout will be remembered automatically after the first interactive or heuristic assignment.")
        return 0
    finally:
        state.close()


def _run_serve(args: argparse.Namespace) -> int:
    from whisper_batch_subtitles.webui import run_dashboard_server

    config = build_config({"config": args.config} if args.config is not None else {})
    server = run_dashboard_server(config.state_db, host=args.host, port=args.port)
    print(f"Serving read-only status dashboard at http://{args.host}:{args.port}/ (Ctrl+C to stop)")
    print("No authentication -- do not expose this beyond localhost/a trusted network.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _run_watch_loop(
    *,
    run_once: Callable[[], int],
    interval_seconds: float,
    logger: logging.Logger,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    try:
        while True:
            run_once()
            sleep_fn(interval_seconds)
    except KeyboardInterrupt:
        logger.info("Watch mode stopped by user.")
        return 0


def _run_pipeline_command(args: argparse.Namespace) -> int:
    _validate_tooling()
    profile = detect_hardware()
    recommendation = recommend_runtime(profile)

    cli_values = _namespace_to_config_values(args)
    save_profile_name = cli_values.get("save_profile")
    preset_values = preset_overrides(cli_values.get("preset"))

    config = build_config({**preset_values, **cli_values}, recommendation=recommendation)

    if not config.gpu_device_indices and config.device == "cuda" and profile.gpus:
        config.gpu_device_indices = [gpu.index for gpu in profile.gpus]

    config.ensure_directories()

    if save_profile_name:
        target = profile_path(save_profile_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(config_to_yaml_text(config), encoding="utf-8")
        print(f"Saved profile to {target}")
        return 0

    if config.tui:
        from whisper_batch_subtitles.tui import is_available as tui_is_available

        if not tui_is_available():
            raise SystemExit("--tui requires the 'rich' package. Install it with: pip install -e .[tui]")

    setup_logging(
        config.log_level,
        text_log_path=config.text_log_path,
        json_log_path=config.json_log_path,
        console_output=not (config.live_status or config.tui),
    )
    logger = logging.getLogger("whisper_batch_subtitles")
    logger.info("Hardware profile:\n%s", format_hardware_report(profile, recommendation))
    logger.info(
        "Runtime config | root_dir=%s model=%s device=%s compute_type=%s translate=%s targets=%s",
        config.root_dir,
        config.model,
        config.device,
        config.compute_type,
        config.translate,
        ",".join(config.effective_target_languages) if config.effective_target_languages else "none",
    )

    state = StateStore(config.state_db)
    try:
        def run_once() -> int:
            failures = PipelineRunner(config, state).run()
            counts = state.status_counts()
            logger.info("State summary: %s", counts)
            return failures

        if not config.watch:
            failures = run_once()
            return 0 if failures == 0 else 1

        logger.info(
            "Watch mode enabled: rescanning every %.0f seconds. Press Ctrl+C to stop.",
            config.watch_interval_seconds,
        )
        return _run_watch_loop(run_once=run_once, interval_seconds=config.watch_interval_seconds, logger=logger)
    finally:
        state.close()


def _namespace_to_config_values(args: argparse.Namespace) -> dict[str, Any]:
    values = vars(args).copy()
    values.pop("command", None)
    values = {key: value for key, value in values.items() if value is not None}
    return values


def _validate_tooling() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise SystemExit(f"Missing required external tools: {', '.join(missing)}")
