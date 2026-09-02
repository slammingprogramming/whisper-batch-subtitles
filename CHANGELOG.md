# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/).

## [1.3.0]

### Added

- Benchmarking mode (`hardware --benchmark`): transcribes a synthetic tone with VAD disabled to measure
  real transcription throughput on the current hardware, with a batch-size suggestion if it's struggling.
- A real multi-panel terminal (TUI) dashboard (`--tui`, optional `rich` extra): live counts, throughput,
  ETA, per-stage queue depth, and current file per stage, replacing scrolling log lines.
- Dynamic auto-scaling of extraction (ffmpeg) workers (`--dynamic-ffmpeg-workers`,
  `--ffmpeg-workers-min`/`--ffmpeg-workers-max`), bounded and based on observed queue backlog.
- A read-only local web dashboard (`serve` command): status counts and recently-updated jobs as HTML or
  JSON, reading `state.sqlite3` directly. No authentication — localhost by default.
- A `Dockerfile` and `.dockerignore` for containerized runs (CPU-focused by default).

## [1.2.0]

### Added

- Real multi-GPU device routing: each transcription worker is pinned to its own CUDA `device_index`,
  round-robined across detected/configured GPUs.
- Named presets (`fastest`, `balanced`, `archive-quality`, `low-vram`, `cpu-only`) and an interactive
  `wizard` command for first-time setup.
- Watch/daemon mode (`--watch`, `--watch-interval-seconds`) for continuously-monitored folders.
- Reusable named job profiles (`--profile`, `--save-profile`), loaded from `profiles/<name>.yaml` as a
  config layer between the main config file and environment variables.
- Content-signature duplicate detection (`duplicate_detection: off|warn|skip`) across differently-named or
  differently-located copies of the same file.
- Pluggable translation backends: Google (default, unchanged) and an optional DeepL backend.
- Subtitle formatting quality: text cleanup and balanced line-wrapping, on by default.
- Repeated-segment suppression, targeting a known Whisper hallucination-loop failure mode.
- Type-aware retry classification: permanent errors (corrupt files, missing streams) fail fast instead of
  wasting retry attempts.
- An opt-in single-line live status display (`--live-status`).

### Fixed

- Discovery no longer crashes the entire batch run when a single file fails to inspect (e.g. corrupt
  media) — that file is now logged and skipped, and the run continues.
- A multi-track job with every stream assigned role `ignore` now fails loudly instead of silently
  completing with an empty subtitle file.
- The translator no longer silently discards a translation on a batch-result length mismatch; it now
  retries per-segment, consistent with how it already handled outright batch failures.
- The diarization cache is now keyed by diarizer identity (mode + model), so switching `pyannote_model` or
  `diarization_mode` between runs no longer silently reuses stale speaker-turn data from a different
  backend.
- The shared single-track diarizer instance is now lock-protected against concurrent access from multiple
  transcription worker threads.
- `ffmpeg` extraction failures now surface the real `ffmpeg` stderr output instead of a bare, undiagnostic
  subprocess error.

## [1.1.0] and earlier

Baseline feature set: recursive media discovery, hardware-aware model/worker defaults, a resumable
SQLite-backed job pipeline, persistent audio/transcript/translation caches, multi-track audio inspection
and per-directory role profiles with speaker-labeled merged transcripts, optional single-track diarization
via `pyannote.audio` (in-process or via a separate split environment), YAML/environment-variable
configuration, and structured logging with periodic progress reporting.
