# Whisper Batch Subtitles

[![CI](https://github.com/slammingprogramming/whisper-batch-subtitles/actions/workflows/ci.yml/badge.svg)](https://github.com/slammingprogramming/whisper-batch-subtitles/actions/workflows/ci.yml)
[![License: AGPL v3 or later](https://img.shields.io/badge/License-AGPLv3--or--later-blue.svg)](LICENSE)

High-throughput batch transcription and subtitle generation for large media libraries, built on `faster-whisper`.

This project is designed for people processing more than a handful of files at a time: channel archives, lecture folders, interview dumps, podcast backlogs, surveillance exports, and other media collections where throughput, resumability, and low babysitting matter.

Instead of treating transcription as a one-off script, Whisper Batch Subtitles treats it as a pipeline:

- discover media recursively
- extract and cache audio
- transcribe with hardware-aware defaults
- optionally translate into one or more languages
- write subtitle outputs
- resume safely after interruption

## Why This Project

Most Whisper batch scripts work well for a quick test, then start to break down when the workload gets large.

This project aims to be more practical for real libraries:

- resumable runs with durable SQLite job state
- persistent caches so work is not repeated unnecessarily
- parallel pipeline stages for better hardware usage
- automatic hardware inspection and runtime recommendations
- CLI, config file, and environment-variable based control
- logs and progress reporting that are useful during long runs

## Features

- Recursive media scanning, with configurable scan order (newest/oldest/largest/smallest first)
- Subtitle generation in `.srt` format, with optional line wrapping, punctuation/whitespace cleanup, and repeated-segment (hallucination-loop) suppression
- Optional translation output such as `.en.srt`, `.es.srt`, `.ja.srt`, via a pluggable backend (Google by default, DeepL optional)
- Faster-Whisper backend, with real multi-GPU routing (each transcription worker pinned to its own CUDA device)
- FFmpeg-based audio extraction
- Hardware-aware defaults for CPU/GPU systems, plus named presets (`fastest`, `balanced`, `archive-quality`, `low-vram`, `cpu-only`) and an interactive setup wizard
- Reusable job profiles (`--profile`/`--save-profile`) for switching between named configurations
- Watch/daemon mode for continuously-monitored folders
- Persistent audio, transcript, and translation caches
- Retry handling with type-aware classification (a corrupt file fails fast instead of wasting retries) and failure tracking
- Content-signature duplicate detection across differently-named/located copies of the same file
- YAML config support, with config file, named profile, and environment-variable layering
- Structured logs, periodic throughput reporting, an optional self-updating live status line, and a real multi-panel terminal (TUI) dashboard
- A built-in benchmarking mode that measures real transcription throughput on your hardware
- Dynamic auto-scaling of extraction (ffmpeg) workers based on observed queue backlog
- A read-only local web dashboard for checking pipeline status from a browser
- Multi-track audio inspection and per-directory role profiles
- Speaker-labeled merged transcripts when separate voice tracks are available
- Optional single-track diarization via `pyannote.audio`
- Optional split-environment diarization via subprocess + JSON handoff
- A Dockerfile for containerized deployment (CPU by default, adaptable for GPU)

## Status

The project is currently at `v1.3`.

It already includes the core pipeline architecture, resumability, caching, hardware detection with real multi-GPU routing and empirical benchmarking, queue-based orchestration with dynamic extraction-worker scaling, multi-track role handling, speaker-aware subtitle output, presets/wizard/profiles, watch mode, pluggable translation, duplicate detection, a real TUI dashboard, a read-only web dashboard, and a Dockerfile. Bigger roadmap items like distributed/multi-machine workers, a standalone executable, and a full job-submission API are still future work — see [Project Direction](#project-direction) below and [CHANGELOG.md](CHANGELOG.md) for what's shipped so far.

## Requirements

- Python 3.10+
- `ffmpeg`
- `ffprobe`

You will also want a CUDA-capable GPU for the best throughput, but CPU-only operation is supported.

## Installation

Install the main transcription environment:

```powershell
pip install -e .
```

If you want everything in one environment, you can also install the optional diarization dependency:

```powershell
pip install -e .[diarization]
```

To use the DeepL translation backend instead of the default Google backend, install its extra too:

```powershell
pip install -e .[deepl]
```

For the `--tui` live terminal dashboard, install the `tui` extra:

```powershell
pip install -e .[tui]
```

If you are not installing as a package yet, the module entrypoint still works from the repo root.

A `Dockerfile` is also provided at the repo root for containerized runs (`docker build -t wbs .`); see the
comments in it for adapting to a GPU/CUDA base image. It ships with sensible defaults but hasn't been
build-tested in every environment, so treat it as a solid starting point to verify on your own setup.

### Recommended Split-Environment Setup

For many Windows CUDA setups, the cleanest architecture is to keep transcription and diarization in separate environments:

1. Main runtime environment
   For `faster-whisper`, FFmpeg extraction, batching, translation, and the normal pipeline
2. Diarization environment
   For `pyannote.audio`, newer Torch builds, and speaker models

Example:

```powershell
# main environment
pip install -e .

# separate diarization environment
py -3.13 -m venv .venv-diarization
.venv-diarization\Scripts\activate
pip install pyannote.audio
pip install -e .
```

Then point the main config at that diarization Python:

```yaml
diarization_mode: auto
diarization_external_python: .venv-diarization\Scripts\python.exe
```

With that setup, the main pipeline hands cached audio to the diarization helper through a subprocess and reads back JSON speaker turns. This avoids Torch version conflicts between the transcription stack and the diarization stack.

## Quick Start

Inspect the local machine and see the recommended runtime settings:

```powershell
python -m whisper_batch_subtitles hardware
```

Generate a starter config tuned to the current machine:

```powershell
python -m whisper_batch_subtitles init-config
```

Or answer a few questions interactively and let the wizard write the config for you:

```powershell
python -m whisper_batch_subtitles wizard
```

Prefer a one-shot named preset instead of the interactive wizard?

```powershell
python -m whisper_batch_subtitles init-config --preset archive-quality
```

Presets: `fastest`, `balanced`, `archive-quality`, `low-vram`, `cpu-only`. You can also pass `--preset`
directly to `run` without writing a config file first.

Inspect a recording's audio streams and see the guessed or saved role mapping:

```powershell
python -m whisper_batch_subtitles inspect-tracks process\example.mkv
```

Run the pipeline against a media folder:

```powershell
python -m whisper_batch_subtitles run --root-dir process
```

The legacy script entrypoint still works:

```powershell
python whispertranscribetranslate.py --root-dir process
```

Save a tuned run as a reusable named profile, then reuse it later:

```powershell
python -m whisper_batch_subtitles run --root-dir process --preset fastest --save-profile podcast
python -m whisper_batch_subtitles run --root-dir process --profile podcast
```

Keep watching a folder and pick up new files automatically instead of a one-shot scan:

```powershell
python -m whisper_batch_subtitles run --root-dir process --watch --watch-interval-seconds 300
```

Measure actual transcription throughput on your hardware instead of trusting static heuristics:

```powershell
python -m whisper_batch_subtitles hardware --benchmark
```

Watch a live terminal dashboard instead of scrolling log lines (needs `pip install -e .[tui]`):

```powershell
python -m whisper_batch_subtitles run --root-dir process --tui
```

Let the number of extraction workers grow and shrink automatically with the workload:

```powershell
python -m whisper_batch_subtitles run --root-dir process --dynamic-ffmpeg-workers --ffmpeg-workers-max 8
```

Check progress from a browser (read-only, no authentication, localhost by default):

```powershell
python -m whisper_batch_subtitles serve
```

## Example

```powershell
python -m whisper_batch_subtitles run `
  --root-dir process `
  --device cuda `
  --model small `
  --batch-size 8 `
  --ffmpeg-workers 2 `
  --translation-workers 4 `
  --target-language en `
  --target-language es `
  --resume
```

## Configuration

Configuration can come from five places, lowest to highest precedence:

- built-in defaults
- a YAML config file (`--config`, default `whisper-batch-subtitles.yaml` in the current directory)
- a named profile (`--profile <name>`, loaded from `profiles/<name>.yaml`)
- environment variables prefixed with `WBS_`
- CLI flags

`run --preset <name>` is separate from profiles: it fills in defaults at CLI-flag precedence, so it beats
the config file/profile/env layers but still loses to anything you also type explicitly on the command
line ("run with this preset, but let me override this one setting").

Example:

```powershell
WBS_MODEL=medium
WBS_TARGET_LANGUAGES=en,es
python -m whisper_batch_subtitles run --root-dir process
```

Create a machine-tuned config template with:

```powershell
python -m whisper_batch_subtitles init-config
```

Some notable options beyond the obvious ones:

- `gpu_device_indices`: which CUDA device index each transcription worker routes to (repeat `--gpu-device`
  on the CLI); auto-populated from all detected GPUs if left unset
- `subtitle_cleanup_text` / `subtitle_max_line_chars` / `subtitle_max_lines`: subtitle text cleanup and
  balanced line-wrapping (on by default)
- `suppress_repeated_segments`: collapse runs of 3+ identical consecutive segments (a known Whisper
  looping failure mode); on by default
- `duplicate_detection`: `off`, `warn`, or `skip` — flag or skip files whose content matches an
  already-completed file elsewhere in the library
- `translation_backend`: `google` (default) or `deepl` (needs `pip install -e .[deepl]` and
  `deepl_api_key`/`DEEPL_API_KEY`)
- `watch` / `watch_interval_seconds`: keep rescanning the folder on an interval instead of exiting after one pass
- `tui`: a live multi-panel terminal dashboard instead of log lines (needs `pip install -e .[tui]`); takes
  precedence over `live_status` if both are set
- `dynamic_ffmpeg_workers` / `ffmpeg_workers_min` / `ffmpeg_workers_max`: auto-scale extraction workers
  within these bounds based on observed queue backlog, instead of a fixed `ffmpeg_workers` count

Important speaker-related options:

- `diarization_mode`: `off`, `auto`, `pyannote`, or `external`
- `speaker_labels`: include speaker labels in subtitle text when available
- `write_role_subtitles`: emit extra files like `.me.srt` and `.others.srt` for multi-track recordings
- `prompt_for_track_roles`: ask once for unknown multi-track layouts and remember the answer per directory
- `diarization_external_python`: Python executable for a separate diarization environment
- `diarization_external_timeout_seconds`: timeout for external diarization helper runs

## Multi-Track Workflows

Some recording setups produce multiple audio streams, for example:

- your microphone
- voice chat or remote participants
- game or desktop audio

When Whisper Batch Subtitles detects multiple audio streams, it can:

- inspect the stream metadata
- guess roles such as `me`, `others`, `mixed`, `system`, or `ignore`
- prompt you to confirm the mapping
- remember that mapping for the directory so the next matching recording is automatic

When separate voice tracks are available, the pipeline transcribes them separately and merges them into a single speaker-labeled subtitle stream. This is often much more accurate than forcing multiple speakers through one mixed track.

## Diarization

There are two speaker-aware paths:

- Multi-track speaker separation: best when your recordings already store separate voice streams
- Single-track diarization: optional backend support through `pyannote.audio`

If `pyannote.audio` is installed in the main environment, `diarization_mode: pyannote` or `auto` can label speakers for single mixed audio tracks directly.

If you want to keep diarization isolated in a different environment, set:

```yaml
diarization_mode: auto
diarization_external_python: .venv-diarization\Scripts\python.exe
```

In that mode, the pipeline launches `whisper_batch_subtitles.diarization_helper` in the external environment, passes it the cached audio path, and reads back a cached JSON file of speaker turns.

If no diarization backend is available, the pipeline falls back gracefully and still works as a normal transcription run.

## Runtime State

By default, the pipeline stores operational state in `.whisper-batch-subtitles/`:

- `state.sqlite3` for durable job metadata
- `cache/audio/` for extracted audio
- `cache/transcripts/` for transcript cache data
- `cache/translations/` for translated subtitle cache data
- `logs/runtime.log` for human-readable logs
- `logs/events.jsonl` for structured log events

This makes reruns much safer and faster on large datasets.

## Project Direction

The long-term goal is to grow this into a reliable media ingestion and transcription platform rather than a simple batch script. The core pipeline architecture (queue-based stages, hardware-aware tuning with real multi-GPU routing and empirical benchmarking, resumable SQLite-backed state, caching, presets/wizard/profiles, watch mode, pluggable translation, duplicate detection, dynamic extraction-worker scaling, a real TUI dashboard, a read-only web dashboard, a Dockerfile) is already in place; what's left is mostly bigger, separate efforts rather than incremental additions:

- dynamic scheduling extended beyond the extraction stage (transcription workers stay statically sized since each pins a real model to a GPU)
- chunked/streaming audio extraction instead of always extracting a full cached file first
- distributed/multi-machine workers
- a standalone executable
- a job-submission REST API, a real web frontend, and a plugin system (the current web dashboard is deliberately read-only — status only, not job management)

Open an issue if you'd like to discuss priorities or pick one of these up.

## Development

Install the dev extras and run the test suite:

```powershell
pip install -e .[dev]
pytest
```

The suite is hermetic (no GPU, no network) except for a handful of `ffmpeg`/`ffprobe` integration tests
that generate a synthetic tone via `lavfi` and auto-skip if those tools aren't on `PATH`.

## Contributing

Issues, ideas, performance reports, and architecture feedback are all welcome. If you test this on unusual hardware or large real-world datasets, that feedback is especially valuable. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and pull request guidelines, and [CHANGELOG.md](CHANGELOG.md) for release history. This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## Security

Please don't file public issues with vulnerability details — see [SECURITY.md](SECURITY.md) for the private disclosure process.

## License

Licensed under the GNU Affero General Public License, version 3 or (at your option) any later version (AGPL-3.0-or-later). See [LICENSE](LICENSE) for the full text.
