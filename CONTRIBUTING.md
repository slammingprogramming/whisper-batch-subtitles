# Contributing

Thanks for considering a contribution. This is a small, actively-developed project — issues, bug reports,
and pull requests are all welcome.

## Before you start

- **Security vulnerabilities**: see [SECURITY.md](SECURITY.md) — please don't file a public issue with
  vulnerability details.
- **Bigger changes**: for anything nontrivial (a new pipeline stage, a new backend, a schema change), open
  an issue first to discuss the approach before writing code. It's much easier to align on design before a
  PR exists than to rework a finished one.
- **Code of Conduct**: this project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## Development setup

Requires Python 3.10+ and `ffmpeg`/`ffprobe` on your `PATH`.

```bash
git clone <this-repo>
cd whisper-batch-subtitles
pip install -e .[dev]
```

Optional extras, if you're touching that area of the code:

```bash
pip install -e .[dev,diarization]   # pyannote.audio-based speaker diarization
pip install -e .[dev,deepl]         # DeepL translation backend
pip install -e .[dev,tui]           # rich-based --tui dashboard
```

## Running the tests

```bash
pytest
```

The suite is hermetic (no GPU, no network, no real model downloads) except a handful of `ffmpeg`/`ffprobe`
integration tests that generate a synthetic tone and auto-skip if those tools aren't on `PATH`. All tests
should pass before you open a PR; please add tests for new behavior and for any bug you fix.

## Code style

There's no enforced linter/formatter configured yet — match the existing style by eye:

- `from __future__ import annotations` and full type hints in every module
- `@dataclass(slots=True)` for data containers
- `pathlib.Path` everywhere, no raw string paths in logic
- `threading`/`queue.Queue` for concurrency, not `asyncio` or `multiprocessing`
- Optional dependencies (`pyannote.audio`, `deepl`, `rich`) are always imported lazily inside the function
  that needs them, never at module top level, so a plain `pip install -e .` stays fully functional

## Pull requests

- Keep PRs focused — one logical change per PR is easier to review than a bundle of unrelated fixes.
- Include or update tests for the behavior you're changing.
- Describe *why* the change is needed, not just what it does — the code itself already shows what changed.
- Make sure `pytest` passes locally before opening the PR; CI will also run it.

## Questions

Open a [discussion or issue](../../issues) for anything that isn't a bug report or a security concern.
