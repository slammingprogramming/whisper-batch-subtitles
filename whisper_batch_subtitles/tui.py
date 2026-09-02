from __future__ import annotations

from typing import Any


def is_available() -> bool:
    try:
        import rich  # noqa: F401
    except ImportError:
        return False
    return True


def build_dashboard(
    snapshot: dict[str, Any],
    queue_parts: dict[str, int],
    current_stage_files: dict[str, str],
) -> Any:
    """Build a multi-panel rich renderable summarizing pipeline progress: overall
    counts/throughput/ETA, per-stage queue depth, and the file each stage is currently
    working on. Pure with respect to rendering -- callers own the `Live` loop."""
    from rich.console import Group
    from rich.table import Table

    completed = snapshot["completed"]
    failed = snapshot["failed"]
    queued = snapshot["queued"]
    skipped = snapshot["skipped"]
    elapsed_hours = snapshot["elapsed_seconds"] / 3600
    throughput = completed / elapsed_hours if elapsed_hours > 0 else 0.0
    remaining = max(queued - completed - failed, 0)
    eta_hours = (remaining / throughput) if throughput > 0 else None
    eta_text = f"{eta_hours:.2f}h" if eta_hours is not None else "unknown"

    summary = Table(title="Whisper Batch Subtitles", show_header=False)
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("Discovered", str(snapshot["discovered"]))
    summary.add_row("Queued", str(queued))
    summary.add_row("Completed", str(completed))
    summary.add_row("Skipped", str(skipped))
    summary.add_row("Failed", str(failed), style="red" if failed else None)
    summary.add_row("Throughput", f"{throughput:.2f} files/hour")
    summary.add_row("ETA", eta_text)

    queues = Table(title="Queue depth")
    queues.add_column("Stage")
    queues.add_column("Depth", justify="right")
    for stage, depth in queue_parts.items():
        queues.add_row(stage, str(depth))

    current = Table(title="Currently processing")
    current.add_column("Stage")
    current.add_column("File")
    if current_stage_files:
        for stage, relative_path in current_stage_files.items():
            current.add_row(stage, relative_path)
    else:
        current.add_row("-", "(idle)")

    return Group(summary, queues, current)


def render_dashboard_to_text(
    snapshot: dict[str, Any],
    queue_parts: dict[str, int],
    current_stage_files: dict[str, str],
    *,
    width: int = 100,
) -> str:
    """Render the dashboard to a plain string -- used for testing without a live
    terminal, and as a fallback if a caller wants the content without rich's own
    live-redraw machinery."""
    import io

    from rich.console import Console

    buffer = io.StringIO()
    console = Console(file=buffer, width=width, force_terminal=False)
    console.print(build_dashboard(snapshot, queue_parts, current_stage_files))
    return buffer.getvalue()
