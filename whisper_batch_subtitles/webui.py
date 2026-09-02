from __future__ import annotations

import html
import json
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def fetch_dashboard_data(database_path: Path) -> dict[str, Any]:
    """Read-only snapshot of pipeline state for the dashboard.

    Opens its own read-only connection per call rather than sharing one with the
    pipeline's writer -- this module is meant to run as a separate, independent
    process (`serve`) reading the same state.sqlite3 the pipeline writes to. SQLite
    tolerates concurrent readers, but a reader can still transiently hit "database is
    locked" while a write transaction is in flight; that's surfaced as
    `available: False` with an error message instead of raising, so a page refresh a
    moment later just works.
    """
    if not database_path.exists():
        return {"available": False, "error": f"No state database found at {database_path}"}

    try:
        connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True, timeout=2.0)
        connection.row_factory = sqlite3.Row
    except sqlite3.OperationalError as error:
        return {"available": False, "error": str(error)}

    try:
        status_rows = connection.execute(
            "SELECT status, COUNT(*) AS count FROM media_jobs GROUP BY status"
        ).fetchall()
        recent_rows = connection.execute(
            """
            SELECT relative_path, status, stage, last_error, updated_at
            FROM media_jobs ORDER BY updated_at DESC LIMIT 25
            """
        ).fetchall()
        return {
            "available": True,
            "status_counts": {str(row["status"]): int(row["count"]) for row in status_rows},
            "recent_jobs": [dict(row) for row in recent_rows],
        }
    except sqlite3.OperationalError as error:
        return {"available": False, "error": str(error)}
    finally:
        connection.close()


def render_dashboard_html(data: dict[str, Any]) -> str:
    if not data.get("available"):
        error = html.escape(str(data.get("error", "unknown")))
        return (
            "<!doctype html><html><head><title>Whisper Batch Subtitles</title>"
            "<meta http-equiv='refresh' content='5'></head>"
            f"<body><h1>Whisper Batch Subtitles</h1><p>Data temporarily unavailable: {error}</p>"
            "</body></html>"
        )

    status_counts: dict[str, int] = data["status_counts"]
    total = sum(status_counts.values())
    status_rows = "".join(
        f"<tr><td>{html.escape(str(status))}</td><td>{count}</td></tr>"
        for status, count in sorted(status_counts.items())
    )
    job_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(job['relative_path']))}</td>"
        f"<td>{html.escape(str(job['status']))}</td>"
        f"<td>{html.escape(str(job['stage'] or ''))}</td>"
        f"<td>{html.escape(str(job['last_error'] or ''))}</td>"
        f"<td>{html.escape(str(job['updated_at'] or ''))}</td>"
        "</tr>"
        for job in data["recent_jobs"]
    )
    return f"""<!doctype html>
<html>
<head>
<title>Whisper Batch Subtitles</title>
<meta http-equiv="refresh" content="5">
<meta charset="utf-8">
<style>
body {{ font-family: sans-serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; margin-bottom: 2rem; }}
td, th {{ border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: left; }}
h1 {{ margin-bottom: 0.2rem; }}
.subtitle {{ color: #666; margin-top: 0; }}
</style>
</head>
<body>
<h1>Whisper Batch Subtitles</h1>
<p class="subtitle">{total} total tracked jobs. Read-only, auto-refreshes every 5 seconds.</p>
<h2>Status counts</h2>
<table><tr><th>Status</th><th>Count</th></tr>{status_rows}</table>
<h2>Recently updated jobs</h2>
<table><tr><th>File</th><th>Status</th><th>Stage</th><th>Last error</th><th>Updated</th></tr>{job_rows}</table>
</body>
</html>"""


def make_handler(database_path: Path) -> type[BaseHTTPRequestHandler]:
    class DashboardRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - required stdlib method name
            data = fetch_dashboard_data(database_path)
            if self.path.startswith("/api/status"):
                body = json.dumps(data).encode("utf-8")
                content_type = "application/json"
            else:
                body = render_dashboard_html(data).encode("utf-8")
                content_type = "text/html; charset=utf-8"

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass  # quiet by default -- the pipeline's own logger already covers activity

    return DashboardRequestHandler


def run_dashboard_server(database_path: Path, *, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Build (but don't yet serve on) a dashboard HTTP server bound to `host:port`.

    No authentication of any kind -- this is meant for localhost or a trusted network
    only. Do not bind to a public interface without putting a reverse proxy with auth
    in front of it.
    """
    handler_cls = make_handler(database_path)
    return ThreadingHTTPServer((host, port), handler_cls)
