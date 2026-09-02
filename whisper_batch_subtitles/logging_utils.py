from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "event"):
            payload["event"] = record.event
        if hasattr(record, "fields"):
            payload["fields"] = record.fields
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def setup_logging(
    level: str, *, text_log_path: Path | None, json_log_path: Path | None, console_output: bool = True
) -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        root_logger.addHandler(console_handler)

    if text_log_path is not None:
        text_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(text_log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)

    if json_log_path is not None:
        json_log_path.parent.mkdir(parents=True, exist_ok=True)
        json_handler = logging.FileHandler(json_log_path, encoding="utf-8")
        json_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(json_handler)
