"""Minimal structured logging helper.

We avoid pulling in a full logging library. ``rich`` handles console formatting
when available; otherwise we fall back to the stdlib ``logging`` module with a
sane format string.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger exactly once.

    Reads ``FINQA_LOG_LEVEL`` from the environment if ``level`` is not passed.
    Safe to call multiple times; subsequent calls are no-ops.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = (level or os.getenv("FINQA_LOG_LEVEL") or "INFO").upper()
    numeric = getattr(logging, resolved, logging.INFO)

    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            markup=False,
            show_path=False,
            show_time=True,
            show_level=True,
        )
        fmt = "%(message)s"
    except ImportError:
        handler = logging.StreamHandler(stream=sys.stderr)
        fmt = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"

    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(numeric)
    root.handlers = [handler]

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    _CONFIGURED = True


def add_file_handler(
    log_file: "str | os.PathLike[str]",
    file_level: int = logging.INFO,
    console_level: int = logging.WARNING,
) -> None:
    """Split logging: detailed records go to *log_file*, console shows WARNING+.

    Call this after :func:`configure_logging` (or it will be called implicitly).
    Safe to call multiple times — each call appends a new FileHandler and
    re-raises the console handler's level.
    """
    configure_logging()
    root = logging.getLogger()

    # Quieten every existing console handler so verbose INFO logs stop printing.
    for h in root.handlers:
        if not isinstance(h, logging.FileHandler):
            h.setLevel(max(h.level, console_level))

    # Add a plain-text file handler that captures everything at file_level+.
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-5s %(name)s: %(message)s")
    )
    root.addHandler(fh)
    # Root must be at least as permissive as the file handler.
    root.setLevel(min(root.level, file_level))


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring the root logger lazily."""
    configure_logging()
    return logging.getLogger(name)


def log_kv(logger: logging.Logger, level: int, message: str, **kv: Any) -> None:
    """Log a message with key=value pairs appended, for lightweight structure."""
    if not kv:
        logger.log(level, message)
        return
    parts = " ".join(f"{k}={v}" for k, v in kv.items())
    logger.log(level, f"{message} {parts}")
