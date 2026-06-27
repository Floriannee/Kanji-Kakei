"""Logging configuration.

Call ``setup_logging`` once at the start of any entry-point script. Every
module elsewhere just does ``logger = logging.getLogger(__name__)`` and the
config here decides where it goes. No print statements anywhere in the
codebase.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from config import settings

_CONFIGURED = False


def setup_logging(
    level: str | int | None = None,
    log_file: str | Path | None = None,
) -> None:
    """Configure root logging once. Idempotent.

    Logs go to stderr and, by default, to outputs/logs/pipeline.log.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = level or settings.log_level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    log_file = Path(log_file) if log_file else Path(settings.outputs_dir) / "logs" / "pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    handlers.append(file_handler)

    formatter = logging.Formatter(fmt, datefmt)
    for h in handlers:
        h.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if some library already configured root.
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)

    _CONFIGURED = True
    logging.getLogger(__name__).debug("Logging configured at level %s", level)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper that ensures logging is set up."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
