"""
Shared logging utilities: coloured console handler, JSON file handler,
and a top-level ``setup_logging`` factory.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class ColoredConsoleFormatter(logging.Formatter):
    """ANSI-coloured console formatter for human-readable output."""

    _COLORS = {
        logging.DEBUG: "\033[36m",    # cyan
        logging.INFO: "\033[32m",     # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",    # red
        logging.CRITICAL: "\033[35m", # magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{self._RESET}"
        return super().format(record)


class JsonFileHandler(logging.FileHandler):
    """Writes one JSON object per log line for structured log ingestion."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
                "level": record.levelname,
                "logger": record.name,
                "msg": self.format(record),
            }
            if record.exc_info:
                entry["exc"] = self.formatException(record.exc_info)
            self.stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.flush()
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Resource usage helper
# ---------------------------------------------------------------------------

def log_resource_usage(logger: logging.Logger) -> None:
    """Log current process memory usage (RSS) if psutil is available."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem_mb = proc.memory_info().rss / 1024 ** 2
        logger.info("Resource usage | RSS=%.1f MB", mem_mb)
    except ImportError:
        logger.debug("psutil not installed; skipping resource usage log.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def setup_logging(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    json_format: bool = True,
) -> logging.Logger:
    """Configure and return a named logger with console + optional file output.

    Args:
        name: Logger name (used to retrieve the logger via ``logging.getLogger``).
        log_dir: Directory to write log files.  If None, file logging is skipped.
        level: Logging level (e.g. ``logging.INFO``).
        json_format: If True, the file handler writes JSON lines; otherwise plain text.

    Returns:
        The configured ``logging.Logger`` instance.
    """
    log = logging.getLogger(name)
    log.setLevel(level)

    if log.handlers:
        # Avoid duplicate handlers on repeated calls
        return log

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_fmt = "%Y-%m-%dT%H:%M:%S"

    # Console handler (coloured)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(
        ColoredConsoleFormatter(fmt=fmt, datefmt=date_fmt)
    )
    log.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}.log"
        if json_format:
            file_handler = JsonFileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=date_fmt))
        file_handler.setLevel(level)
        log.addHandler(file_handler)

    return log
