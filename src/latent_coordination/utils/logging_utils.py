"""Logging utilities for the Mechanistic Disentanglement package, delegating to the shared layer."""

import logging
from pathlib import Path
from shared.logging_utils import setup_logging as shared_setup_logging, JsonFileHandler, ColoredConsoleFormatter, log_resource_usage

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

def setup_logging(cfg: dict) -> logging.Logger:
    """Configure logging using project configuration dictionary."""
    log_cfg = cfg.get("logging", {})
    log_dir = Path(log_cfg.get("log_dir", "logs/mechanistic"))
    level_str = log_cfg.get("level", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)
    json_format = log_cfg.get("json_logs", True)
    
    return shared_setup_logging(
        name="mechanistic_pipeline",
        log_dir=log_dir,
        level=level,
        json_format=json_format,
    )

