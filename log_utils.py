"""
Shared logging setup for VoiceLLM pipeline.
Each run gets its own timestamped file in logs/.
Terminal output is preserved alongside the file.
"""

import logging
import os
import sys
from datetime import datetime

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")


def setup(name: str) -> logging.Logger:
    """
    Create a logger that writes to both stderr and a timestamped log file.
    name: short label used in the filename, e.g. "lm", "tts", "wire"
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"{ts}_{name}.log")

    fmt     = logging.Formatter("%(asctime)s.%(msecs)03d  %(levelname)-5s  %(message)s",
                                datefmt="%H:%M:%S")
    logger  = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # File handler — full DEBUG level
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler — same output, mirrors to stderr
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger
