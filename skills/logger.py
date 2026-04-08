"""
skills/logger.py
────────────────
Shared structured logger for the RAG pipeline.
Produces colour-coded console output and a rotating file log.

Usage:
    from skills.logger import get_logger
    log = get_logger(__name__)
    log.info("Step complete", extra={"docs": 12, "chunks": 84})
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path

try:
    import colorlog
    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "rag_pipeline.log"

# ── colour map ────────────────────────────────────────────────────────────
_COLOURS = {
    "DEBUG":    "cyan",
    "INFO":     "green",
    "WARNING":  "yellow",
    "ERROR":    "red",
    "CRITICAL": "bold_red",
}

_FMT_CONSOLE = "%(log_color)s%(levelname)-8s%(reset)s │ %(name)-28s │ %(message)s"
_FMT_FILE    = "%(asctime)s │ %(levelname)-8s │ %(name)-28s │ %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"


def _make_console_handler() -> logging.Handler:
    h = logging.StreamHandler(sys.stdout)
    if _HAS_COLORLOG:
        h.setFormatter(colorlog.ColoredFormatter(
            _FMT_CONSOLE,
            datefmt=_DATE_FMT,
            log_colors=_COLOURS,
        ))
    else:
        h.setFormatter(logging.Formatter(
            "%(levelname)-8s │ %(name)-28s │ %(message)s",
            datefmt=_DATE_FMT,
        ))
    return h


def _make_file_handler() -> logging.Handler:
    h = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,   # 5 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    h.setFormatter(logging.Formatter(_FMT_FILE, datefmt=_DATE_FMT))
    return h


# ── root setup (called once) ──────────────────────────────────────────────
def _configure_root() -> None:
    root = logging.getLogger()
    if root.handlers:
        return                       # already configured
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level, logging.INFO))
    root.addHandler(_make_console_handler())
    root.addHandler(_make_file_handler())

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "langchain"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_configure_root()


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Call once per module at import time."""
    return logging.getLogger(name)


# ── step-progress helper ──────────────────────────────────────────────────
class StepLogger:
    """
    Thin wrapper that logs a labelled progress line at each pipeline step.

    Example
    -------
    sl = StepLogger("ingest")
    sl.start("Loading documents from ./data")
    docs = load(...)
    sl.done(f"Loaded {len(docs)} documents")
    sl.step("Splitting into chunks")
    chunks = split(docs)
    sl.done(f"{len(chunks)} chunks created")
    """

    def __init__(self, stage: str):
        self._log = get_logger(f"pipeline.{stage}")
        self._stage = stage

    def start(self, msg: str, **kw) -> None:
        self._log.info(f"▶  {msg}", extra=kw)

    def step(self, msg: str, **kw) -> None:
        self._log.info(f"   ├─ {msg}", extra=kw)

    def done(self, msg: str, **kw) -> None:
        self._log.info(f"   ✔  {msg}", extra=kw)

    def warn(self, msg: str, **kw) -> None:
        self._log.warning(f"   ⚠  {msg}", extra=kw)

    def error(self, msg: str, **kw) -> None:
        self._log.error(f"   ✖  {msg}", extra=kw)
