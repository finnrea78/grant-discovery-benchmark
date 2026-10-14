"""Log handler setup for grid runs. The only place handlers are created.

The console gets INFO so a long run stays readable. The run log next to
`grid_run.json` gets DEBUG, so a failed night can be investigated from disk.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path

# The package logger; every `getLogger(__name__)` under `grant_eval` feeds into it.
PACKAGE_LOGGER = "grant_eval"

RUN_LOG_NAME = "grid_run.log"

# Rotate by size and append: a resumed run writes to the same file, so truncating
# would lose the log of the night that halted.
RUN_LOG_MAX_BYTES: int = 32 * 1024 * 1024
RUN_LOG_BACKUPS: int = 9

# Console verbosity, for when DEBUG is wanted on stdout too. `run.py` also exposes
# `--log-level`, which wins over this.
LEVEL_ENV_VAR = "GRANT_EVAL_LOG_LEVEL"

# Same layout as the old print-based lines (UTC stamp, space, message).
_CONSOLE_FORMAT = "%(asctime)s %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

_CONSOLE_TAG = "console"
_RUN_LOG_TAG = "run_log"
_TAG_ATTR = "_grant_eval_tag"


class _UtcFormatter(logging.Formatter):
    """UTC timestamps, so a run that spans a clock change still reads in order."""

    converter = time.gmtime


class _StdoutHandler(logging.StreamHandler):
    """A `StreamHandler` that looks up `sys.stdout` each time it writes.

    This way it follows redirects such as pytest's per-test capture.
    """

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value) -> None:
        # `StreamHandler.__init__` assigns here; the property above is the answer.
        pass


def package_logger() -> logging.Logger:
    return logging.getLogger(PACKAGE_LOGGER)


def _tagged(logger: logging.Logger, tag: str) -> list[logging.Handler]:
    return [h for h in logger.handlers if getattr(h, _TAG_ATTR, None) == tag]


def level_from_env(default: int = logging.INFO) -> int:
    """The console level named by `LEVEL_ENV_VAR`, or `default` if unset or invalid."""
    raw = os.environ.get(LEVEL_ENV_VAR, "").strip().upper()
    if not raw:
        return default
    resolved = logging.getLevelName(raw)
    return resolved if isinstance(resolved, int) else default


def ensure_console(level: int | None = None) -> logging.Handler:
    """Attach the stdout handler to the package logger once, and return it."""
    logger = package_logger()
    # DEBUG at the logger, so the run log can have records the console filters out.
    logger.setLevel(logging.DEBUG)
    # litellm configures the root logger, so propagating would print every line twice.
    logger.propagate = False

    existing = _tagged(logger, _CONSOLE_TAG)
    if existing:
        handler = existing[0]
    else:
        handler = _StdoutHandler()
        setattr(handler, _TAG_ATTR, _CONSOLE_TAG)
        handler.setFormatter(_UtcFormatter(_CONSOLE_FORMAT, _TIME_FORMAT))
        logger.addHandler(handler)
    handler.setLevel(level_from_env() if level is None else level)
    return handler


def add_run_log(results_root, level: int = logging.DEBUG) -> Path:
    """Attach the rotating run log at `<results_root>/grid_run.log` and return its path.

    Any earlier run log handler is removed first, so there is only ever one.
    """
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / RUN_LOG_NAME
    logger = package_logger()

    for handler in _tagged(logger, _RUN_LOG_TAG):
        logger.removeHandler(handler)
        handler.close()

    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=RUN_LOG_MAX_BYTES,
        backupCount=RUN_LOG_BACKUPS,
        encoding="utf-8",
        delay=True,
    )
    setattr(handler, _TAG_ATTR, _RUN_LOG_TAG)
    handler.setLevel(level)
    handler.setFormatter(_UtcFormatter(_FILE_FORMAT, _TIME_FORMAT))
    logger.addHandler(handler)
    return path


def configure(results_root=None, *, console_level: int | None = None) -> Path | None:
    """Set up both handlers and return the run log path, or `None` for a dry run."""
    ensure_console(console_level)
    if results_root is None:
        return None
    return add_run_log(results_root)
