"""Structured logging for the PPE-compliance project.

Every script and notebook uses :func:`get_logger` instead of ``print`` so runs emit a
consistent, timestamped, level-tagged stream (engineering standard: structured logging,
never ``print``). One console handler is attached per logger name; repeated calls return
the same configured logger without stacking duplicate handlers.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _force_utf8_console() -> None:
    """Make stdout/stderr UTF-8 so non-ASCII output cannot kill a run.

    The Windows console defaults to cp1252, which cannot encode the characters this project
    legitimately uses in reports and log lines (``±`` in "mean ± 95 % CI", em dashes, ``⚠``).
    Without this, a *completed* pass dies on its final ``print`` with ``UnicodeEncodeError``
    and returns a non-zero exit code — which is how `src/eda.py` came to exit 1 after writing
    every one of its artefacts successfully. Failing on the report, after the work is done, is
    the worst place to fail: it makes success look like failure to any calling script.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # already detached or not a real stream
                pass


_force_utf8_console()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a console logger configured with the project format.

    Args:
        name: Logger name, conventionally the module's ``__name__``.
        level: Minimum level to emit (default :data:`logging.INFO`).

    Returns:
        A :class:`logging.Logger` with exactly one stream handler and propagation
        disabled, so messages are not duplicated by ancestor loggers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    return logger
