"""Structured logging for the PPE-compliance project.

Every script and notebook uses :func:`get_logger` instead of ``print`` so runs emit a
consistent, timestamped, level-tagged stream (engineering standard: structured logging,
never ``print``). One console handler is attached per logger name; repeated calls return
the same configured logger without stacking duplicate handlers.
"""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


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
