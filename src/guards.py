"""Data-validation guards that enforce non-negotiable conditions on every run.

Right now there is one, and it exists because a person gave permission with strings
attached: **Pictor-PPE is evaluation-only** (supervisor condition, 2026-07-26). It may be
scored against, never trained or fine-tuned on. That is easy to honour today and easy to
break in three months, when a config is copied, a path is edited and nobody remembers why
the third dataset was different. So the condition is enforced in code rather than in memory:
every training entry point calls :func:`assert_training_data_excludes_pictor` before a run
starts, and a violation raises instead of training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Substrings that identify the evaluation-only set even if the configured root moves.
PICTOR_MARKERS = ("pictor",)


def _paths_in(value: Any) -> list[str]:
    """Every string that could name a path, from an arbitrarily nested config value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _paths_in(v)]
    if isinstance(value, (list, tuple, set)):
        return [s for v in value for s in _paths_in(v)]
    if isinstance(value, Path):
        return [str(value)]
    return []


def assert_training_data_excludes_pictor(config: dict, pictor_root: Path | None = None) -> None:
    """Refuse to start a training run whose data configuration can reach Pictor-PPE.

    Checks the ``data`` key (the dataset YAML a run trains on) and, defensively, every
    other string in the config, since a copied config can smuggle the path in through
    ``path``, ``train`` or a custom key.

    One value is exempt: ``datasets.pictor_root`` itself. The root is *declared* on purpose so
    the audit and evaluation paths can resolve it, and a config is not a training run; what
    matters is that no training input ever points there. Declaring it stays legal; reaching it
    from anywhere else does not.

    Args:
        config: The resolved run configuration.
        pictor_root: Configured Pictor root, if any. Matching is by normalised substring,
            so a moved or renamed copy is still caught by :data:`PICTOR_MARKERS`.

    Raises:
        ValueError: If any configured path refers to the evaluation-only dataset.
    """
    markers = list(PICTOR_MARKERS)
    if pictor_root is not None:
        markers.append(Path(pictor_root).name.lower())

    declared = config.get("datasets", {})
    scanned = {key: value for key, value in config.items() if key != "datasets"}
    scanned["datasets"] = {k: v for k, v in declared.items() if k != "pictor_root"}

    for candidate in _paths_in(scanned):
        normalised = candidate.replace("\\", "/").lower()
        for marker in markers:
            if marker and marker in normalised:
                raise ValueError(
                    "Pictor-PPE is EVALUATION-ONLY (supervisor condition, 2026-07-26): it "
                    "must never be trained or fine-tuned on. Offending config value: "
                    f"{candidate!r}. Remove it, or score the model on Pictor through the "
                    "evaluation path instead."
                )
    logger.debug("training-data guard passed: no Pictor reference in the run config")
