"""Environment sanity check: run BEFORE any training.

Confirms a CUDA-enabled PyTorch sees the RTX 4070 Laptop GPU, reports the exact versions
to freeze into the vault (00-coding-context .2 + 00-Key-Facts), and asserts the dataset
roots exist. Dataset roots are read from ``configs/base.yaml`` (never hardcoded here).

Exit codes:
    0 -- full success (CUDA available, GPU visible, both dataset roots present).
    1 -- CUDA unavailable, a required import is missing, or a dataset root is absent.

Usage::

    python scripts/verify_env.py
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import get_logger

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _REPO_ROOT / "configs" / "base.yaml"
_EXPECTED_GPU = "RTX 4070 Laptop"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"

logger = get_logger("verify_env")


def _dataset_roots(config_path: Path) -> dict[str, Path]:
    """Read the SH17 and CHV dataset roots from the base config.

    Args:
        config_path: Path to ``configs/base.yaml``.

    Returns:
        Mapping of dataset name -> resolved root path.

    Raises:
        FileNotFoundError: If the config file is missing.
        KeyError: If the ``datasets`` block or a required root key is absent.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    datasets = config.get("datasets")
    if not datasets:
        raise KeyError(f"'datasets' block missing from {config_path}")
    return {
        "SH17": Path(datasets["sh17_root"]),
        "CHV": Path(datasets["chv_root"]),
    }


def main() -> int:
    """Run all environment checks and return a process exit code."""
    ok = True

    logger.info("Python           : %s", platform.python_version())

    try:
        import torch
    except ImportError:
        logger.error("torch not installed. Install the CUDA build first (README .1):")
        logger.error("  pip install torch torchvision --index-url %s", TORCH_INDEX_URL)
        return 1

    logger.info("torch            : %s", torch.__version__)
    logger.info("torch CUDA build : %s", torch.version.cuda)

    cuda_available = torch.cuda.is_available()
    logger.info("CUDA available   : %s", cuda_available)
    if not cuda_available:
        logger.error("CUDA not available -- a CPU-only torch was installed.")
        logger.error("Uninstall torch/torchvision and reinstall from the cu124 index.")
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    logger.info("GPU              : %s", gpu_name)
    if _EXPECTED_GPU not in gpu_name:
        logger.warning("Expected a '%s' GPU; got '%s'.", _EXPECTED_GPU, gpu_name)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info("VRAM             : %.1f GB", vram_gb)

    try:
        import ultralytics

        logger.info("ultralytics      : %s", ultralytics.__version__)
    except ImportError:
        logger.error("ultralytics not installed -- run: pip install -r requirements.txt")
        ok = False

    try:
        for name, root in _dataset_roots(_CONFIG_PATH).items():
            if root.is_dir():
                logger.info("dataset %-8s : %s (found)", name, root)
            else:
                logger.error("dataset %-8s : %s (MISSING)", name, root)
                ok = False
    except (FileNotFoundError, KeyError) as exc:
        logger.error("Could not resolve dataset roots: %s", exc)
        ok = False

    if ok:
        logger.info("Environment ready. Freeze these versions into the vault.")
    else:
        logger.error("Environment NOT ready. Fix the errors above, then re-run.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
