"""Seed everything for reproducible training.

Sets Python, NumPy, and (if installed) PyTorch RNGs, plus the CuDNN
deterministic flags. Import torch lazily so utils stays light for the
inference-only install.
"""

from __future__ import annotations

import os
import random

import numpy as np

from utils.logging_setup import get_logger

logger = get_logger(__name__)


def seed_everything(seed: int) -> None:
    """Make a training run reproducible. Call before building any model."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Seeded python/numpy/torch with %d", seed)
    except ImportError:
        logger.info("Seeded python/numpy with %d (torch not installed)", seed)
