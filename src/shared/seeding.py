"""Deterministic seeding shared by all three pipelines.

Configs declare ``project.seed`` (default 42); calling :func:`set_seed` at pipeline start
makes k-means centroid init, CVAE training, SVD, dropout, and dataset shuffling reproducible
across runs and across ``--resume``.
"""

from __future__ import annotations

import logging
import os
import random

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch RNGs.

    Parameters
    ----------
    seed : int
        The seed value.
    deterministic : bool
        If True, also request deterministic cuDNN kernels (slower, fully reproducible).
        Left False by default so throughput is unaffected.

    Returns
    -------
    int
        The seed that was set (for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass

    logger.info("Global seed set to %d (deterministic=%s).", seed, deterministic)
    return seed
