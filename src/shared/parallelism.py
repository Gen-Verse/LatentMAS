"""
Device management and parallel execution utilities shared across pipelines.

Provides:
    DeviceManager  : Best-device selection (CUDA > MPS > CPU).
    ParallelRunner : Thread-pool map helper with progress logging.
    batch_generator: Simple iterable chunker.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Generator, Iterable, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

class DeviceManager:
    """Utility class for selecting the best available compute device."""

    @staticmethod
    def get_best_device() -> torch.device:
        """Return the best available device: CUDA > MPS > CPU.

        Returns:
            A ``torch.device`` instance.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(
                "DeviceManager: Using CUDA device '%s'",
                torch.cuda.get_device_name(0),
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("DeviceManager: Using MPS (Apple Silicon) device.")
        else:
            device = torch.device("cpu")
            logger.info("DeviceManager: No GPU available; using CPU.")
        return device

    @staticmethod
    def list_cuda_devices() -> List[str]:
        """Return a list of available CUDA device names."""
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

class ParallelRunner:
    """Thread-pool based parallel execution for CPU-bound pipeline stages.

    Note: For GPU-bound workloads use a single thread to avoid CUDA context
    contention.  This runner is designed for CPU-parallel preprocessing steps
    (SVD fitting, metric computation, etc.).
    """

    @staticmethod
    def run_threads(
        fn: Callable,
        args_list: List[Tuple],
        max_workers: int = 4,
        desc: str = "Running",
    ) -> List[Any]:
        """Run *fn* in a thread pool over *args_list*.

        Args:
            fn: Callable accepting positional args from each tuple in *args_list*.
            args_list: List of argument tuples, one per call to *fn*.
            max_workers: Maximum number of concurrent threads.
            desc: Human-readable description logged at start/end.

        Returns:
            List of return values in the same order as *args_list*.
        """
        n = len(args_list)
        logger.info("%s | spawning %d threads (max_workers=%d)", desc, n, max_workers)

        results: List[Any] = [None] * n
        idx_map = {}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(fn, *args): i
                for i, args in enumerate(args_list)
            }
            completed = 0
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                    completed += 1
                    logger.debug("%s | %d/%d done", desc, completed, n)
                except Exception as exc:
                    logger.error(
                        "%s | task %d failed: %s", desc, i, exc, exc_info=True
                    )
                    raise

        logger.info("%s | all %d tasks completed", desc, n)
        return results


# ---------------------------------------------------------------------------
# Batch generator
# ---------------------------------------------------------------------------

def batch_generator(
    iterable: Iterable,
    batch_size: int,
) -> Generator[List, None, None]:
    """Yield successive fixed-size chunks from *iterable*.

    Args:
        iterable: Any iterable (list, dataset, etc.).
        batch_size: Number of elements per chunk.

    Yields:
        Lists of at most *batch_size* elements.

    Example:
        >>> list(batch_generator(range(10), 3))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    batch: List = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
