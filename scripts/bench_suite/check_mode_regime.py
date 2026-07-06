#!/usr/bin/env python
"""Print cached mode-result files whose code_regime predates the router fix.

Usage: check_mode_regime.py <_results_dir> [mode ...]

For each requested mode (default: token_based_mas, latent_based_mas_ours),
scans <_results_dir>/*mode__<mode>.pt for CheckpointManager.cache_result
payloads (see src/shared/checkpointing.py) and prints one path per line for
every file whose stamped code_regime.router != "prototype-seeded" (missing
entirely, e.g. pre-3f75cfb runs, also counts as stale). Prints nothing, and
exits 0, if a directory/file can't be read -- this is a read-only probe used
by scripts/bench_suite/requeue_router_fix.sh to decide what to invalidate.
"""
import sys
from pathlib import Path

import torch

FIXED_REGIME = "prototype-seeded"
DEFAULT_MODES = ("token_based_mas", "latent_based_mas_ours")


def main() -> None:
    results_dir = Path(sys.argv[1])
    modes = sys.argv[2:] or DEFAULT_MODES
    if not results_dir.is_dir():
        return
    for mode in modes:
        for path in sorted(results_dir.glob(f"*mode__{mode}.pt")):
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                regime = payload.get("obj", {}).get("code_regime", {})
            except Exception:
                regime = {}
            if regime.get("router") != FIXED_REGIME:
                print(path)


if __name__ == "__main__":
    main()
