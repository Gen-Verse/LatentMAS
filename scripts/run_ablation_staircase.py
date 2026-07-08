"""Staircase ablation CLI (LRL-MRRE-MAS strategy.md §7.3).

Runs the corrected staircase table (rows 0-6 + 7a loss-term split, plus any
``ablation.extra_rows`` from the YAML) as full CoordinationPipeline runs, one
isolated output/checkpoint directory per row, and writes a single consolidated
artifact under ARTIFACTS/ablation_results/.

Usage:
    # Audit what would run (no models loaded) — ALWAYS do this first; a full
    # staircase is multiple GPU-days at paper-scale sample counts.
    python scripts/run_ablation_staircase.py --config configs/latent_coordination.yaml --dry-run

    # Run a subset of rows by id or name
    python scripts/run_ablation_staircase.py --config configs/latent_coordination.yaml --rows 0,1,2

    # Full staircase
    python scripts/run_ablation_staircase.py --config configs/latent_coordination.yaml
"""

import os
# Reduce CUDA fragmentation OOMs. Must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import logging
import sys

import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger("run_ablation_staircase")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Base pipeline YAML config.")
    parser.add_argument("--rows", default=None,
                        help="Comma-separated row ids or names (default: all rows).")
    parser.add_argument("--out", default="ARTIFACTS/ablation_results",
                        help="Root directory for per-row runs + the consolidated artifact.")
    parser.add_argument("--stages", default=None,
                        help="Comma-separated pipeline stage letters per row (default: all A-G).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Derive and dump each row's config without running anything.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    from latent_coordination.eval.ablation_staircase import (
        load_extra_rows,
        run_staircase,
        select_rows,
    )

    rows = select_rows(
        [r.strip() for r in args.rows.split(",")] if args.rows else None,
        load_extra_rows(base_cfg),
    )
    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None

    if not args.dry_run:
        # Standing directive #2 (strategy.md §0): compute scan before GPU work.
        from compute_scan import run_compute_scan
        run_compute_scan()

    consolidated = run_staircase(
        base_cfg, rows, args.out, stages=stages, dry_run=args.dry_run,
    )
    print(f"\nStaircase {'dry-run' if args.dry_run else 'run'} complete: "
          f"{len(rows)} row(s) → {consolidated['artifact_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
