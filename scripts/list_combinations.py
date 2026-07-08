"""List/filter valid (model x baseline x benchmark x language x metric) combinations.

Usage:
    python scripts/list_combinations.py
    python scripts/list_combinations.py --benchmark belebele --language th
    python scripts/list_combinations.py --baseline LatentMASBaseline --format json
    python scripts/list_combinations.py --include-recommended
    python scripts/list_combinations.py --check Qwen/Qwen2.5-7B-Instruct LatentMASBaseline mgsm th exact_match
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from shared.combinations import (  # noqa: E402
    BASELINES, BENCHMARKS, MODELS, METRICS,
    enumerate_valid_combinations, validate_combination,
)

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", action="append", default=None, help="Filter to this model (repeatable).")
    parser.add_argument("--baseline", action="append", default=None, help="Filter to this baseline (repeatable).")
    parser.add_argument("--benchmark", action="append", default=None, help="Filter to this benchmark (repeatable).")
    parser.add_argument("--language", action="append", default=None, help="Filter to this language (repeatable).")
    parser.add_argument("--metric", action="append", default=None, help="Filter to this metric (repeatable).")
    parser.add_argument(
        "--include-recommended", action="store_true", default=False,
        help="Also include 'recommended' (not-yet-configured) models/metrics, not just what's runnable today.",
    )
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table")
    parser.add_argument(
        "--check", nargs=5, metavar=("MODEL", "BASELINE", "BENCHMARK", "LANGUAGE", "METRIC"),
        default=None, help="Validate one specific combination instead of enumerating. Use 'None' for LANGUAGE.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.check:
        model, baseline, benchmark, language, metric = args.check
        language = None if language.lower() == "none" else language
        ok, reason = validate_combination(model, baseline, benchmark, language, metric)
        print(f"{'VALID' if ok else 'INVALID'}: {reason}")
        return 0 if ok else 1

    combos = enumerate_valid_combinations(
        models=args.model, baselines=args.baseline, benchmarks=args.benchmark,
        languages=args.language, metrics=args.metric,
        include_recommended=args.include_recommended,
    )

    if args.format == "json":
        print(json.dumps([c.__dict__ for c in combos], indent=2))
    elif args.format == "csv":
        print("model,baseline,benchmark,language,metric")
        for c in combos:
            print(f"{c.model},{c.baseline},{c.benchmark},{c.language or ''},{c.metric}")
    else:
        print(f"{len(combos)} valid combinations "
              f"(models={len(MODELS)}, baselines={len(BASELINES)}, "
              f"benchmarks={len(BENCHMARKS)}, metrics={len(METRICS)} registered)")
        print(f"{'MODEL':<40} {'BASELINE':<22} {'BENCHMARK':<28} {'LANG':<6} {'METRIC'}")
        for c in combos:
            print(f"{c.model:<40} {c.baseline:<22} {c.benchmark:<28} {(c.language or '-'):<6} {c.metric}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
