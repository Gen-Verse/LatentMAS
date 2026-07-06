"""Benchmark-suite status: per-run progress, completed modes, and anomaly scan.

Usage: python scripts/bench_suite/status.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

RUNS = ["hom_mgsm", "hom_belebele_sg", "het_mgsm", "het_belebele_sg"]
LOGDIR = Path("logs/bench_suite")
CKPT_ROOT = Path(".cache/checkpoints/bench_suite")

ANOMALY_PATTERNS = re.compile(
    r"Traceback|CUDA out of memory|RuntimeError|ValueError|LatentDrift"
    r"|GIVING UP|nan|accuracy=1\.000|accuracy=0\.000",
)


def main() -> int:
    for run in RUNS:
        print(f"\n=== {run} ===")
        log = LOGDIR / f"{run}.log"
        if not log.exists():
            print("  log: not started")
            continue
        lines = log.read_text(errors="replace").splitlines()
        # Progress: last stage/mode markers
        markers = [l for l in lines if re.search(
            r"Running Stage [A-G]|Evaluating Mode|Mode '.*' (complete|loaded from cache)", l)]
        for m in markers[-4:]:
            print("  " + m[-160:])
        # Completed modes from the Stage-E result cache
        cache_dir = CKPT_ROOT / run / "coordination"
        if cache_dir.exists():
            cached = sorted(p.name for p in cache_dir.glob("result_*mode*"))
            print(f"  cached mode results: {len(cached)}")
        # Anomalies (dedup, last 8)
        anomalies = [l for l in lines if ANOMALY_PATTERNS.search(l)]
        uniq = list(dict.fromkeys(a.strip()[-160:] for a in anomalies))
        if uniq:
            print(f"  ANOMALIES ({len(anomalies)} lines, {len(uniq)} distinct, last 8):")
            for a in uniq[-8:]:
                print("    ! " + a)
        else:
            print("  anomalies: none")
        print(f"  log lines: {len(lines)}; last: {lines[-1][-160:] if lines else ''}")

    for drv in sorted(LOGDIR.glob("*.driver.log")):
        print(f"\n--- {drv.name} ---")
        for l in drv.read_text().splitlines()[-6:]:
            print("  " + l)

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        print("\nGPUs (idx, mem, util):")
        print("  " + out.replace("\n", "\n  "))
    except Exception as exc:  # noqa: BLE001
        print(f"nvidia-smi unavailable: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
