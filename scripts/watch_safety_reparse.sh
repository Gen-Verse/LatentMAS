#!/usr/bin/env bash
# Watches the in-flight 20260705 bench_suite runs and recomputes post-hoc safety
# metrics (scripts/recompute_safety_rate.py) as each mode cache lands. Exits when
# all 3 modes x 2 configs have been reparsed, or when both pipeline processes have
# stopped (after one final pass), or after 6 days.
set -u
cd "$(dirname "$0")/.."
LOG=logs/bench_suite/safety_reparse_watcher.log
deadline=$(( $(date +%s) + 6*24*3600 ))

count_done() {
python - <<'PY'
import json, pathlib
tot = 0
for c in ["het_belebele_sg", "hom_belebele_sg"]:
    p = pathlib.Path(f"results/bench_suite/{c}/safety_reparse_summary.json")
    if p.exists():
        tot += len({m["mode"] for m in json.loads(p.read_text())["modes"]})
print(tot)
PY
}

while [ "$(date +%s)" -lt "$deadline" ]; do
    python scripts/recompute_safety_rate.py >> "$LOG" 2>&1
    n=$(count_done)
    echo "$(date -u +%FT%TZ) watcher: $n/6 mode caches reparsed" >> "$LOG"
    if [ "$n" -ge 6 ]; then
        echo "$(date -u +%FT%TZ) watcher: all modes reparsed; exiting" >> "$LOG"
        exit 0
    fi
    if ! pgrep -f "run_coordination_pipeline.py --config configs/bench_suite/(het|hom)_belebele_sg" >/dev/null; then
        echo "$(date -u +%FT%TZ) watcher: pipelines stopped; final pass" >> "$LOG"
        python scripts/recompute_safety_rate.py >> "$LOG" 2>&1
        echo "$(date -u +%FT%TZ) watcher: exiting after final pass ($(count_done)/6 reparsed)" >> "$LOG"
        exit 0
    fi
    sleep 1800
done
echo "$(date -u +%FT%TZ) watcher: deadline reached" >> "$LOG"
exit 1
