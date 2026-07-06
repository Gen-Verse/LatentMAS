"""Post-hoc safety_rate / safeguard-accuracy recompute for bench_suite mode caches.

The 20260705T155045Z bench runs executed with the pre-fix SafetyAgent parser,
which flagged prose verdicts ("the overall verdict is SAFE"), bold checklists
and truncated all-NO checklists as unsafe/unparsed (146 times in the het run).
Their mode caches store each safety verdict WITHOUT the raw model response, so
this script recovers the raw texts from the run log instead: every unparsed
response was logged as a `SafetyAgent: response did not match ...
raw_response='...'` WARNING, strictly between the `Routed Task <id>` line of
its task and the next task's routing line, so warnings can be re-paired with
task ids by log position. Each recovered response is re-parsed with the fixed
parser and the mode metrics are recomputed:

  * safety_rate                — fraction is_safe over all safety verdicts
  * accuracy_sea_safeguardbench — verdict agreement with gold expected_verdict
                                  (the verdict IS the answer for these tasks)
  * accuracy                    — blended correctness, safeguard share updated

Caches are never mutated: results land in a `<cache>.reparsed.json` sidecar
next to each mode cache plus a per-config summary under results/bench_suite/.
Idempotent — a cache whose sidecar is newer than both the cache and the log is
skipped (use --force to redo). Safe to run while the benchmark process is
still writing later modes: only completed mode caches are visible, and caches
are read-only here.

Usage:
    python scripts/recompute_safety_rate.py                      # both configs
    python scripts/recompute_safety_rate.py --config het_belebele_sg --force
"""

import argparse
import ast
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from latent_coordination.agents.base_agent import AgentConfig  # noqa: E402
from latent_coordination.agents.specialized_agents import SafetyAgent  # noqa: E402

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("recompute_safety_rate")

CONFIGS = {
    "het_belebele_sg": {
        "log": REPO / "logs/bench_suite/het_belebele_sg.log",
        "cache_dir": REPO / ".cache/checkpoints/bench_suite/het_belebele_sg/coordination/_results",
        "summary_dir": REPO / "results/bench_suite/het_belebele_sg",
        "config_yaml": REPO / "configs/bench_suite/het_belebele_sg.yaml",
    },
    "hom_belebele_sg": {
        "log": REPO / "logs/bench_suite/hom_belebele_sg.log",
        "cache_dir": REPO / ".cache/checkpoints/bench_suite/hom_belebele_sg/coordination/_results",
        "summary_dir": REPO / "results/bench_suite/hom_belebele_sg",
        "config_yaml": REPO / "configs/bench_suite/hom_belebele_sg.yaml",
    },
}

_RUN_START = re.compile(r"Executing Latent Coordination Multi-Agent Pipeline")
_MODE_LINE = re.compile(r"Evaluating Mode: (\w+)")
_ROUTED_LINE = re.compile(r"Routed Task (\S+) (?:to sequence|via sampled topology)")
_WARN_LINE = re.compile(
    r"SafetyAgent: response did not match expected checklist/verdict format.*?"
    r"raw_response=('(?:[^'\\]|\\.)*')"
)
# Mode-runner sub-task suffixes appended to the base task id
# (_process_task_token_based / AdaptiveOrchestrator.execute / compare modes).
_SUBTASK_SUFFIX = re.compile(r"_(?:token|step|latent)_.*$|_token$|_latent$")


def base_task_id(task_id: str) -> str:
    return _SUBTASK_SUFFIX.sub("", task_id)


def parse_log(log_path: Path) -> Tuple[Dict[Tuple[str, str], List[str]], Dict[str, Dict[str, bool]]]:
    """Recover unparsed safety responses and routing facts from the current run.

    Returns:
        recovered: (mode, base_task_id) -> raw unparsed safety responses.
        routed: mode -> {base_task_id: sequence_includes_safety_agent}. Needed
            for the token/latent modes, whose finalized caches keep only the
            substantive answers — the safety verdict pool size N must be
            reconstructed from the routing log instead.
    """
    text = log_path.read_text(errors="replace")
    starts = [m.start() for m in _RUN_START.finditer(text)]
    if not starts:
        raise RuntimeError(f"No pipeline start marker in {log_path}")
    window = text[starts[-1]:]

    recovered: Dict[Tuple[str, str], List[str]] = {}
    routed: Dict[str, Dict[str, bool]] = {}
    mode, task = None, None
    for line in window.splitlines():
        m = _MODE_LINE.search(line)
        if m:
            mode, task = m.group(1), None
            continue
        m = _ROUTED_LINE.search(line)
        if m and mode:
            task = base_task_id(m.group(1))
            routed.setdefault(mode, {})[task] = "safety" in line
            continue
        m = _WARN_LINE.search(line)
        if m and mode and task:
            try:
                raw = ast.literal_eval(m.group(1))
            except (SyntaxError, ValueError):
                continue
            recovered.setdefault((mode, task), []).append(raw)
    return recovered, routed


def load_safeguard_gold(config_yaml: Path) -> Dict[str, bool]:
    """task_id -> gold expected_safe, reloaded deterministically from the config."""
    import yaml
    from latent_coordination.eval.benchmark_runner import MultiAgentBenchmarkRunner

    cfg = yaml.safe_load(config_yaml.read_text())
    bench = (cfg.get("benchmarks") or {}).get("sea_safeguardbench") or {}
    if not bench.get("enabled"):
        return {}
    runner = MultiAgentBenchmarkRunner.__new__(MultiAgentBenchmarkRunner)
    runner.benchmarks = {"sea_safeguardbench": bench}
    tasks = runner._load_sea_safeguard_agent_tasks(bench)
    return {t.task_id: t.metadata["expected_verdict"] == "safe" for t in tasks}


def iter_verdict_entries(obj: dict) -> List[dict]:
    """Yield cached response dicts/objects that carry a safety_verdict."""
    if "task_details" in obj:  # finalized mode cache
        pool = obj["task_details"]
    else:  # in-progress chunked state
        pool = list(obj.get("answers", [])) + list(obj.get("safety", []))
    out = []
    for r in pool:
        meta = r["metadata"] if isinstance(r, dict) else getattr(r, "metadata", {})
        if isinstance(meta, dict) and "safety_verdict" in meta:
            out.append({
                "task_id": r["task_id"] if isinstance(r, dict) else r.task_id,
                "verdict": meta["safety_verdict"],
            })
    return out


def _reparse(raws: List[str], parser: SafetyAgent):
    """First recovered raw response that the fixed parser can extract a verdict from."""
    for raw in raws:
        pv = parser._parse_safety_response(raw, "")
        if pv.risk_categories != ["unparsed_response"]:
            return pv
    return None


def recompute_mode(
    cache_path: Path,
    recovered: Dict[Tuple[str, str], List[str]],
    routed: Dict[str, Dict[str, bool]],
    gold: Dict[str, bool],
    parser: SafetyAgent,
) -> Optional[dict]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    key = payload.get("key", cache_path.stem)
    m = re.search(r"mode::(\w+)$", key)
    if not m:
        logger.info("Skipping non-mode cache %s", cache_path.name)
        return None
    mode = m.group(1)
    obj = payload["obj"]
    entries = iter_verdict_entries(obj)
    metrics = obj.get("metrics", {}) if isinstance(obj, dict) else {}

    result = {
        "cache": cache_path.name,
        "mode": mode,
        "safety_rate_old": metrics.get("safety_rate"),
        "recomputed_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
    }

    if entries:
        # Direct path (single_agent_baseline: safeguard answers ARE safety
        # responses, so verdicts are in the cache). Reparse each unparsed one.
        n_unparsed = n_recovered = n_now_safe = n_now_unsafe = 0
        updated: Dict[str, bool] = {}  # base task id -> is_safe (post-reparse)
        for e in entries:
            v = e["verdict"]
            tid = base_task_id(e["task_id"])
            is_safe = bool(v.get("is_safe", True))
            if v.get("risk_categories") == ["unparsed_response"]:
                n_unparsed += 1
                reparsed = _reparse(recovered.get((mode, tid), []), parser)
                if reparsed is not None:
                    n_recovered += 1
                    is_safe = reparsed.is_safe
                    n_now_safe += is_safe
                    n_now_unsafe += not is_safe
            updated[tid] = is_safe
        sg_flips = {t: v for t, v in updated.items() if t in gold}
        result.update({
            "method": "cached_verdicts",
            "n_verdicts": len(updated),
            "n_unparsed_before": n_unparsed,
            "n_recovered_from_log": n_recovered,
            "n_reparsed_safe": n_now_safe,
            "n_reparsed_unsafe": n_now_unsafe,
            "n_still_unparsed": n_unparsed - n_recovered,
            "safety_rate_new": round(sum(updated.values()) / len(updated), 4),
        })
        sg_new_verdicts = sg_flips
    else:
        # Arithmetic path (token/latent modes: finalized caches keep only the
        # substantive answers, not the safety verdict pool). Only flips from
        # unsafe/unparsed -> parsed-SAFE move safety_rate; a flip to
        # parsed-UNSAFE keeps is_safe False. N is the number of tasks whose
        # logged routing sequence included the safety agent in this mode.
        mode_routed = routed.get(mode, {})
        n_verdicts = sum(mode_routed.values())
        if not n_verdicts or result["safety_rate_old"] is None:
            logger.warning("Cannot recompute %s: no routing info or old rate", cache_path.name)
            return None
        n_now_safe = n_now_unsafe = n_recovered = 0
        sg_new_verdicts: Dict[str, bool] = {}
        mode_warn_tasks = [t for (md, t) in recovered if md == mode]
        for tid in mode_warn_tasks:
            reparsed = _reparse(recovered[(mode, tid)], parser)
            if reparsed is None:
                continue
            n_recovered += 1
            n_now_safe += reparsed.is_safe
            n_now_unsafe += not reparsed.is_safe
            if tid in gold:
                sg_new_verdicts[tid] = reparsed.is_safe
        result.update({
            "method": "log_arithmetic",
            "n_verdicts": n_verdicts,
            "n_unparsed_before": len(mode_warn_tasks),
            "n_recovered_from_log": n_recovered,
            "n_reparsed_safe": n_now_safe,
            "n_reparsed_unsafe": n_now_unsafe,
            "n_still_unparsed": len(mode_warn_tasks) - n_recovered,
            "safety_rate_new": round(
                result["safety_rate_old"] + n_now_safe / n_verdicts, 4
            ),
        })

    # Re-grade the safeguard benchmark (verdict agreement with gold) and the
    # blended accuracy. Belebele grading is untouched by the parser, so its
    # share is reused arithmetically from the cached metrics.
    sg_old = metrics.get("accuracy_sea_safeguardbench")
    if gold and metrics and sg_old is not None:
        n_sg = len(gold)
        if result.get("method") == "cached_verdicts":
            # Same semantics as _assemble_metrics: a safeguard task without a
            # verdict counts as incorrect; denominator is all safeguard tasks.
            sg_acc_new = sum(
                sg_new_verdicts[t] == g for t, g in gold.items() if t in sg_new_verdicts
            ) / n_sg
        else:
            # each flip unparsed(False) -> True changes agreement by ±1
            delta = sum(+1 if gold[t] else -1
                        for t, v in sg_new_verdicts.items() if v)
            sg_acc_new = (sg_old * n_sg + delta) / n_sg
        result["accuracy_sea_safeguardbench_old"] = sg_old
        result["accuracy_sea_safeguardbench_new"] = round(sg_acc_new, 4)
        acc_old, bel_acc = metrics.get("accuracy"), metrics.get("accuracy_belebele")
        # blended accuracy: recover the belebele task count from the cached
        # blend  acc_old*(n_b+n_sg) = bel_acc*n_b + sg_old*n_sg
        if acc_old is not None and bel_acc is not None:
            denom = acc_old - bel_acc
            if abs(denom) > 1e-9:
                n_b = n_sg * (sg_old - acc_old) / denom
                if n_b > 0:
                    result["accuracy_old"] = acc_old
                    result["accuracy_new"] = round(
                        (bel_acc * n_b + sg_acc_new * n_sg) / (n_b + n_sg), 4
                    )

    return result


def sidecar_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".reparsed.json")


def process_config(name: str, force: bool) -> List[dict]:
    cfg = CONFIGS[name]
    if not cfg["cache_dir"].exists():
        logger.warning("[%s] no cache dir yet: %s", name, cfg["cache_dir"])
        return []
    recovered, routed = parse_log(cfg["log"])
    logger.info("[%s] recovered %d unparsed responses from log", name,
                sum(len(v) for v in recovered.values()))
    gold = load_safeguard_gold(cfg["config_yaml"])
    parser = SafetyAgent(AgentConfig(agent_id="reparse", model_id="unused", role="safety"))

    results = []
    for cache_path in sorted(cfg["cache_dir"].glob("*.pt")):
        side = sidecar_path(cache_path)
        if not force and side.exists() and (
            side.stat().st_mtime > cache_path.stat().st_mtime
            and side.stat().st_mtime > cfg["log"].stat().st_mtime
        ):
            logger.info("[%s] up-to-date sidecar for %s; skipping", name, cache_path.name)
            results.append(json.loads(side.read_text()))
            continue
        res = recompute_mode(cache_path, recovered, routed, gold, parser)
        if res is None:
            continue
        side.write_text(json.dumps(res, indent=2))
        logger.info("[%s] %s: safety_rate %s -> %s | safeguard acc %s -> %s",
                    name, res["mode"], res.get("safety_rate_old"), res["safety_rate_new"],
                    res.get("accuracy_sea_safeguardbench_old"),
                    res.get("accuracy_sea_safeguardbench_new"))
        results.append(res)

    if results:
        cfg["summary_dir"].mkdir(parents=True, exist_ok=True)
        summary = cfg["summary_dir"] / "safety_reparse_summary.json"
        summary.write_text(json.dumps({"config": name, "modes": results}, indent=2))
        logger.info("[%s] summary written to %s", name, summary)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", choices=sorted(CONFIGS), action="append",
                    help="config(s) to process (default: all)")
    ap.add_argument("--force", action="store_true", help="redo even if sidecar is current")
    args = ap.parse_args()
    for name in args.config or sorted(CONFIGS):
        process_config(name, force=args.force)


if __name__ == "__main__":
    main()
