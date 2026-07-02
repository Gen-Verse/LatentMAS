"""Post-hoc text metrics for a saved FLORES+ MultiAgentBenchmarkReport JSON.

MultiAgentBenchmarkRunner.run_eval() (src/latent_coordination/eval/benchmark_runner.py)
already computes accuracy/latency/token_cost/safety_rate inline, plus chrf/xcomet/cometkiwi
if enabled via configs/*.yaml's benchmarks.flores_plus.translation_metrics. This script adds
the remaining *text-only* metrics from dev_doc.md's metric list -- bleu, sfr_ifl (Script
Fidelity Rate / Involuntary Fidelity Loss), language_consistency -- as a separate pass over
an already-saved report, so they don't cost any extra GPU time.

Deliberately NOT computed here: adversarial_drift and information_theoretic. Both operate
on latent hidden-state tensors (UniversalLatentHub encode/decode, HSIC over hidden states),
not on generated text -- and MultiAgentBenchmarkReport.to_dict() strips AgentResponse's
latent_state tensor when making the report JSON-safe (see benchmark_runner.py), so the
tensors this would need simply aren't in the saved file. They'd have to be computed inline
during the run itself, not post-hoc from a report.

Usage:
    python scripts/compute_posthoc_metrics.py results/coordination/multiagent_benchmark_*.json
    python scripts/compute_posthoc_metrics.py <report.json> --output <report_posthoc.json>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Same 6 FLORES+ pairs hardcoded in MultiAgentBenchmarkRunner._load_real_tasks -- the
# only real task source this pipeline evaluates against.
FLORES_LANG_PAIRS = {
    "th": "tha_Thai", "my": "mya_Mymr", "km": "khm_Khmr",
    "lo": "lao_Laoo", "am": "amh_Ethi", "sw": "swh_Latn",
}
TASK_ID_RE = re.compile(r"^flores_plus_([a-z]+)_(\d+)")


def _load_flores_references(languages: List[str], cache_dir: Optional[str] = None) -> Dict[Tuple[str, int], Tuple[str, str]]:
    """Return {(iso_code, index): (source_en_text, reference_target_text)}."""
    from datasets import load_dataset

    refs: Dict[Tuple[str, int], Tuple[str, str]] = {}
    en_ds = load_dataset("openlanguagedata/flores_plus", name="eng_Latn", split="devtest", cache_dir=cache_dir)
    for iso in languages:
        flores_code = FLORES_LANG_PAIRS.get(iso)
        if flores_code is None:
            logger.warning("Skipping unknown FLORES+ language '%s' (not in the hardcoded pipeline set).", iso)
            continue
        tgt_ds = load_dataset("openlanguagedata/flores_plus", name=flores_code, split="devtest", cache_dir=cache_dir)
        n = min(len(en_ds), len(tgt_ds))
        for i in range(n):
            refs[(iso, i)] = (en_ds[i]["text"], tgt_ds[i]["text"])
    return refs


def _match_mode_responses(
    task_details: List[Dict], refs: Dict[Tuple[str, int], Tuple[str, str]],
) -> Tuple[List[str], List[str], List[str]]:
    """Return (predictions, references, target_languages) for responses whose task_id
    is traceable back to a FLORES+ (iso_code, index) pair. Silently skips (with a
    logged count) responses whose task_id doesn't match -- e.g. safety-agent verdicts
    or any non-FLORES+ task source."""
    predictions, references, langs = [], [], []
    n_skipped = 0
    for resp in task_details:
        m = TASK_ID_RE.match(resp.get("task_id", ""))
        if not m:
            n_skipped += 1
            continue
        iso, idx = m.group(1), int(m.group(2))
        ref_pair = refs.get((iso, idx))
        if ref_pair is None:
            n_skipped += 1
            continue
        predictions.append(resp.get("output_text", "") or "")
        references.append(ref_pair[1])
        langs.append(iso)
    if n_skipped:
        logger.info("Skipped %d response(s) with no traceable FLORES+ reference.", n_skipped)
    return predictions, references, langs


def compute_posthoc_metrics_for_mode(
    predictions: List[str], references: List[str], target_languages: List[str],
) -> Dict:
    from shared.metrics import compute_bleu
    from latent_coordination.eval.script_fidelity import ScriptFidelityEvaluator, LanguageConsistencyEvaluator

    if not predictions:
        return {}

    bleu = compute_bleu(predictions, references)

    sfr_report = ScriptFidelityEvaluator().evaluate_generated(predictions, target_languages)
    lc_report = LanguageConsistencyEvaluator().evaluate_batch(predictions, target_languages)

    return {
        "bleu": bleu,
        "sfr": sfr_report.mean_sfr,
        "ifl": 1.0 - sfr_report.mean_sfr,
        "sfr_per_language": sfr_report.per_language,
        "language_consistency": lc_report.consistency_rate,
        "language_consistency_per_language": lc_report.per_language,
        "language_consistency_n_unscorable": lc_report.n_unscorable,
        "n_scored": len(predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", help="Path to a multiagent_benchmark_*.json report.")
    parser.add_argument("--output", default=None, help="Output path (default: <report>_posthoc.json).")
    parser.add_argument("--cache-dir", default=None, help="HF datasets cache dir.")
    args = parser.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    task_details = report.get("task_details", {})
    if not task_details:
        raise ValueError(f"{report_path} has no task_details -- nothing to score.")

    # Discover which languages are actually present before hitting the network.
    languages = sorted({
        m.group(1) for mode_details in task_details.values()
        for resp in mode_details
        if (m := TASK_ID_RE.match(resp.get("task_id", "")))
    })
    if not languages:
        raise ValueError(
            f"{report_path}: no response task_id matched the flores_plus_<lang>_<idx> "
            "pattern -- this report doesn't look like a FLORES+ run."
        )
    logger.info("Languages found in report: %s", languages)
    refs = _load_flores_references(languages, cache_dir=args.cache_dir)

    for mode, mode_details in task_details.items():
        predictions, references, langs = _match_mode_responses(mode_details, refs)
        metrics = compute_posthoc_metrics_for_mode(predictions, references, langs)
        if not metrics:
            logger.warning("Mode '%s': no scorable responses, skipping.", mode)
            continue
        report.setdefault("results_by_mode", {}).setdefault(mode, {}).update(metrics)
        logger.info(
            "Mode '%-24s' | n=%3d | bleu=%.2f sfr=%.3f ifl=%.3f lc=%.3f",
            mode, metrics["n_scored"], metrics["bleu"], metrics["sfr"], metrics["ifl"],
            metrics["language_consistency"],
        )

    out_path = Path(args.output) if args.output else report_path.with_name(report_path.stem + "_posthoc.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
