"""Builds results/experimental_analysis/REPORT.md + plots from live coordination-pipeline
runs. Re-run any time (e.g. after a "Mode complete" log line) to refresh with whatever
per-mode results have landed in each instance's checkpoint cache so far -- this never
waits for a full pipeline run to finish, since MultiAgentBenchmarkRunner.run_eval()
caches each comm-mode's result via CheckpointManager.cache_result() as soon as that mode
finishes, well before the final JSON report is written.

Usage:
    python scripts/build_experimental_report.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "experimental_analysis"
PLOTS_DIR = OUT_DIR / "plots"

# dataviz skill's validated categorical palette (light mode), fixed assignment order --
# never re-cycled per filtered subset.
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"

MODES = ["single_agent_baseline", "token_based_mas", "latent_based_mas_ours"]
MODE_LABELS = {
    "single_agent_baseline": "Baseline (single agent)",
    "token_based_mas": "MAS (token-based)",
    "latent_based_mas_ours": "MAS (latent-based, ours)",
}


@dataclass
class Instance:
    name: str
    output_dir: str
    languages: List[str]
    devices: str


INSTANCES = [
    Instance("th_my_km", "results/coordination_heterogeneous_timeboxed", ["th", "my", "km"], "cuda:0-3"),
    Instance("lo_am_sw", "results/coordination_heterogeneous_timeboxed_2", ["lo", "am", "sw"], "cuda:4-7"),
]


def _load_cached_modes(output_dir: str) -> Dict[str, dict]:
    """Return {mode: metrics_dict} for whichever comm-modes have finished so far."""
    cache_root = REPO_ROOT / output_dir / "checkpoints" / "coordination" / "_results"
    results: Dict[str, dict] = {}
    if not cache_root.exists():
        return results
    for p in cache_root.glob("*.pt"):
        try:
            payload = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        key = payload.get("key", "")
        mode = key.split("::mode::")[-1] if "::mode::" in key else None
        if mode in MODES:
            results[mode] = payload["obj"]["metrics"]
    return results


def gather() -> Dict[str, Dict[str, dict]]:
    return {inst.name: _load_cached_modes(inst.output_dir) for inst in INSTANCES}


def _bar_plot(ax, group_labels: List[str], series: Dict[str, List[Optional[float]]], ylabel: str, title: str):
    n_groups = len(group_labels)
    n_series = len(series)
    width = 0.8 / max(n_series, 1)
    x = range(n_groups)
    for i, (name, values) in enumerate(series.items()):
        offs = [xi + (i - (n_series - 1) / 2) * width for xi in x]
        vals = [v if v is not None else 0 for v in values]
        bars = ax.bar(offs, vals, width=width * 0.9, color=PALETTE[i % len(PALETTE)], label=name)
        for b, v in zip(bars, values):
            if v is not None:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)
    ax.set_xticks(list(x))
    ax.set_xticklabels(group_labels)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY)
    if n_series > 1:
        ax.legend(frameon=False, fontsize=8)


def make_plots(data: Dict[str, Dict[str, dict]]) -> List[str]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    completed_modes = [m for m in MODES if all(m in data[inst.name] for inst in INSTANCES)]
    if not completed_modes:
        return written

    group_labels = [MODE_LABELS[m] for m in completed_modes]

    metric_specs = [
        ("accuracy", "Accuracy (completeness proxy)"),
        ("latency_ms", "Latency (ms/task)"),
        ("chrf", "chrF (translation quality)"),
        ("safety_rate", "Safety pass rate"),
    ]
    for metric_key, title in metric_specs:
        series = {}
        for inst in INSTANCES:
            vals = [data[inst.name].get(m, {}).get(metric_key) for m in completed_modes]
            if any(v is not None for v in vals):
                series[f"{inst.name} ({','.join(inst.languages)})"] = vals
        if not series:
            continue
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        _bar_plot(ax, group_labels, series, metric_key, title)
        fig.tight_layout()
        out_path = PLOTS_DIR / f"{metric_key}_by_mode.png"
        fig.savefig(out_path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        written.append(out_path.name)
    return written


ANOMALIES_SECTION = """## Anomalies found & corrected during this run

1. **Per-agent model_id bug (fixed in code, `coordination_pipeline.py`).** Stage A/C
   originally read a single global `self._agent_model_id` (== the *first* config entry,
   i.e. the orchestrator's model) for every specialized agent, silently collapsing the
   heterogeneous llama/qwen2/cohere pool into 4 copies of the orchestrator's model. Fixed
   to resolve each agent's own `model_id`/`hidden_dim` from its own config entry. Verified
   live: `agent_trans` (Sailor2, qwen2) now registers at `hidden_dim=3584`, distinct from
   `agent_reason`/`agent_safety` (llama/cohere, both 4096).
2. **`--stages` was a no-op (fixed in code, `coordination_pipeline.py::run`).**
   `CoordinationPipeline.run(stages=...)` used to ignore its `stages` argument entirely,
   always executing all 7 internal stages regardless of what was requested. Fixed with an
   `_ensure()` gate that recomputes requested stages and loads unrequested ones from
   checkpoint (or raises if no checkpoint exists).
3. **Routing-driven cost asymmetry in `single_agent_baseline` (config fix).**
   `single_agent_baseline` only executes whichever agent the router selects first. The two
   parallel instances' independently-fitted centroids consistently routed to different
   lead agents (`agent_safety`, `max_new_tokens=128` vs `agent_trans`, `max_new_tokens=256`),
   producing an artificial ~2.2x cost gap between instances that had nothing to do with
   language difficulty. Fixed by using a uniform `max_new_tokens=96` across all three
   specialized agents in `configs/latent_coordination_heterogeneous_timeboxed.yaml`.
4. **Cost-model correction (dev_doc.md Section 5).** The original per-combination time
   estimate assumed a generic small output-token budget per benchmark task-type. The real
   configured `max_new_tokens` per role (128/256/512 in the full-size heterogeneous config)
   and measured real decode throughput (~5.4 tok/s, not the assumed ~10 tok/s) made the
   3-agent comm-modes (`token_based_mas`/`latent_based_mas_ours`, all 3 agents run per task)
   cost ~26-28h/instance at n=100/language -- about 6x the original estimate. Both running
   instances were killed and relaunched with `configs/latent_coordination_heterogeneous_
   timeboxed.yaml` (uniform 96 tokens, n=50/language) to bring this back to ~4-6h/instance.
5. **`unbabel-comet` dependency conflict (config fix, disabled by default).** Listed in
   `pyproject.toml` but never actually installed in this environment. `pip install
   unbabel-comet` resolved to a dependency set that force-upgraded `transformers`/
   `accelerate` (4.46.3->4.57.6, 1.1.1->1.13.0 observed), breaking the versions this
   pipeline's agent generation is pinned to -- and even after installing it and reverting
   the transformers/accelerate upgrade, a real run then crashed with `ValueError: Backend
   should be defined in the BACKENDS_MAPPING. Offending backend: tensorflow_text` inside
   COMET's own tokenizer init. `xcomet`/`cometkiwi` are now disabled by default in
   `configs/latent_coordination_heterogeneous_timeboxed.yaml` and
   `configs/latent_coordination.yaml` pending an isolated-env fix. `chrf` (sacrebleu-only,
   no such dependency risk) stays on.
6. **`accuracy=0.000` for `single_agent_baseline` on the `lo,am,sw` instance is real,
   not a crash -- but not comparable across instances.** `single_agent_baseline` only
   executes whichever agent the router selects first. On this instance the router
   selected `agent_safety` for all 151/151 tasks observed, and `SafetyAgent.process`
   always formats its output as `[SAFE]` / `[UNSAFE: ...]` -- which
   `_compute_accuracy` explicitly treats as a non-substantive response (starts with
   `[`), by design, to avoid miscounting raw safety verdicts as answers. So every
   response on this instance scored 0 under the completeness proxy, correctly per its
   definition, but the resulting 0.000 says nothing about translation quality -- it
   reflects a routing artifact, not a broken pipeline. The `th,my,km` instance routed
   to `agent_reason` instead (a substantive answer format), so its accuracy number is
   meaningful. Treat `single_agent_baseline`'s `accuracy` as unreliable whenever the
   router happens to pick the safety agent. **Correction: `chrf` is equally
   contaminated, not exempt** -- `_eval_single_agent` passes the exact same lead-agent
   `output_text` list to both `_compute_accuracy` and `_compute_translation_quality`
   (`benchmark_runner.py` lines ~177-188), so when the lead agent is `agent_safety`,
   `chrf` is scored against `[SAFE]`/`[UNSAFE: ...]` strings too, not an actual
   translation. On this instance, *both* `accuracy` and `chrf` for
   `single_agent_baseline` are meaningless, not just `accuracy`. `token_based_mas` and
   `latent_based_mas_ours` don't have this problem: they score the *last non-safety*
   response in the chain (`eval.scoring::select_answer`), so all agents always run
   regardless of lead-agent choice.
7. **CRITICAL (fixed in code): `AgentTask.context` leaked the gold FLORES+ translation
   into every agent's own prompt.** `benchmark_runner._load_real_tasks` set
   `context=tgt_text` (the gold target-language translation), and both
   `TranslationAgent._build_translation_prompt` and `ReasoningAgent._build_cot_prompt`
   embed `task.context` verbatim in their prompt (`SafetyAgent.process` prepends it to
   the text it evaluates too). Confirmed empirically before the fix: a
   `ReasoningAgent` response on `flores_plus_th_1` directly echoed the Thai reference
   it had been handed as "context," inside a `<think>` block -- not a translation of
   the English source, a copy of the answer key. This contaminated `chrf`/`accuracy`
   for `single_agent_baseline` and the *first* hop of every comm-mode on every prior
   run in this session. **Fix:** added a separate `AgentTask.reference` field (read
   only by `_compute_translation_quality` for scoring); `_load_real_tasks` now sets
   `reference=tgt_text` and leaves `context` empty for the first hop. All results
   reported before this fix (all runs prior to 2026-07-02 ~16:20 UTC) should be
   considered contaminated and are superseded by the numbers in this document.
   `router.route()` was checked and confirmed unaffected (it only ever encoded
   `task.query`, never `task.context`), so the earlier routing-asymmetry finding
   (#3/#6) remains valid independent of this fix.
8. **`token_cost` is not comparable across these language groups -- likely a
   whitespace-tokenization artifact, not a real efficiency difference.** Post-fix
   `single_agent_baseline`: `th_my_km` averages 6.2 words/response vs `lo_am_sw`'s
   21.3, despite `th_my_km` having *higher* latency (22356ms vs 17807ms) -- more
   time to produce fewer "words" is the opposite of what a real difference would
   predict. `token_cost` is computed as `len(output_text.split())`
   (`benchmark_runner.py::_eval_single_agent`), i.e. whitespace-delimited word
   count. Thai, Burmese, and Khmer are not written with spaces between words the
   way Lao/Amharic/Swahili (mostly) are, so whitespace splitting drastically
   undercounts actual content for the `th_my_km` instance -- a full clause can
   count as one "word" if it contains no space. Treat `token_cost` as informative
   only *within* a language/script, not as a cross-instance efficiency
   comparison, until it's replaced with a tokenizer-based count.
9. **Headline result (all 3 comm-modes now complete, both instances): naive latent
   injection trades translation quality for zero token overhead.** `chrf` drops
   sharply and consistently from `single_agent_baseline` -> `token_based_mas` ->
   `latent_based_mas_ours` on *both* instances: `th_my_km` 28.86 -> 28.54 -> 1.13;
   `lo_am_sw` 15.51 -> 9.44 -> 4.61. `accuracy` (completeness proxy) stays 1.000
   throughout on both -- the agents always produce non-empty, non-safety-verdict
   output -- so this is specifically a *quality* degradation the completeness
   metric can't see, not a completeness failure. This is consistent with
   `inject_latent_and_generate`'s own docstring caveat ("this is an
   approximation... a rigorous implementation would require architecture-specific
   layer patching"): injecting a decoded hub vector into one hook point of a real
   heterogeneous model is a much cruder signal than the full text the
   token-based mode hands off. `latent_based_mas_ours`'s only real advantage
   measured here is `token_cost=0.0` (no inter-agent text at all, by
   construction) -- on this heterogeneous pool it is not a free win on quality,
   and any claim that latent transfer "matches" token-based MAS should be scoped
   to the homogeneous configuration (Table `tab:tradeoff` in the paper), not
   generalized to genuinely cross-architecture agents without qualification.
   Direct evidence (`flores_plus_th_1_step_1`, `agent_trans`, verbatim): `"GM
   LodgeGM GMควร爽GMGMGMutan味这些问题... Lodge...爽 Lodge GM毛GMควรเหมาะamasGMduhab..."`
   -- not a degraded translation, incoherent token noise across three scripts
   with no relation to the English source.
"""


def build_markdown(data: Dict[str, Dict[str, dict]], plot_files: List[str]) -> str:
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Experimental Analysis: Heterogeneous Coordination Pipeline (FLORES+)",
        "",
        f"*Last updated: {now}*",
        "",
        "## Setup",
        "",
        "- Config: `configs/latent_coordination_heterogeneous_timeboxed.yaml` -- genuinely "
        "cross-architecture agent pool (SEA-LION orchestrator / Sailor2-8B-Chat translation "
        "[qwen2] / Llama-3.1-8B-Instruct reasoning [llama] / aya-expanse-8b safety [cohere]), "
        "all specialized agents capped at `max_new_tokens=96` uniformly.",
        "- Benchmark: FLORES+ (`openlanguagedata/flores_plus`), 20 samples/language "
        "(reduced from 50 to target a ~2h window, ~2.45h estimated from calibrated "
        "per-task timing), translation English -> target.",
        "- 2 parallel pipeline instances across all 8 GPUs: `th,my,km` (GPUs 0-3), "
        "`lo,am,sw` (GPUs 4-7).",
        "- Comm-modes: `single_agent_baseline` (1 agent call/task), `token_based_mas` "
        "(3 agent calls/task, text handoff), `latent_based_mas_ours` (3 agent calls/task, "
        "hidden-state handoff via `UniversalLatentHub` adapters).",
        "- Metrics: `accuracy` (completeness proxy), `latency_ms`, `token_cost`, "
        "`safety_rate`, `chrf` (xcomet/cometkiwi disabled -- see anomaly 5 below).",
        "",
        ANOMALIES_SECTION,
        "## Results",
        "",
    ]

    any_data = any(data[inst.name] for inst in INSTANCES)
    if not any_data:
        lines.append("*No comm-mode has finished on either instance yet. This section "
                      "will populate automatically as each mode completes -- re-run "
                      "`python scripts/build_experimental_report.py` after each "
                      "\"Mode '...' complete\" log line.*")
    else:
        for mode in MODES:
            lines.append(f"### `{mode}` ({MODE_LABELS[mode]})")
            lines.append("")
            done = [inst for inst in INSTANCES if mode in data[inst.name]]
            if not done:
                lines.append("*Not finished on either instance yet.*")
                lines.append("")
                continue
            lines.append("| Instance | Languages | accuracy | latency_ms | token_cost | safety_rate | chrf |")
            lines.append("|---|---|---|---|---|---|---|")
            for inst in INSTANCES:
                m = data[inst.name].get(mode)
                if m is None:
                    lines.append(f"| {inst.name} | {','.join(inst.languages)} | *pending* | | | | |")
                    continue
                lines.append(
                    f"| {inst.name} | {','.join(inst.languages)} | "
                    f"{m.get('accuracy', float('nan')):.3f} | "
                    f"{m.get('latency_ms', float('nan')):.1f} | "
                    f"{m.get('token_cost', float('nan')):.1f} | "
                    f"{m.get('safety_rate', float('nan')):.3f} | "
                    f"{m.get('chrf', float('nan')):.2f} |"
                )
            lines.append("")

    lines.append("## Plots")
    lines.append("")
    if plot_files:
        for f in plot_files:
            lines.append(f"![{f}](plots/{f})")
            lines.append("")
    else:
        lines.append("*No plots yet -- generated once at least one comm-mode has "
                      "completed on both instances.*")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = gather()
    plot_files = make_plots(data)
    md = build_markdown(data, plot_files)
    out_path = OUT_DIR / "REPORT.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path}")
    for inst in INSTANCES:
        done_modes = list(data[inst.name].keys())
        print(f"  {inst.name}: modes complete = {done_modes or '(none yet)'}")


if __name__ == "__main__":
    main()
