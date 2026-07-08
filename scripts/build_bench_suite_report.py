"""Builds results/experimental_analysis/REPORT_bench_suite.md + plots from the
bench_suite runs (het/hom x belebele_sg/mgsm). Re-run any time — it reads whatever
per-mode results have landed in each config's checkpoint cache so far (each comm-mode
is cached by MultiAgentBenchmarkRunner as soon as it finishes), applies the post-hoc
safety-verdict reparse sidecars written by scripts/recompute_safety_rate.py, and
overlays the LatentMAS/ThoughtComm baseline runs.

Usage:
    python scripts/build_bench_suite_report.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict
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
REPORT_PATH = OUT_DIR / "REPORT_bench_suite.md"

# dataviz reference palette (validated set, fixed slot order — never re-cycled).
# het -> slot 1 (blue), hom -> slot 2 (aqua) everywhere; slot 3 reserved.
SERIES = {"het": "#2a78d6", "hom": "#1baf7a"}
SEQ_LIGHT, SEQ_DARK = "#86b6ef", "#2a78d6"  # blue ramp steps 250 / 450 (old -> reparsed)
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

MODES = ["single_agent_baseline", "token_based_mas", "latent_based_mas_ours"]
MODE_LABELS = {
    "single_agent_baseline": "Single agent",
    "token_based_mas": "Token MAS",
    "latent_based_mas_ours": "Latent MAS (ours)",
}

CONFIGS = ["het_belebele_sg", "hom_belebele_sg", "het_mgsm", "hom_mgsm"]
BELEBELE_LANGS = ["en", "th", "my", "km", "lo", "am", "sw", "bn", "te"]


def _cache_root(config: str) -> Path:
    return REPO_ROOT / ".cache" / "checkpoints" / "bench_suite" / config / "coordination" / "_results"


def _lang_of(task_id: str) -> str:
    for p in task_id.split("_"):
        if len(p) == 2 and p.isalpha():
            return p
    return "??"


def load_config_modes(config: str) -> Dict[str, dict]:
    """{mode: {"metrics", "per_lang_latency", "reparse"}} for finished modes."""
    root = _cache_root(config)
    out: Dict[str, dict] = {}
    if not root.exists():
        return out
    for p in sorted(root.glob("*__mode__*.pt")):
        mode = p.name.split("__mode__")[-1].removesuffix(".pt")
        if mode not in MODES or p.name.endswith(".stale-safety-bug"):
            continue
        payload = torch.load(p, map_location="cpu", weights_only=False)
        obj = payload.get("obj", payload)
        metrics = dict(obj.get("metrics", {}))
        per_lang = defaultdict(list)
        for d in obj.get("task_details", []):
            per_lang[_lang_of(d.get("task_id", ""))].append(float(d.get("elapsed_ms") or 0.0))
        entry = {
            "metrics": metrics,
            "per_lang_latency_ms": {k: statistics.mean(v) for k, v in per_lang.items() if v},
        }
        sidecar = p.with_name(p.name + ".reparsed.json")
        if sidecar.exists():
            entry["reparse"] = json.loads(sidecar.read_text())
        out[mode] = entry
    return out


def load_baselines() -> List[dict]:
    rows = []
    for method_dir in (REPO_ROOT / "results" / "baselines").iterdir():
        if not method_dir.is_dir():
            continue
        for f in sorted(method_dir.glob("*.json")):
            d = json.loads(f.read_text())
            rows.append({
                "method": method_dir.name,
                "benchmark": d.get("benchmark"),
                "language": d.get("language"),
                "accuracy": d.get("accuracy"),
                "file": f.name,
            })
    return rows


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(BASELINE_AXIS)
    ax.spines["bottom"].set_color(BASELINE_AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _grouped_bars(ax, groups: List[str], series: Dict[str, List[Optional[float]]],
                  colors: Dict[str, str], fmt: str = "{:.2f}"):
    n = len(series)
    width = min(0.8 / max(n, 1), 0.3)
    for i, (name, vals) in enumerate(series.items()):
        xs = [x + (i - (n - 1) / 2) * width for x in range(len(groups))]
        plotted = [(x, v) for x, v in zip(xs, vals) if v is not None]
        bars = ax.bar([x for x, _ in plotted], [v for _, v in plotted],
                      width=width * 0.92, color=colors[name], label=name, zorder=3)
        for b, (_, v) in zip(bars, plotted):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)


def plot_accuracy_by_mode(data, baselines) -> Optional[str]:
    """Belebele accuracy per comm-mode, het vs hom, with the LatentMAS reference."""
    modes_present = [m for m in MODES
                     if any(m in data.get(f"{k}_belebele_sg", {}) for k in SERIES)]
    if not modes_present:
        return None
    series = {}
    for k in SERIES:
        vals = [data.get(f"{k}_belebele_sg", {}).get(m, {}).get("metrics", {}).get("accuracy_belebele")
                for m in modes_present]
        if any(v is not None for v in vals):
            series[k] = vals
    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
    _style_axes(ax)
    _grouped_bars(ax, [MODE_LABELS[m] for m in modes_present], series, SERIES)
    ref = next((b for b in baselines
                if b["method"] == "latentmas" and b["benchmark"] == "belebele"), None)
    if ref and ref["accuracy"] is not None:
        ax.axhline(ref["accuracy"], color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.annotate(f"LatentMAS (th only) {ref['accuracy']:.2f}",
                    xy=(0.99, ref["accuracy"]), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=7.5, color=INK_SECONDARY)
    ax.axhline(0.25, color=INK_MUTED, linewidth=0.9, linestyle=(0, (1, 2)), zorder=2)
    ax.annotate("chance 0.25", xy=(0.01, 0.25), xycoords=("axes fraction", "data"),
                ha="left", va="bottom", fontsize=7.5, color=INK_MUTED)
    ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
    ax.set_ylabel("Belebele accuracy (9 langs, log-likelihood)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Belebele accuracy by communication mode", color=INK_PRIMARY, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return _save(fig, "bench_suite_belebele_accuracy_by_mode.png")


def plot_safeguard(data) -> Optional[str]:
    """SEA-SafeguardBench: verdict-agreement accuracy and safety_rate old vs reparsed."""
    rows = []  # (config, mode, metrics, reparse)
    for cfg in ("het_belebele_sg", "hom_belebele_sg"):
        for m in MODES:
            e = data.get(cfg, {}).get(m)
            if e and "accuracy_sea_safeguardbench" in e.get("metrics", {}):
                rows.append((cfg, m, e["metrics"], e.get("reparse") or {}))
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), facecolor=SURFACE)
    labels = [f"{c.split('_')[0]}\n{MODE_LABELS[m]}" for c, m, _, _ in rows]

    ax = axes[0]
    _style_axes(ax)
    bars = ax.bar(range(len(rows)), [r[2]["accuracy_sea_safeguardbench"] for r in rows],
                  width=0.55, color=[SERIES[c.split("_")[0]] for c, _, _, _ in rows], zorder=3)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY)
    ax.axhline(0.5, color=INK_MUTED, linewidth=0.9, linestyle=(0, (1, 2)), zorder=2)
    ax.annotate("chance 0.5", xy=(0.01, 0.5), xycoords=("axes fraction", "data"),
                ha="left", va="bottom", fontsize=7.5, color=INK_MUTED)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Verdict agreement", color=INK_SECONDARY, fontsize=9)
    ax.set_title("SEA-SafeguardBench accuracy (th+my)", color=INK_PRIMARY, fontsize=10.5, loc="left")

    ax = axes[1]
    _style_axes(ax)
    pair = {
        "as logged": [r[3].get("safety_rate_old", r[2].get("safety_rate")) for r in rows],
        "reparsed": [r[3].get("safety_rate_new", r[2].get("safety_rate")) for r in rows],
    }
    _grouped_bars(ax, labels, pair, {"as logged": SEQ_LIGHT, "reparsed": SEQ_DARK})
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("safety_rate", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Safety pass rate: strict parser vs reparse", color=INK_PRIMARY, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return _save(fig, "bench_suite_safeguard.png")


def plot_per_lang_latency(data) -> Optional[str]:
    mode = "single_agent_baseline"
    series = {}
    for k in SERIES:
        lat = data.get(f"{k}_belebele_sg", {}).get(mode, {}).get("per_lang_latency_ms", {})
        vals = [lat.get(lg, None) and lat[lg] / 1000.0 for lg in BELEBELE_LANGS]
        if any(v is not None for v in vals):
            series[k] = vals
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    _style_axes(ax)
    _grouped_bars(ax, BELEBELE_LANGS, series, SERIES, fmt="{:.1f}")
    ax.set_ylabel("Mean latency (s/task)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Single-agent latency per Belebele language", color=INK_PRIMARY, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return _save(fig, "bench_suite_latency_per_lang.png")


def _save(fig, name: str) -> str:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / name, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return name


def build_report(data, baselines, plot_names: List[str]) -> str:
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Experimental Analysis: bench_suite (Belebele + SEA-SafeguardBench, MGSM pending)",
        "",
        f"*Last regenerated: {now} — re-run `python scripts/build_bench_suite_report.py` "
        "as further comm-modes finish.*",
        "",
        "## Scope & caveats",
        "",
        "- Configs: `het_belebele_sg` (SEA-LION orch / Sailor2 trans / Llama-3.1 reason / "
        "aya-expanse safety) and `hom_belebele_sg` (4x SEA-LION), n=200/lang, 9 Belebele "
        "languages + SEA-SafeguardBench th+my (100 sampled tasks/lang).",
        "- **Runs launched 2026-07-05T15:50 execute the pre-fix code**: degenerate uniform "
        "attention router (all 3 roles dispatched, confidence pinned ~0.335) and strict "
        "safety-verdict parser. Token/latent rows are therefore *fixed 3-agent pipeline* "
        "results, not adaptive routing; het `safety_rate` is corrected post-hoc by "
        "`scripts/recompute_safety_rate.py` (sidecars applied here automatically).",
        "- **Belebele accuracy is answer-independent across comm-modes** (teacher-forced "
        "log-likelihood probe of the scoring agent's model), so mode-vs-mode Belebele deltas "
        "reflect scoring-agent identity only. Mode differences show up in "
        "SEA-SafeguardBench agreement, latency, and token cost.",
        "",
        "## Results by config & mode",
        "",
    ]
    metric_cols = ["accuracy", "accuracy_belebele", "accuracy_sea_safeguardbench",
                   "safety_rate", "latency_ms", "token_cost"]
    header = "| config | mode | " + " | ".join(metric_cols) + " |"
    lines += [header, "|" + "---|" * (len(metric_cols) + 2)]
    for cfg in CONFIGS:
        modes = data.get(cfg) or {}
        if not modes:
            lines.append(f"| {cfg} | *(not started)* |" + " |" * len(metric_cols))
            continue
        for m in MODES:
            if m not in modes:
                continue
            met = modes[m]["metrics"]
            rep = modes[m].get("reparse") or {}
            cells = []
            for c in metric_cols:
                v = met.get(c)
                if c == "safety_rate" and "safety_rate_new" in rep:
                    cells.append(f"{rep['safety_rate_old']:.3f} -> **{rep['safety_rate_new']:.3f}** (reparsed)")
                elif isinstance(v, float):
                    cells.append(f"{v:.3f}" if c != "latency_ms" else f"{v:,.0f}")
                else:
                    cells.append("" if v is None else str(v))
            lines.append(f"| {cfg} | {MODE_LABELS[m]} | " + " | ".join(cells) + " |")
    lines += ["", "## Baselines (head-to-head references)", ""]
    lines += ["| method | benchmark | language | accuracy |", "|---|---|---|---|"]
    for b in baselines:
        lines.append(f"| {b['method']} | {b['benchmark']} | {b['language']} | {b['accuracy']:.3f} |")
    lines += [
        "",
        "> **Baseline rows dated ≤ 2026-07-06 are INVALID as method comparisons** "
        "(see results/baselines/README_INVALID.md): both runners discarded their "
        "communicated latent, making them byte-identical single-model prompt chains. "
        "Fixed via soft-prefix injection (baselines/latent_prefix.py); rerun before citing.",
        "",
        "## Plots",
        "",
    ]
    lines += [f"![{n}](plots/{n})" for n in plot_names]
    lines.append("")
    return "\n".join(lines)


def main():
    data = {cfg: load_config_modes(cfg) for cfg in CONFIGS}
    baselines = load_baselines()
    plot_names = [n for n in (
        plot_accuracy_by_mode(data, baselines),
        plot_safeguard(data),
        plot_per_lang_latency(data),
    ) if n]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(data, baselines, plot_names))
    print(f"wrote {REPORT_PATH}")
    for n in plot_names:
        print(f"wrote {PLOTS_DIR / n}")


if __name__ == "__main__":
    main()
