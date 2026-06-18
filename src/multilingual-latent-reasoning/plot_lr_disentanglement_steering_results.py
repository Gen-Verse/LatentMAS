#!/usr/bin/env python3
"""Plot LatentMAS language-reasoning steering results from existing CSVs.

The script scans a results root for run directories containing
`latent_agent_similarity_examples.csv` or its `.partial.csv` variant. It then
extracts exact-match accuracy across language-reasoning steering strengths and
writes paper-style comparison figures.

No model is loaded and no GPU is used.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULTS_ROOT = Path(
    "src/multilingual-latent-reasoning/results_latent_mas_agents/Qwen3-4B"
)
LANG_ORDER = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]


def infer_strength(run_name: str) -> float | None:
    if "baseline" in run_name or re.search(r"(^|_)s0($|_)", run_name):
        return 0.0
    match = re.search(r"(?:^|_)s([0-9]+(?:\.[0-9]+)?)(?:$|_)", run_name)
    if match:
        return float(match.group(1))
    match = re.search(r"strength[_-]?([0-9]+(?:\.[0-9]+)?)", run_name)
    if match:
        return float(match.group(1))
    return None


def find_examples_csv(run_dir: Path) -> Path | None:
    final_csv = run_dir / "latent_agent_similarity_examples.csv"
    partial_csv = run_dir / "latent_agent_similarity_examples.partial.csv"
    if final_csv.exists():
        return final_csv
    if partial_csv.exists():
        return partial_csv
    return None


def discover_runs(results_root: Path, pattern: str | None) -> list[dict]:
    run_dirs = [p for p in sorted(results_root.iterdir()) if p.is_dir()]
    rows = []
    for run_dir in run_dirs:
        if pattern and pattern not in run_dir.name:
            continue
        strength = infer_strength(run_dir.name)
        if strength is None:
            continue
        csv_path = find_examples_csv(run_dir)
        if csv_path is None:
            continue
        rows.append(
            {
                "run": run_dir.name,
                "run_dir": run_dir,
                "csv_path": csv_path,
                "strength": strength,
                "status": "final" if csv_path.name.endswith("examples.csv") else "partial",
            }
        )
    return rows


def normalize_correct(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)


def load_run(run: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(run["csv_path"])
    required = {"lang", "idx", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{run['csv_path']} is missing columns: {sorted(missing)}")
    df = df.copy()
    df["correct_float"] = normalize_correct(df["correct"])
    df["run"] = run["run"]
    df["strength"] = run["strength"]
    df["status"] = run["status"]
    if "prediction" not in df.columns:
        df["prediction"] = np.nan
    if "gold" not in df.columns:
        df["gold"] = np.nan

    lang = (
        df.groupby(["run", "strength", "status", "lang"], as_index=False)
        .agg(
            accuracy=("correct_float", "mean"),
            correct=("correct_float", "sum"),
            total=("correct_float", "size"),
        )
        .sort_values(["strength", "lang"])
    )
    return df, lang


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"[plot] {path}")


def style_axes(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_macro_accuracy(lang_summary: pd.DataFrame, out_dir: Path) -> None:
    macro = (
        lang_summary.groupby(["run", "strength", "status"], as_index=False)
        .agg(macro_accuracy=("accuracy", "mean"), examples=("total", "sum"))
        .sort_values("strength")
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(macro["strength"], macro["macro_accuracy"], marker="o", linewidth=2.3)
    for _, row in macro.iterrows():
        ax.annotate(
            f"{row['macro_accuracy']:.3f}",
            (row["strength"], row["macro_accuracy"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    style_axes(
        ax,
        "Macro Accuracy Across L-R Steering Strength",
        "Steering strength alpha",
        "Macro exact-match accuracy",
    )
    ax.set_ylim(0, min(1.0, max(0.05, float(macro["macro_accuracy"].max()) + 0.12)))
    savefig(out_dir / "macro_accuracy_vs_steering_strength.png")


def plot_language_curves(lang_summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    langs = [l for l in LANG_ORDER if l in set(lang_summary["lang"])]
    for lang in langs:
        part = lang_summary[lang_summary["lang"] == lang].sort_values("strength")
        ax.plot(part["strength"], part["accuracy"], marker="o", linewidth=1.8, label=lang)
    style_axes(
        ax,
        "Per-Language Accuracy Across Steering Strength",
        "Steering strength alpha",
        "Exact-match accuracy",
    )
    ax.set_ylim(0, 1)
    ax.legend(ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    savefig(out_dir / "language_accuracy_curves.png")


def plot_delta_heatmap(lang_summary: pd.DataFrame, out_dir: Path) -> None:
    baseline = lang_summary[lang_summary["strength"] == 0.0][["lang", "accuracy"]]
    if baseline.empty:
        print("[skip] no baseline strength 0.0 found for delta heatmap")
        return
    baseline = baseline.rename(columns={"accuracy": "baseline_accuracy"})
    df = lang_summary.merge(baseline, on="lang", how="left")
    df["delta_accuracy"] = df["accuracy"] - df["baseline_accuracy"]
    pivot = df.pivot_table(index="lang", columns="strength", values="delta_accuracy")
    pivot = pivot.reindex([l for l in LANG_ORDER if l in pivot.index])

    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = max(0.05, float(np.max(np.abs(finite)))) if finite.size else 0.05

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    im = ax.imshow(values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), labels=[f"{x:g}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_title("Accuracy Change Relative to Baseline", fontsize=13, pad=10)
    ax.set_xlabel("Steering strength alpha")
    ax.set_ylabel("Language")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Delta accuracy")
    savefig(out_dir / "delta_accuracy_heatmap_vs_baseline.png")


def plot_best_vs_baseline(lang_summary: pd.DataFrame, out_dir: Path) -> None:
    baseline = lang_summary[lang_summary["strength"] == 0.0][["lang", "accuracy"]]
    if baseline.empty:
        print("[skip] no baseline strength 0.0 found for best-vs-baseline plot")
        return
    best = (
        lang_summary.sort_values("accuracy", ascending=False)
        .groupby("lang", as_index=False)
        .first()[["lang", "strength", "accuracy"]]
        .rename(columns={"strength": "best_strength", "accuracy": "best_accuracy"})
    )
    df = baseline.merge(best, on="lang", how="inner").rename(
        columns={"accuracy": "baseline_accuracy"}
    )
    df["gain"] = df["best_accuracy"] - df["baseline_accuracy"]
    df = df.set_index("lang").reindex([l for l in LANG_ORDER if l in set(df["lang"])]).reset_index()

    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    ax.bar(x - width / 2, df["baseline_accuracy"], width, label="baseline", color="#6b7280")
    ax.bar(x + width / 2, df["best_accuracy"], width, label="best steered", color="#3f7f93")
    for i, row in df.iterrows():
        ax.text(
            i + width / 2,
            min(0.98, row["best_accuracy"] + 0.025),
            f"s={row['best_strength']:g}",
            ha="center",
            fontsize=7,
        )
    ax.set_xticks(x, labels=df["lang"])
    ax.set_ylim(0, 1)
    style_axes(ax, "Baseline vs Best Steering Strength by Language", "Language", "Accuracy")
    ax.legend(frameon=False)
    savefig(out_dir / "baseline_vs_best_steered_accuracy.png")


def plot_win_loss(examples: pd.DataFrame, out_dir: Path) -> None:
    baseline = examples[examples["strength"] == 0.0][["lang", "idx", "correct_float"]]
    if baseline.empty:
        print("[skip] no baseline strength 0.0 found for win-loss plot")
        return
    baseline = baseline.rename(columns={"correct_float": "baseline_correct"})
    steered = examples[examples["strength"] != 0.0].merge(baseline, on=["lang", "idx"], how="inner")
    if steered.empty:
        print("[skip] no steered examples overlap baseline for win-loss plot")
        return
    steered["change"] = steered["correct_float"] - steered["baseline_correct"]
    rows = []
    for (strength, lang), group in steered.groupby(["strength", "lang"]):
        rows.append(
            {
                "strength": strength,
                "lang": lang,
                "fixed": int((group["change"] == 1).sum()),
                "broken": int((group["change"] == -1).sum()),
                "net": int(group["change"].sum()),
            }
        )
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="lang", columns="strength", values="net")
    pivot = pivot.reindex([l for l in LANG_ORDER if l in pivot.index])
    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = max(1.0, float(np.max(np.abs(finite)))) if finite.size else 1.0

    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    im = ax.imshow(values, cmap="PiYG", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), labels=[f"{x:g}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_title("Net Problems Fixed by Steering vs Baseline", fontsize=13, pad=10)
    ax.set_xlabel("Steering strength alpha")
    ax.set_ylabel("Language")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{int(val):+d}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fixed - broken")
    savefig(out_dir / "net_fixed_problem_heatmap_vs_baseline.png")


def plot_problem_sensitivity(examples: pd.DataFrame, out_dir: Path) -> None:
    """Show which problem indices are most helped or hurt across all steering levels."""
    baseline = examples[examples["strength"] == 0.0][["lang", "idx", "correct_float"]]
    steered = examples[examples["strength"] != 0.0].merge(
        baseline.rename(columns={"correct_float": "baseline_correct"}),
        on=["lang", "idx"],
        how="inner",
    )
    if steered.empty:
        return
    steered["change"] = steered["correct_float"] - steered["baseline_correct"]
    by_idx = (
        steered.groupby("idx", as_index=False)
        .agg(mean_delta=("change", "mean"), net_delta=("change", "sum"), comparisons=("change", "size"))
        .sort_values("mean_delta", ascending=False)
    )
    top = pd.concat([by_idx.head(12), by_idx.tail(12)]).drop_duplicates("idx")

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    colors = np.where(top["mean_delta"] >= 0, "#3f7f93", "#b23a48")
    ax.bar(top["idx"].astype(str), top["mean_delta"], color=colors)
    style_axes(
        ax,
        "Most Helped / Hurt MGSM Problem Indices",
        "Problem index",
        "Mean correctness delta vs baseline",
    )
    ax.axhline(0, color="#222222", linewidth=0.9)
    savefig(out_dir / "problem_indices_most_helped_hurt.png")
    by_idx.to_csv(out_dir / "problem_index_steering_sensitivity.csv", index=False)


def write_tables(examples: pd.DataFrame, lang_summary: pd.DataFrame, out_dir: Path) -> None:
    macro = (
        lang_summary.groupby(["run", "strength", "status"], as_index=False)
        .agg(macro_accuracy=("accuracy", "mean"), micro_accuracy=("correct", "sum"), total=("total", "sum"))
        .sort_values("strength")
    )
    macro["micro_accuracy"] = macro["micro_accuracy"] / macro["total"]
    macro.to_csv(out_dir / "steering_macro_summary.csv", index=False)

    wide = lang_summary.pivot_table(index="strength", columns="lang", values="accuracy")
    wide = wide[[l for l in LANG_ORDER if l in wide.columns]]
    wide.to_csv(out_dir / "steering_language_accuracy_table.csv")

    baseline = lang_summary[lang_summary["strength"] == 0.0][["lang", "accuracy"]].rename(
        columns={"accuracy": "baseline_accuracy"}
    )
    if not baseline.empty:
        delta = lang_summary.merge(baseline, on="lang", how="left")
        delta["delta_accuracy"] = delta["accuracy"] - delta["baseline_accuracy"]
        delta.to_csv(out_dir / "steering_language_delta_table.csv", index=False)

    examples[
        ["run", "strength", "status", "lang", "idx", "correct_float", "prediction", "gold"]
    ].to_csv(out_dir / "steering_example_correctness_long.csv", index=False)


def write_index(out_dir: Path, runs: list[dict]) -> None:
    lines = [
        "# L-R Disentanglement Steering Figures",
        "",
        "These figures were produced from existing LatentMAS CSV outputs. No LRS or hidden-state recomputation is used.",
        "",
        "## Included Runs",
        "",
    ]
    for run in runs:
        lines.append(
            f"- `{run['run']}`: strength `{run['strength']:g}`, `{run['status']}`, source `{run['csv_path']}`"
        )
    lines.extend(["", "## Figures", ""])
    for png in sorted(out_dir.glob("*.png")):
        title = png.stem.replace("_", " ").title()
        lines.extend([f"### {title}", "", f"![{title}]({png.name})", ""])
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {out_dir / 'README.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--pattern",
        default="mgsm_first50_latent_mas",
        help="Substring used to select run directories. Use '' to include all strength-detectable runs.",
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.results_root / "lr_disentanglement_steering_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = args.pattern if args.pattern else None
    runs = discover_runs(args.results_root, pattern)
    if not runs:
        raise SystemExit(f"No steering runs found under {args.results_root} with pattern={args.pattern!r}")

    example_frames = []
    lang_frames = []
    for run in runs:
        print(f"[read] strength={run['strength']:g} status={run['status']} {run['csv_path']}")
        examples, lang_summary = load_run(run)
        example_frames.append(examples)
        lang_frames.append(lang_summary)

    examples = pd.concat(example_frames, ignore_index=True)
    lang_summary = pd.concat(lang_frames, ignore_index=True)

    write_tables(examples, lang_summary, out_dir)
    plot_macro_accuracy(lang_summary, out_dir)
    plot_language_curves(lang_summary, out_dir)
    plot_delta_heatmap(lang_summary, out_dir)
    plot_best_vs_baseline(lang_summary, out_dir)
    plot_win_loss(examples, out_dir)
    plot_problem_sensitivity(examples, out_dir)
    write_index(out_dir, runs)


if __name__ == "__main__":
    main()
