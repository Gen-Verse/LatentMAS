#!/usr/bin/env python3
"""Plot layer-wise latent cosine similarity to English from LatentMAS traces.

Expected trace format is the one produced by run_latent_mas_mgsm_batch_analysis.py:

    {
      "languages": [...],
      "traces": {
        "bn": [
          {
            "idx": 0,
            "agents": {
              "planner": {"hidden": np.ndarray[steps, layers, hidden_dim]},
              ...
            },
          },
          ...
        ],
      },
    }

The script compares each language/role/example hidden state to the matching
English role/example hidden state at each layer, then averages over examples,
roles, or both for presentation plots.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


LOW_RESOURCE_LANGS = {"bn", "sw", "te", "th"}


def load_pickle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "traces" not in payload:
        raise ValueError(f"{path} does not look like a LatentMAS trace pickle with a 'traces' key.")
    return payload


def trace_index(traces: dict, lang: str) -> Dict[int, dict]:
    return {int(row["idx"]): row for row in traces.get(lang, [])}


def vector_by_layer(hidden: np.ndarray, step_strategy: str) -> np.ndarray:
    arr = np.asarray(hidden, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Expected hidden shape [steps,layers,dim] or [layers,dim], got {arr.shape}")
    if step_strategy == "last":
        return arr[-1]
    if step_strategy == "mean":
        return arr.mean(axis=0)
    if step_strategy == "first":
        return arr[0]
    raise ValueError(f"Unknown step strategy: {step_strategy}")


def cosine_by_layer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"Cannot compare hidden states with shapes {a.shape} and {b.shape}")
    numerator = (a * b).sum(axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return numerator / np.maximum(denom, 1e-12)


def compute_layerwise_rows(
    payload: dict,
    label: str,
    ref_lang: str,
    roles: Iterable[str],
    step_strategy: str,
) -> List[dict]:
    traces = payload["traces"]
    languages = payload.get("languages") or sorted(traces)
    ref_examples = trace_index(traces, ref_lang)
    rows: List[dict] = []

    for lang in languages:
        examples = trace_index(traces, lang)
        common_indices = sorted(set(examples) & set(ref_examples))
        if not common_indices:
            continue
        for idx in common_indices:
            ex = examples[idx]
            ref_ex = ref_examples[idx]
            for role in roles:
                agent = ex.get("agents", {}).get(role)
                ref_agent = ref_ex.get("agents", {}).get(role)
                if not agent or not ref_agent:
                    continue
                hidden = agent.get("hidden")
                ref_hidden = ref_agent.get("hidden")
                if hidden is None or ref_hidden is None:
                    continue
                lang_layers = vector_by_layer(hidden, step_strategy)
                ref_layers = vector_by_layer(ref_hidden, step_strategy)
                cosines = cosine_by_layer(lang_layers, ref_layers)
                for layer, cosine in enumerate(cosines):
                    rows.append(
                        {
                            "run": label,
                            "lang": lang,
                            "idx": idx,
                            "role": role,
                            "layer": layer,
                            "cosine_to_english": float(cosine),
                            "is_low_resource": lang in LOW_RESOURCE_LANGS,
                        }
                    )
    return rows


def group_mean(rows: List[dict], keys: Tuple[str, ...]) -> List[dict]:
    buckets: Dict[Tuple, List[float]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        buckets.setdefault(key, []).append(float(row["cosine_to_english"]))
    out = []
    for key, vals in sorted(buckets.items()):
        result = {k: v for k, v in zip(keys, key)}
        result["mean_cosine_to_english"] = float(np.mean(vals))
        result["n"] = len(vals)
        out.append(result)
    return out


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_overall_curves(summary_rows: List[dict], out_path: Path) -> None:
    runs = sorted({r["run"] for r in summary_rows})
    plt.figure(figsize=(9, 5.2))
    for run in runs:
        xs = [int(r["layer"]) for r in summary_rows if r["run"] == run]
        ys = [float(r["mean_cosine_to_english"]) for r in summary_rows if r["run"] == run]
        plt.plot(xs, ys, marker="o", linewidth=2, markersize=3, label=run)
    plt.xlabel("Layer")
    plt.ylabel("Mean cosine similarity to English")
    plt.title("Layer-wise latent cosine similarity to English")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_language_facets(summary_rows: List[dict], out_path: Path) -> None:
    langs = [x for x in sorted({r["lang"] for r in summary_rows}) if x != "en"]
    runs = sorted({r["run"] for r in summary_rows})
    ncols = 4
    nrows = int(np.ceil(len(langs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.8 * nrows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, lang in zip(axes, langs):
        for run in runs:
            subset = [r for r in summary_rows if r["lang"] == lang and r["run"] == run]
            xs = [int(r["layer"]) for r in subset]
            ys = [float(r["mean_cosine_to_english"]) for r in subset]
            ax.plot(xs, ys, linewidth=1.8, label=run)
        ax.set_title(lang)
        ax.grid(alpha=0.2)
    for ax in axes[len(langs):]:
        ax.axis("off")
    fig.supxlabel("Layer")
    fig.supylabel("Mean cosine similarity to English")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(len(runs), 1), frameon=False)
    fig.suptitle("Cosine-to-English by language and layer", y=1.02)
    fig.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_delta_heatmap(summary_rows: List[dict], baseline_label: str, steered_label: str, out_path: Path) -> None:
    by_key = {(r["run"], r["lang"], int(r["layer"])): float(r["mean_cosine_to_english"]) for r in summary_rows}
    langs = [x for x in sorted({r["lang"] for r in summary_rows}) if x != "en"]
    layers = sorted({int(r["layer"]) for r in summary_rows})
    mat = np.full((len(langs), len(layers)), np.nan, dtype=np.float32)
    for i, lang in enumerate(langs):
        for j, layer in enumerate(layers):
            base = by_key.get((baseline_label, lang, layer))
            steered = by_key.get((steered_label, lang, layer))
            if base is not None and steered is not None:
                mat[i, j] = steered - base
    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 0.01
    vmax = max(vmax, 0.01)
    plt.figure(figsize=(12, 4.8))
    im = plt.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.yticks(range(len(langs)), langs)
    plt.xticks(range(len(layers)), layers, rotation=90)
    plt.xlabel("Layer")
    plt.ylabel("Language")
    plt.title(f"Delta cosine-to-English: {steered_label} minus {baseline_label}")
    cbar = plt.colorbar(im)
    cbar.set_label("Delta cosine")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run specs as label=/path/to/latent_mas_mgsm_batch_traces.pkl",
    )
    parser.add_argument("--ref_lang", default="en")
    parser.add_argument("--roles", default="planner,critic,refiner,judger")
    parser.add_argument("--step_strategy", choices=["first", "last", "mean"], default="mean")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--baseline_label", default=None)
    parser.add_argument("--steered_label", default=None)
    args = parser.parse_args()

    roles = [x.strip() for x in args.roles.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    labels: List[str] = []
    for spec in args.runs:
        if "=" not in spec:
            raise ValueError(f"Run spec must be label=path, got: {spec}")
        label, path_s = spec.split("=", 1)
        labels.append(label)
        payload = load_pickle(Path(path_s))
        all_rows.extend(compute_layerwise_rows(payload, label, args.ref_lang, roles, args.step_strategy))

    if not all_rows:
        raise RuntimeError("No layer-wise cosine rows were produced. Check roles/languages/trace structure.")

    layer_summary = group_mean(all_rows, ("run", "layer"))
    language_layer_summary = group_mean(all_rows, ("run", "lang", "layer"))
    role_layer_summary = group_mean(all_rows, ("run", "role", "layer"))

    write_csv(out_dir / "layerwise_cosine_to_english_long.csv", all_rows)
    write_csv(out_dir / "layerwise_cosine_to_english_by_run_layer.csv", layer_summary)
    write_csv(out_dir / "layerwise_cosine_to_english_by_run_lang_layer.csv", language_layer_summary)
    write_csv(out_dir / "layerwise_cosine_to_english_by_run_role_layer.csv", role_layer_summary)

    plot_overall_curves(layer_summary, out_dir / "overall_layerwise_cosine_to_english.png")
    plot_language_facets(language_layer_summary, out_dir / "language_layerwise_cosine_to_english.png")

    baseline_label = args.baseline_label or (labels[0] if labels else None)
    steered_label = args.steered_label or (labels[1] if len(labels) > 1 else None)
    if baseline_label and steered_label and baseline_label != steered_label:
        plot_delta_heatmap(
            language_layer_summary,
            baseline_label,
            steered_label,
            out_dir / "delta_layerwise_cosine_to_english_heatmap.png",
        )

    print(f"[OK] wrote {out_dir}")


if __name__ == "__main__":
    main()
