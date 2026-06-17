import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


STAGES = {
    "shared_after_planner": ["planner"],
    "shared_after_critic": ["planner", "critic"],
    "shared_after_refiner": ["planner", "critic", "refiner"],
    "shared_with_judger": ["planner", "critic", "refiner", "judger"],
}


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), eps)
    return float(np.dot(a, b) / denom)


def pooled_shared_vector(example: Dict, roles: List[str], layer_strategy: str) -> np.ndarray:
    vectors = []
    for role in roles:
        hidden = example["agents"][role]["hidden"].astype(np.float32)
        if layer_strategy == "final_layer":
            role_vectors = hidden[:, -1, :]
        elif layer_strategy == "all_layers":
            role_vectors = hidden.reshape(-1, hidden.shape[-1])
        else:
            raise ValueError(f"Unsupported layer_strategy: {layer_strategy}")
        vectors.append(role_vectors)
    return np.concatenate(vectors, axis=0).mean(axis=0)


def load_payload(path: Path) -> Dict:
    with path.open("rb") as f:
        return pickle.load(f)


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_vectors(payload: Dict, layer_strategy: str) -> Dict:
    vectors = {}
    for lang in payload["languages"]:
        vectors[lang] = {}
        for ex in payload["traces"][lang]:
            idx = int(ex["idx"])
            vectors[lang][idx] = {}
            for stage, roles in STAGES.items():
                vectors[lang][idx][stage] = pooled_shared_vector(ex, roles, layer_strategy)
    return vectors


def build_problem_rows(payload: Dict, vectors: Dict) -> List[Dict]:
    langs = payload["languages"]
    by_lang = {
        lang: {int(ex["idx"]): ex for ex in payload["traces"][lang]}
        for lang in langs
    }
    idxs = sorted(set.intersection(*(set(by_lang[lang]) for lang in langs)))

    rows = []
    for idx in idxs:
        for stage in STAGES:
            for lang in langs:
                ex = by_lang[lang][idx]
                row = {
                    "stage": stage,
                    "lang": lang,
                    "idx": idx,
                    "correct": bool(ex["correct"]),
                    "prediction": ex["prediction"],
                    "gold": ex["gold"],
                    "pooled_norm": float(np.linalg.norm(vectors[lang][idx][stage])),
                }
                vals = []
                for other in langs:
                    val = cosine(vectors[lang][idx][stage], vectors[other][idx][stage])
                    row[f"cosine_to_{other}"] = val
                    if other != lang:
                        vals.append(val)
                row["mean_cosine_to_other_languages"] = float(np.mean(vals)) if vals else 1.0
                rows.append(row)
    return rows


def build_cosine_matrix_rows(payload: Dict, vectors: Dict) -> List[Dict]:
    langs = payload["languages"]
    rows = []
    for stage in STAGES:
        for lang in langs:
            row = {"stage": stage, "lang": lang}
            for other in langs:
                vals = []
                common = sorted(set(vectors[lang]) & set(vectors[other]))
                for idx in common:
                    vals.append(cosine(vectors[lang][idx][stage], vectors[other][idx][stage]))
                row[f"cosine_to_{other}"] = float(np.mean(vals)) if vals else float("nan")
            rows.append(row)
    return rows


def build_language_summary_rows(problem_rows: List[Dict], language_summary: Dict) -> List[Dict]:
    df = pd.DataFrame(problem_rows)
    rows = []
    for (stage, lang), group in df.groupby(["stage", "lang"], sort=False):
        cosine_to_en = pd.to_numeric(group.get("cosine_to_en"), errors="coerce")
        correct = group["correct"].astype(float)
        saved_summary = language_summary.get(lang, {})
        total = int(saved_summary.get("total", len(group)))
        num_correct = int(saved_summary.get("correct", correct.sum()))
        accuracy = float(saved_summary.get("accuracy", correct.mean()))
        row = {
            "stage": stage,
            "lang": lang,
            "accuracy": accuracy,
            "correct": num_correct,
            "total": total,
            "mean_cosine_to_en": float(cosine_to_en.mean()),
            "median_cosine_to_en": float(cosine_to_en.median()),
            "mean_cosine_to_other_languages": float(group["mean_cosine_to_other_languages"].mean()),
            "mean_pooled_norm": float(group["pooled_norm"].mean()),
            "corr_problem_cosine_to_en_with_correct": float(cosine_to_en.corr(correct))
            if cosine_to_en.nunique(dropna=True) > 1 and correct.nunique(dropna=True) > 1
            else float("nan"),
        }
        rows.append(row)
    return rows


def build_stage_summary_rows(language_rows: List[Dict]) -> List[Dict]:
    df = pd.DataFrame(language_rows)
    rows = []
    for stage, group in df.groupby("stage", sort=False):
        accuracy = group["accuracy"].astype(float)
        cos_en = group["mean_cosine_to_en"].astype(float)
        cos_other = group["mean_cosine_to_other_languages"].astype(float)
        rows.append(
            {
                "stage": stage,
                "mean_accuracy": float(accuracy.mean()),
                "mean_cosine_to_en": float(cos_en.mean()),
                "mean_cosine_to_other_languages": float(cos_other.mean()),
                "corr_language_cosine_to_en_with_accuracy": float(cos_en.corr(accuracy)),
                "corr_language_cosine_to_other_languages_with_accuracy": float(cos_other.corr(accuracy)),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to latent_mas_mgsm_batch_traces.pkl.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write shared latent-space CSVs.",
    )
    parser.add_argument(
        "--layer_strategy",
        choices=["final_layer", "all_layers"],
        default="final_layer",
        help="Pool final-layer step vectors or all step/layer vectors.",
    )
    args = parser.parse_args()

    payload = load_payload(Path(args.input))
    vectors = build_vectors(payload, args.layer_strategy)

    problem_rows = build_problem_rows(payload, vectors)
    matrix_rows = build_cosine_matrix_rows(payload, vectors)
    language_rows = build_language_summary_rows(problem_rows, payload["language_summary"])
    stage_rows = build_stage_summary_rows(language_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "shared_latent_stage_problem_rows.csv", problem_rows)
    write_csv(out_dir / "shared_latent_stage_cosine_matrix.csv", matrix_rows)
    write_csv(out_dir / "shared_latent_stage_language_summary.csv", language_rows)
    write_csv(out_dir / "shared_latent_stage_summary.csv", stage_rows)

    print(f"[OK] wrote {out_dir}")
    print(pd.DataFrame(stage_rows).to_string(index=False))


if __name__ == "__main__":
    main()
