import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


STAGES = {
    "shared_after_planner": ["planner"],
    "shared_after_critic": ["planner", "critic"],
    "shared_after_refiner": ["planner", "critic", "refiner"],
    "shared_with_judger": ["planner", "critic", "refiner", "judger"],
}


def load_pickle(path: Path) -> Dict:
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


def safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return float("nan")
    if a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return float("nan")
    if method == "spearman":
        # Avoid pandas importing scipy.stats.spearmanr, which can fail on older
        # system libstdc++ installs. Spearman is Pearson over ranked values.
        a = a[mask].rank(method="average")
        b = b[mask].rank(method="average")
        return float(a.corr(b, method="pearson"))
    return float(a[mask].corr(b[mask], method=method))


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def select_layer_values(matrix: np.ndarray, layer_strategy: str) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError(f"Expected step x layer matrix, got shape {matrix.shape}")
    if layer_strategy == "final_layer":
        return matrix[:, -1]
    if layer_strategy == "best_layer":
        return matrix.min(axis=1)
    if layer_strategy == "mean_layer":
        return matrix.mean(axis=1)
    raise ValueError(f"Unsupported layer_strategy: {layer_strategy}")


def concat_shared_path(example: Dict, roles: Iterable[str], key: str, layer_strategy: str) -> np.ndarray:
    pieces = []
    for role in roles:
        matrix = np.asarray(example["agents"][role]["logitlens"][key], dtype=np.float32)
        pieces.append(select_layer_values(matrix, layer_strategy))
    return np.concatenate(pieces, axis=0)


def emergence_score(rank_path: np.ndarray, threshold: int) -> Dict:
    emerged = np.where(rank_path <= threshold)[0]
    if len(emerged) == 0:
        return {
            "emergence_position": None,
            "latent_reasoning_score": 0.0,
            "emerged": False,
        }
    pos = int(emerged[0])
    return {
        "emergence_position": pos,
        "latent_reasoning_score": float(1.0 - (pos / float(len(rank_path)))),
        "emerged": True,
    }


def path_summary(rank_path: np.ndarray, logprob_path: np.ndarray) -> Dict:
    best_pos = int(np.argmin(rank_path))
    final_pos = int(len(rank_path) - 1)
    return {
        "path_len": int(len(rank_path)),
        "best_rank": float(rank_path[best_pos]),
        "best_rank_position": best_pos,
        "final_rank": float(rank_path[final_pos]),
        "best_logprob": float(np.max(logprob_path)),
        "best_logprob_position": int(np.argmax(logprob_path)),
        "final_logprob": float(logprob_path[final_pos]),
    }


def build_problem_rows(payload: Dict, thresholds: List[int], layer_strategy: str) -> List[Dict]:
    rows = []
    for lang in payload["languages"]:
        for example in payload["traces"][lang]:
            for stage, roles in STAGES.items():
                rank_path = concat_shared_path(example, roles, "rank_gold_first", layer_strategy)
                logprob_path = concat_shared_path(example, roles, "logprob_gold_first", layer_strategy)
                base = {
                    "lang": lang,
                    "idx": int(example["idx"]),
                    "stage": stage,
                    "roles": "+".join(roles),
                    "correct": bool(example["correct"]),
                    "prediction": example.get("prediction"),
                    "gold": example.get("gold"),
                    "layer_strategy": layer_strategy,
                }
                base.update(path_summary(rank_path, logprob_path))
                for threshold in thresholds:
                    row = dict(base)
                    row["rank_threshold"] = threshold
                    row.update(emergence_score(rank_path, threshold))
                    rows.append(row)
    return rows


def add_rank_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["correct_float"] = out["correct"].astype(str).str.lower().isin(("true", "1")).astype(float)
    out["best_rank_score"] = 1.0 / (1.0 + np.log1p(pd.to_numeric(out["best_rank"], errors="coerce")))
    out["final_rank_score"] = 1.0 / (1.0 + np.log1p(pd.to_numeric(out["final_rank"], errors="coerce")))
    out["best_inverse_rank"] = 1.0 / pd.to_numeric(out["best_rank"], errors="coerce").clip(lower=1.0)
    out["final_inverse_rank"] = 1.0 / pd.to_numeric(out["final_rank"], errors="coerce").clip(lower=1.0)
    out["logprob_gain"] = (
        pd.to_numeric(out["final_logprob"], errors="coerce")
        - pd.to_numeric(out["best_logprob"], errors="coerce")
    )
    return out


def build_score_correlation_rows(problem_rows: List[Dict]) -> List[Dict]:
    df = add_rank_score_columns(pd.DataFrame(problem_rows))
    candidate_cols = [
        "latent_reasoning_score",
        "emerged",
        "best_rank_score",
        "final_rank_score",
        "best_inverse_rank",
        "final_inverse_rank",
        "best_logprob",
        "final_logprob",
        "logprob_gain",
    ]
    rows = []
    for (stage, threshold), group in df.groupby(["stage", "rank_threshold"], sort=False):
        language_group = group.groupby("lang", as_index=False).mean(numeric_only=True)
        for score in candidate_cols:
            rows.append(
                {
                    "stage": stage,
                    "rank_threshold": int(threshold),
                    "score": score,
                    "example_pearson": safe_corr(group[score], group["correct_float"], "pearson"),
                    "example_spearman": safe_corr(group[score], group["correct_float"], "spearman"),
                    "language_pearson": safe_corr(
                        language_group[score],
                        language_group["correct_float"],
                        "pearson",
                    ),
                    "language_spearman": safe_corr(
                        language_group[score],
                        language_group["correct_float"],
                        "spearman",
                    ),
                    "score_mean_correct": float(group.loc[group["correct_float"] == 1.0, score].mean()),
                    "score_mean_wrong": float(group.loc[group["correct_float"] == 0.0, score].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out["accuracy_aligned"] = out["language_pearson"] > 0
    out["language_abs_pearson"] = out["language_pearson"].abs()
    out["example_abs_pearson"] = out["example_pearson"].abs()
    out = out.sort_values(
        ["accuracy_aligned", "language_abs_pearson", "example_abs_pearson"],
        ascending=[False, False, False],
    )
    return out.to_dict(orient="records")


def build_language_summary_rows(problem_rows: List[Dict]) -> List[Dict]:
    df = add_rank_score_columns(pd.DataFrame(problem_rows))
    rows = []
    for (stage, threshold, lang), group in df.groupby(
        ["stage", "rank_threshold", "lang"],
        sort=False,
    ):
        rows.append(
            {
                "stage": stage,
                "rank_threshold": int(threshold),
                "lang": lang,
                "accuracy": float(group["correct_float"].mean()),
                "total": int(len(group)),
                "shared_lrs": float(group["latent_reasoning_score"].mean()),
                "shared_emergence_rate": float(group["emerged"].mean()),
                "mean_best_rank": float(group["best_rank"].mean()),
                "mean_final_rank": float(group["final_rank"].mean()),
                "mean_best_rank_score": float(group["best_rank_score"].mean()),
                "mean_final_rank_score": float(group["final_rank_score"].mean()),
                "mean_best_logprob": float(group["best_logprob"].mean()),
                "mean_final_logprob": float(group["final_logprob"].mean()),
                "corr_problem_shared_lrs_with_correct": safe_corr(
                    group["latent_reasoning_score"],
                    group["correct_float"],
                    "pearson",
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to latent_mas_mgsm_batch_traces.pkl.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory for shared latent reasoning score CSVs.",
    )
    parser.add_argument(
        "--thresholds",
        default="1,5,10,25,50,100,250,500,1000,2500,5000,10000",
        help="Comma-separated gold first-token rank thresholds.",
    )
    parser.add_argument(
        "--layer_strategy",
        choices=["final_layer", "best_layer", "mean_layer"],
        default="final_layer",
        help="How to reduce the layer dimension before computing shared-path emergence.",
    )
    args = parser.parse_args()

    payload = load_pickle(Path(args.input))
    thresholds = parse_int_list(args.thresholds)
    problem_rows = build_problem_rows(payload, thresholds, args.layer_strategy)
    correlation_rows = build_score_correlation_rows(problem_rows)
    language_rows = build_language_summary_rows(problem_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "shared_lrs_problem_rows.csv", problem_rows)
    write_csv(out_dir / "shared_lrs_language_summary.csv", language_rows)
    write_csv(out_dir / "shared_lrs_score_correlations.csv", correlation_rows)

    top = pd.DataFrame(correlation_rows).head(25)
    top.to_csv(out_dir / "shared_lrs_top25.csv", index=False)

    print(f"[OK] wrote {out_dir}")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
