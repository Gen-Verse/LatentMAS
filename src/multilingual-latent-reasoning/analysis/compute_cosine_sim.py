import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


# ------------------------------------------------------------
# Utilities from your accuracy script (simplified)
# ------------------------------------------------------------

_LANG_NORMALIZE = {
    "en": "EN",
    "fr": "FR",
    "de": "DE",
    "zh": "ZH",
    "ja": "JA",
    "ru": "RU",
    "es": "ES",
    "sw": "SW",
    "bn": "BN",
    "te": "TE",
    "th": "TH",
}


def normalize_lang_key(lang: str) -> str:
    if not lang:
        return "EN"
    key = lang.strip()
    lower = key.lower()
    return _LANG_NORMALIZE.get(lower, key.upper())


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_answer(ans) -> Tuple[Optional[float], str]:
    """
    Normalize numeric-ish answers.

    Returns:
        (numeric_value_or_None, canonical_string)
    """
    import re

    if ans is None:
        return None, ""

    s = str(ans)

    # number just before '}' if any
    m = re.search(r"([-+]?\d[\d,]*(?:\.\d+)?)(?=\s*})", s)
    if not m:
        # fallback: first number anywhere
        m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", s)

    if not m:
        return None, ""

    num_str_raw = m.group(0)
    num_str_clean = num_str_raw.replace(",", "")

    try:
        val = float(num_str_clean)
    except ValueError:
        return None, ""

    if abs(val - int(val)) < 1e-12:
        canon = str(int(val))
    else:
        canon = str(val).rstrip("0").rstrip(".")

    return val, canon


def compare_answers(pred_raw: Optional[str], gold_raw) -> bool:
    """
    Compare predicted answer string and gold answer.
    """
    if pred_raw is None:
        return False

    pred_num, pred_str = normalize_answer(pred_raw)
    gold_num, gold_str = normalize_answer(gold_raw)

    if not gold_str:
        return False

    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) < 1e-6

    return pred_str == gold_str


# ------------------------------------------------------------
# Cosine similarity helpers
# ------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """
    Cosine similarity between two 1D vectors a, b.
    """
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    denom = max(na * nb, eps)
    return dot / denom


# ------------------------------------------------------------
# Core analysis
# ------------------------------------------------------------

def build_idx_to_steps(examples: List[Dict[str, Any]]) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Build mapping:
      idx -> { percent -> hidden_arr }

    where hidden_arr has shape (num_layers, hidden_dim).
    """
    out: Dict[int, Dict[int, np.ndarray]] = {}
    for ex in examples:
        idx = int(ex["idx"])
        step_dict: Dict[int, np.ndarray] = {}
        for step in ex["steps"]:
            p = int(step["percent"])
            h = step["hidden"]  # (num_layers, hidden_dim)
            step_dict[p] = h
        out[idx] = step_dict
    return out


def compute_correctness_for_target(
    trunc_path: Path,
    k: int,
    mode: str = "hack",
    direction: str = "forward",
    ratio_key: str = "100%",
    max_examples: Optional[int] = None,
) -> Dict[int, bool]:
    """
    For the TARGET language, compute correctness per example index
    using pass@k for a specific mode/direction/ratio.

    Returns:
      idx_to_correct: dict[int -> bool]
    """
    if not trunc_path.exists():
        raise FileNotFoundError(f"Truncation file not found: {trunc_path}")

    data = load_json(trunc_path)
    if max_examples is not None:
        data = data[:max_examples]

    idx_to_correct: Dict[int, bool] = {}

    for ex in data:
        idx_val = int(ex.get("idx", 0))
        gold = ex.get("gold_answer", None)

        mode_obj = ex.get(mode, {})
        dir_obj = mode_obj.get(direction, {})
        ratio_data = dir_obj.get(ratio_key, None)
        if not ratio_data:
            # If missing, treat as incorrect (or you could skip)
            idx_to_correct[idx_val] = False
            continue

        responses = ratio_data.get("response", [])
        if isinstance(responses, str):
            responses = [responses]

        correct = False
        for pred in responses[:k]:
            if compare_answers(pred, gold):
                correct = True
                break

        idx_to_correct[idx_val] = correct

    return idx_to_correct


def aggregate_cosine_similarity(
    ref_idx2steps: Dict[int, Dict[int, np.ndarray]],
    tgt_idx2steps: Dict[int, Dict[int, np.ndarray]],
    step_percents: List[int],
    idx_filter: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Compute cosine similarity across all common examples between ref and tgt,
    aggregated over examples.

    Args:
      ref_idx2steps, tgt_idx2steps:
        idx -> { percent -> hidden_arr (num_layers, hidden_dim) }

      step_percents:
        List of percentages (ints) defining step order.

      idx_filter:
        Optional list of idx's to restrict to (e.g., correct-only).

    Returns:
      sim[steps, layers]: average cosine similarities.
    """
    common_idxs = sorted(set(ref_idx2steps.keys()) & set(tgt_idx2steps.keys()))
    if idx_filter is not None:
        common_idxs = [i for i in common_idxs if i in idx_filter]

    if not common_idxs:
        raise ValueError("No common indices between ref and tgt for the given filter.")

    # Determine dimensions from first available example
    example_idx = common_idxs[0]
    first_percent = step_percents[0]
    H_ref = ref_idx2steps[example_idx][first_percent]
    num_layers, hidden_dim = H_ref.shape

    num_steps = len(step_percents)
    sim_sum = np.zeros((num_steps, num_layers), dtype=np.float64)
    sim_cnt = np.zeros((num_steps, num_layers), dtype=np.int64)

    for idx in common_idxs:
        ref_steps = ref_idx2steps[idx]
        tgt_steps = tgt_idx2steps[idx]

        for s_idx, p in enumerate(step_percents):
            if p not in ref_steps or p not in tgt_steps:
                continue

            H_ref = ref_steps[p]  # (num_layers, hidden_dim)
            H_tgt = tgt_steps[p]

            if H_ref.shape != H_tgt.shape:
                raise ValueError(
                    f"Shape mismatch for idx={idx}, percent={p}: "
                    f"ref {H_ref.shape}, tgt {H_tgt.shape}"
                )

            for l in range(num_layers):
                v_ref = H_ref[l].astype(np.float32)
                v_tgt = H_tgt[l].astype(np.float32)
                cs = cosine_sim(v_ref, v_tgt)
                sim_sum[s_idx, l] += cs
                sim_cnt[s_idx, l] += 1

    # Avoid division by zero
    sim_cnt_safe = np.where(sim_cnt == 0, 1, sim_cnt)
    sim_avg = sim_sum / sim_cnt_safe

    return sim_avg.astype(np.float32)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        help="Model name (used only to derive model_base).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        # default="juletxara/mgsm",
        # default="shanchen/aime_2024_multilingual",
        default="shanchen/aime_2025_multilingual",
        help="Full dataset name to derive dataset_base.",
    )
    parser.add_argument(
        "--repr_root",
        type=str,
        default="../repr_hidden",
        help="Root directory of *_hidden_states_repr.pkl files.",
    )
    parser.add_argument(
        "--trunc_root",
        type=str,
        default="../results_trunc",
        help="Root directory of *_trunc.json files (for correctness).",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="./cosine_similarities",
        help="Root directory of cosine similarities.",
    )
    parser.add_argument(
        "--ref_lang",
        type=str,
        default="EN",
        help="Reference language (e.g. EN).",
    )
    parser.add_argument(
        "--tgt_langs",
        type=str,
        default="FR,DE,ZH,JA,RU,ES,SW,BN,TE,TH",
        help="Comma-separated target languages, e.g. 'FR,DE,ZH'.",
    )
    parser.add_argument(
        "--input_trunc_suffix",
        type=str,
        default="trunc",
        help="Suffix for trunc JSON: {lang}_{suffix}.json (default: trunc).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Pass@k used to define correctness for target languages.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hack",
        help="Mode for correctness (default: hack).",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="forward",
        help="Direction for correctness (default: forward).",
    )
    parser.add_argument(
        "--ratio",
        type=str,
        default="100%",
        help="Ratio key for correctness (default: 100%%).",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional limit on number of examples when reading trunc JSON.",
    )

    args = parser.parse_args()

    dataset_base = args.dataset_name.split("/")[-1]
    model_base = args.model_name.split("/")[-1]

    ref_lang_norm = normalize_lang_key(args.ref_lang)
    ref_lang_suffix = ref_lang_norm.lower()

    tgt_lang_list_raw = [l.strip() for l in args.tgt_langs.split(",") if l.strip()]
    tgt_lang_norms = [normalize_lang_key(l) for l in tgt_lang_list_raw]
    tgt_lang_suffixes = [ln.lower() for ln in tgt_lang_norms]

    repr_root = Path(args.repr_root)
    trunc_root = Path(args.trunc_root)
    out_root = Path(args.out_root)

    # ---------- Load reference repr pickle ONCE ----------
    ref_pkl_path = (
        repr_root / dataset_base / model_base / f"{ref_lang_suffix}_hidden_states_repr.pkl"
    )
    if not ref_pkl_path.exists():
        raise FileNotFoundError(f"Reference repr pickle not found: {ref_pkl_path}")

    print(f"Loading reference repr from: {ref_pkl_path}")
    with ref_pkl_path.open("rb") as f:
        ref_data = pickle.load(f)

    ref_steps = ref_data["meta"]["step_percents"]
    step_percents = ref_steps
    print(f"Step percents: {step_percents}")

    ref_idx2steps = build_idx_to_steps(ref_data["examples"])

    # ---------- Loop over target languages ----------
    for tgt_lang_norm, tgt_lang_suffix in zip(tgt_lang_norms, tgt_lang_suffixes):
        print(f"\n========== Target language: {tgt_lang_norm} ==========")

        tgt_pkl_path = (
            repr_root / dataset_base / model_base / f"{tgt_lang_suffix}_hidden_states_repr.pkl"
        )
        if not tgt_pkl_path.exists():
            print(f"[SKIP] Target repr pickle not found: {tgt_pkl_path}")
            continue

        print(f"Loading target repr from: {tgt_pkl_path}")
        with tgt_pkl_path.open("rb") as f:
            tgt_data = pickle.load(f)

        tgt_steps = tgt_data["meta"]["step_percents"]
        if tgt_steps != ref_steps:
            raise ValueError(
                f"step_percents mismatch between ref ({ref_steps}) and tgt ({tgt_steps})"
            )

        tgt_idx2steps = build_idx_to_steps(tgt_data["examples"])

        common_idxs_all = sorted(set(ref_idx2steps.keys()) & set(tgt_idx2steps.keys()))
        print(f"#common idxs between {ref_lang_norm} and {tgt_lang_norm}: {len(common_idxs_all)}")

        # ---------- Correctness for this target language ----------
        trunc_path_tgt = (
            trunc_root / dataset_base / model_base / f"{tgt_lang_suffix}_{args.input_trunc_suffix}.json"
        )
        print(f"Loading target truncation data from: {trunc_path_tgt}")
        idx_to_correct = compute_correctness_for_target(
            trunc_path_tgt,
            k=args.k,
            mode=args.mode,
            direction=args.direction,
            ratio_key=args.ratio,
            max_examples=args.max_examples,
        )

        correct_idxs = [i for i in common_idxs_all if idx_to_correct.get(i, False)]
        incorrect_idxs = [i for i in common_idxs_all if not idx_to_correct.get(i, False)]

        print(f"#correct examples (tgt={tgt_lang_norm}, pass@{args.k} at {args.ratio}, "
              f"{args.mode}/{args.direction}): {len(correct_idxs)}")
        print(f"#incorrect examples: {len(incorrect_idxs)}")

        # ---------- Aggregate cosine similarities ----------
        print("Computing cosine similarity for ALL examples...")
        sim_all = aggregate_cosine_similarity(
            ref_idx2steps,
            tgt_idx2steps,
            step_percents=step_percents,
            idx_filter=common_idxs_all,
        )

        sim_correct = None
        sim_incorrect = None

        if correct_idxs:
            print("Computing cosine similarity for CORRECT examples...")
            sim_correct = aggregate_cosine_similarity(
                ref_idx2steps,
                tgt_idx2steps,
                step_percents=step_percents,
                idx_filter=correct_idxs,
            )

        if incorrect_idxs:
            print("Computing cosine similarity for INCORRECT examples...")
            sim_incorrect = aggregate_cosine_similarity(
                ref_idx2steps,
                tgt_idx2steps,
                step_percents=step_percents,
                idx_filter=incorrect_idxs,
            )

        # ---------- Save to NPZ ----------
        # Pattern: {out_root}/{dataset}/{model}/{tgt_ref}_cosine_similarity.npz
        out_dir = out_root / dataset_base / model_base
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{tgt_lang_suffix}_{ref_lang_suffix}_cosine_similarity.npz"

        meta = {
            "dataset": dataset_base,
            "model": model_base,
            "ref_lang": ref_lang_norm,
            "tgt_lang": tgt_lang_norm,
            "step_percents": step_percents,
            "k": args.k,
            "mode": args.mode,
            "direction": args.direction,
            "ratio": args.ratio,
        }

        save_kwargs: Dict[str, Any] = {
            "sim_all": sim_all,
            "step_percents": np.array(step_percents, dtype=np.int32),
            "meta_json": json.dumps(meta),
        }
        if sim_correct is not None:
            save_kwargs["sim_correct"] = sim_correct
        if sim_incorrect is not None:
            save_kwargs["sim_incorrect"] = sim_incorrect

        np.savez(out_path, **save_kwargs)
        print(save_kwargs)
        print(f"[OK] Saved cosine similarity for {tgt_lang_norm} → {out_path}")

    print("\nAll target languages completed.")


if __name__ == "__main__":
    main()
