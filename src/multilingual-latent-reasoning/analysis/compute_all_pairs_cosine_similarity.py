import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"



# -------------------------
# Language normalization
# -------------------------
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


# -------------------------
# IO helpers
# -------------------------
def load_pkl(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Build idx -> percent -> hidden
# -------------------------
def build_idx_to_steps(examples: List[Dict[str, Any]]) -> Dict[int, Dict[int, np.ndarray]]:
    out: Dict[int, Dict[int, np.ndarray]] = {}
    for ex in examples:
        idx = int(ex["idx"])
        sd: Dict[int, np.ndarray] = {}
        for st in ex["steps"]:
            p = int(st["percent"])
            sd[p] = st["hidden"]
        out[idx] = sd
    return out


# -------------------------
# Answer correctness (pass@k), copied/simplified from your code
# -------------------------
def normalize_answer(ans) -> Tuple[Optional[float], str]:
    import re
    if ans is None:
        return None, ""
    s = str(ans)
    m = re.search(r"([-+]?\d[\d,]*(?:\.\d+)?)(?=\s*})", s)
    if not m:
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
    if pred_raw is None:
        return False
    pred_num, pred_str = normalize_answer(pred_raw)
    gold_num, gold_str = normalize_answer(gold_raw)
    if not gold_str:
        return False
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) < 1e-6
    return pred_str == gold_str

def compute_idx_to_correct(
    trunc_path: Path,
    k: int,
    mode: str,
    direction: str,
    ratio_key: str,
    max_examples: Optional[int] = None,
) -> Dict[int, bool]:
    data = load_json(trunc_path)
    if max_examples is not None:
        data = data[:max_examples]

    idx_to_correct: Dict[int, bool] = {}
    for ex in data:
        idx_val = int(ex.get("idx", 0))
        gold = ex.get("gold_answer", None)

        mode_obj = ex.get(mode, {})
        dir_obj = mode_obj.get(direction, {})
        ratio_obj = dir_obj.get(ratio_key, None)
        if not ratio_obj:
            idx_to_correct[idx_val] = False
            continue

        responses = ratio_obj.get("response", [])
        if isinstance(responses, str):
            responses = [responses]

        correct = False
        for pred in responses[:k]:
            if compare_answers(pred, gold):
                correct = True
                break

        idx_to_correct[idx_val] = correct

    return idx_to_correct


# -------------------------
# Cosine similarity (vectorized per layer)
# -------------------------
def cosine_per_layer(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    A, B: (L, D) float32
    return: (L,) float32
    """
    dot = np.sum(A * B, axis=1)
    na = np.linalg.norm(A, axis=1)
    nb = np.linalg.norm(B, axis=1)
    denom = np.maximum(na * nb, eps)
    return (dot / denom).astype(np.float32)


def aggregate_pair_similarity_with_counts(
    idx2steps_a: Dict[int, Dict[int, np.ndarray]],
    idx2steps_b: Dict[int, Dict[int, np.ndarray]],
    step_percents: List[int],
    idx_filter: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate sim over common idxs (and optional idx_filter).
    Returns:
      sim_avg: float32 [S, L]
      cnt:     int32   [S, L]   (#examples contributed to each (step,layer))
    """
    common_idxs = sorted(set(idx2steps_a.keys()) & set(idx2steps_b.keys()))
    if idx_filter is not None:
        idx_set = set(idx_filter)
        common_idxs = [i for i in common_idxs if i in idx_set]

    if not common_idxs:
        # return empty later at caller
        raise ValueError("No common idxs after filtering.")

    # infer shape from first idx that contains first percent in both
    first_percent = step_percents[0]
    found = None
    for idx in common_idxs:
        if first_percent in idx2steps_a[idx] and first_percent in idx2steps_b[idx]:
            found = idx
            break
    if found is None:
        raise ValueError("No common idx has the first step percent in both languages.")

    H0 = idx2steps_a[found][first_percent]
    S = len(step_percents)
    L = H0.shape[0]

    sim_sum = np.zeros((S, L), dtype=np.float64)
    cnt = np.zeros((S, L), dtype=np.int64)

    for idx in common_idxs:
        sa = idx2steps_a[idx]
        sb = idx2steps_b[idx]

        for s_idx, p in enumerate(step_percents):
            if p not in sa or p not in sb:
                continue

            A = sa[p].astype(np.float32)
            B = sb[p].astype(np.float32)

            if A.shape != B.shape:
                raise ValueError(f"Shape mismatch idx={idx}, p={p}: {A.shape} vs {B.shape}")

            cs = cosine_per_layer(A, B)  # (L,)
            sim_sum[s_idx] += cs
            cnt[s_idx] += 1

    cnt_safe = np.where(cnt == 0, 1, cnt)
    sim_avg = (sim_sum / cnt_safe).astype(np.float32)

    return sim_avg, cnt.astype(np.int32)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--dataset_name", 
                        type=str, 
                        # default="juletxara/mgsm",
                        # default="shanchen/aime_2024_multilingual",
                        default="shanchen/aime_2025_multilingual",
                        )

    parser.add_argument("--repr_root", type=str, default="../repr_hidden")
    parser.add_argument("--trunc_root", type=str, default="../results_trunc")
    parser.add_argument("--out_root", type=str, default="./cosine_similarities_allpairs")

    parser.add_argument("--langs", type=str, default="EN,FR,DE,ZH,JA,RU,ES,SW,BN,TE,TH")

    # correctness config (for lang_i)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--mode", type=str, default="hack")
    parser.add_argument("--direction", type=str, default="forward")
    parser.add_argument("--ratio", type=str, default="100%")
    parser.add_argument("--trunc_suffix", type=str, default="trunc")
    parser.add_argument("--max_trunc_examples", type=int, default=None)

    parser.add_argument("--skip_existing", action="store_true")

    args = parser.parse_args()

    dataset_base = args.dataset_name.split("/")[-1]
    model_base = args.model_name.split("/")[-1]
    repr_root = Path(args.repr_root)
    trunc_root = Path(args.trunc_root)
    out_root = Path(args.out_root)

    langs_norm = [normalize_lang_key(x) for x in args.langs.split(",") if x.strip()]
    langs_norm = list(dict.fromkeys(langs_norm))  # stable unique

    # ---- load all repr pickles ----
    lang_idx2steps: Dict[str, Dict[int, Dict[int, np.ndarray]]] = {}
    step_percents_ref: Optional[List[int]] = None

    print(f"Dataset: {dataset_base} | Model: {model_base}")
    print(f"Languages requested ({len(langs_norm)}): {langs_norm}")

    for lang in langs_norm:
        pkl_path = repr_root / dataset_base / model_base / f"{lang.lower()}_hidden_states_repr.pkl"
        if not pkl_path.exists():
            print(f"[WARN] Missing repr pickle for {lang}: {pkl_path}")
            continue

        data = load_pkl(pkl_path)
        sp = data["meta"]["step_percents"]

        if step_percents_ref is None:
            step_percents_ref = sp
        else:
            if sp != step_percents_ref:
                raise ValueError(f"step_percents mismatch for {lang}: {sp} vs {step_percents_ref}")

        lang_idx2steps[lang] = build_idx_to_steps(data["examples"])
        print(f"[OK] Loaded {lang}: {len(data['examples'])} examples")

    if step_percents_ref is None or len(lang_idx2steps) < 2:
        raise RuntimeError("Need at least two languages with valid repr pickles.")

    langs_loaded = [l for l in langs_norm if l in lang_idx2steps]
    N = len(langs_loaded)
    step_percents = step_percents_ref
    print(f"\nLoaded languages ({N}): {langs_loaded}")
    print(f"Step percents: {step_percents}")

    out_dir = out_root / dataset_base / model_base
    out_dir.mkdir(parents=True, exist_ok=True)

    # For each language_i, compute sims to all language_j
    for i, lang_i in enumerate(langs_loaded):
        out_path = out_dir / f"{lang_i.lower()}_all_pairs_cosine_similarity.npz"
        if args.skip_existing and out_path.exists():
            print(f"[SKIP] Exists: {out_path}")
            continue

        print(f"\n========== Building per-language file for: {lang_i} ==========")

        # correctness split for lang_i
        trunc_path_i = trunc_root / dataset_base / model_base / f"{lang_i.lower()}_{args.trunc_suffix}.json"
        if not trunc_path_i.exists():
            print(f"[WARN] Missing trunc file for {lang_i} (cannot do correct/incorrect): {trunc_path_i}")
            idx_to_correct = {}
        else:
            idx_to_correct = compute_idx_to_correct(
                trunc_path=trunc_path_i,
                k=args.k,
                mode=args.mode,
                direction=args.direction,
                ratio_key=args.ratio,
                max_examples=args.max_trunc_examples,
            )

        # Determine idx sets (only for lang_i)
        idxs_i = set(lang_idx2steps[lang_i].keys())
        correct_i = [idx for idx, ok in idx_to_correct.items() if ok and idx in idxs_i]
        incorrect_i = [idx for idx, ok in idx_to_correct.items() if (not ok) and idx in idxs_i]

        print(f"{lang_i}: #idx with repr={len(idxs_i)} | #correct={len(correct_i)} | #incorrect={len(incorrect_i)}")

        # infer shape S,L from any partner
        # (we’ll allocate tensors [N,S,L])
        S = len(step_percents)
        # pick one example to infer L
        any_idx = next(iter(lang_idx2steps[lang_i].keys()))
        any_percent = step_percents[0]
        L = lang_idx2steps[lang_i][any_idx][any_percent].shape[0]

        sim_all_to = np.full((N, S, L), np.nan, dtype=np.float32)
        sim_cor_to = np.full((N, S, L), np.nan, dtype=np.float32)
        sim_inc_to = np.full((N, S, L), np.nan, dtype=np.float32)

        cnt_all_to = np.zeros((N, S, L), dtype=np.int32)
        cnt_cor_to = np.zeros((N, S, L), dtype=np.int32)
        cnt_inc_to = np.zeros((N, S, L), dtype=np.int32)

        # compute against all lang_j
        for j, lang_j in enumerate(langs_loaded):
            if lang_i == lang_j:
                continue

            print(f"  Pair: {lang_i}-{lang_j}")

            # ALL
            try:
                sim_all, cnt_all = aggregate_pair_similarity_with_counts(
                    lang_idx2steps[lang_i], lang_idx2steps[lang_j], step_percents, idx_filter=None
                )
                sim_all_to[j] = sim_all
                cnt_all_to[j] = cnt_all
            except Exception as e:
                print(f"    [WARN] sim_all failed for {lang_i}-{lang_j}: {e}")

            # CORRECT (based on lang_i)
            if correct_i:
                try:
                    sim_c, cnt_c = aggregate_pair_similarity_with_counts(
                        lang_idx2steps[lang_i], lang_idx2steps[lang_j], step_percents, idx_filter=correct_i
                    )
                    sim_cor_to[j] = sim_c
                    cnt_cor_to[j] = cnt_c
                except Exception as e:
                    print(f"    [WARN] sim_correct failed for {lang_i}-{lang_j}: {e}")

            # INCORRECT (based on lang_i)
            if incorrect_i:
                try:
                    sim_ic, cnt_ic = aggregate_pair_similarity_with_counts(
                        lang_idx2steps[lang_i], lang_idx2steps[lang_j], step_percents, idx_filter=incorrect_i
                    )
                    sim_inc_to[j] = sim_ic
                    cnt_inc_to[j] = cnt_ic
                except Exception as e:
                    print(f"    [WARN] sim_incorrect failed for {lang_i}-{lang_j}: {e}")

        meta = {
            "dataset": dataset_base,
            "model": model_base,
            "lang_i": lang_i,
            "langs_order": langs_loaded,
            "step_percents": step_percents,
            "correctness_defined_by": {
                "lang": lang_i,
                "k": args.k,
                "mode": args.mode,
                "direction": args.direction,
                "ratio": args.ratio,
                "trunc_suffix": args.trunc_suffix,
            },
            "note": "sim_*_to_langs[j] corresponds to sim(lang_i, langs_order[j]) over steps x layers, aggregated over matching idx.",
        }

        np.savez(
            out_path,
            step_percents=np.array(step_percents, dtype=np.int32),
            langs_json=json.dumps(langs_loaded),
            sim_all_to_langs=sim_all_to,
            sim_correct_to_langs=sim_cor_to,
            sim_incorrect_to_langs=sim_inc_to,
            cnt_all_to_langs=cnt_all_to,
            cnt_correct_to_langs=cnt_cor_to,
            cnt_incorrect_to_langs=cnt_inc_to,
            meta_json=json.dumps(meta),
        )

        print(f"[OK] Saved: {out_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
