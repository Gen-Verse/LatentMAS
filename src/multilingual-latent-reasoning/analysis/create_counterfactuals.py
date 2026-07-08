#!/usr/bin/env python
import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------
# Import your helpers
# -----------------------
from memorization_test_helper import (
    get_client,
    single_number_edit,
    paraphrase_reorder_with_gemini,
)

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


# ---------------------------------------------------------------------
# Language normalization (same as your trunc script)
# ---------------------------------------------------------------------
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

LANG_NAME = {
    "EN": "English",
    "FR": "French",
    "DE": "German",
    "ZH": "Chinese",
    "JA": "Japanese",
    "RU": "Russian",
    "ES": "Spanish",
    "SW": "Swahili",
    "BN": "Bengali",
    "TE": "Telugu",
    "TH": "Thai",
}

def normalize_lang_key(lang: str) -> str:
    if not lang:
        return "EN"
    key = lang.strip()
    lower = key.lower()
    return _LANG_NORMALIZE.get(lower, key.upper())


# --------------------- basic loading / saving --------------------- #
def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# --------------------- answer normalization --------------------- #
def normalize_answer(ans) -> Tuple[Optional[float], str]:
    """
    Normalize numeric-ish answers.

    Returns: (numeric_value_or_None, canonical_string)
    """
    if ans is None:
        return None, ""

    s = str(ans)

    # number just before a '}' (allow commas/decimals), else first number anywhere
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


# --------------------- question extraction fallback --------------------- #
def extract_question_from_prompt(prompt: str) -> str:
    """
    For MGSM-like prompt:
      "<｜User｜>Problem: ... \n\nThe answer should format ..."
    Returns the extracted Problem body if found; else returns the whole prompt.
    """
    if not prompt:
        return ""
    m = re.search(
        r"Problem:\s*(.*?)(?:\n\s*\n|The answer should format|<\|Assistant\|>)",
        prompt,
        flags=re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return prompt.strip()


# --------------------- main --------------------- #
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="juletxara/mgsm,shanchen/aime_2024_multilingual,shanchen/aime_2025_multilingual",
        help="Comma-separated list of dataset names.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="EN,FR,DE,ZH,JA,RU,ES,SW,BN,TE,TH",
        help="Comma-separated list of language keys.",
    )
    parser.add_argument(
        "--input_trunc_root",
        type=str,
        default="../results_trunc",
        help="Root dir where *_trunc.json are stored.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="counterfactuals",
        help="Where to save counterfactual JSON files.",
    )

    # Selection knobs
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Select examples that are correct under pass@k.",
    )
    parser.add_argument(
        "--ratio",
        type=str,
        default="0%",
        help="Which truncation ratio to use for correctness check (default: 100%).",
    )

    # Counterfactual generation knobs
    parser.add_argument(
        "--numeric_preferred",
        type=str,
        default="first",
        choices=["first", "random"],
        help="Which number to edit: first safe or random safe.",
    )
    parser.add_argument(
        "--numeric_seed_base",
        type=int,
        default=42,
        help="Base seed for numeric edits (idx is mixed in).",
    )
    parser.add_argument(
        "--no_paraphrase",
        type=bool,
        default=False,
        help="If set, do NOT call Gemini; only produce numeric edits.",
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-2.5-flash",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.1,
        help="Sleep between Gemini calls (seconds).",
    )
    # Debug knobs
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optionally limit number of examples read per file (debug).",
    )
    parser.add_argument(
        "--max_selected",
        type=int,
        default=None,
        help="Optionally limit number of selected examples saved (debug).",
    )

    args = parser.parse_args()

    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    lang_list_raw = [l.strip() for l in args.languages.split(",") if l.strip()]

    input_trunc_root = Path(args.input_trunc_root)
    output_root = Path(args.output_root)

    model_base = args.model_name.split("/")[-1]

    # You said: only forward direction + hack mode
    MODE = "hack"
    DIRECTION = "forward"

    # Gemini client (optional)
    client = None
    if not args.no_paraphrase:
        client = get_client()

    print("=== Settings ===")
    print(f"Model:      {args.model_name}")
    print(f"Datasets:   {dataset_list}")
    print(f"Languages:  {lang_list_raw}")
    print(f"Select:     mode={MODE}, direction={DIRECTION}, ratio={args.ratio}, pass@{args.k}")
    print(f"Paraphrase: {not args.no_paraphrase} ({args.gemini_model})")
    print("================\n")

    for dataset_name in dataset_list:
        dataset_base = dataset_name.split("/")[-1]
        print(f"########## DATASET: {dataset_name} ##########")

        for lang in lang_list_raw:
            lang_norm = normalize_lang_key(lang)
            lang_suffix = lang_norm.lower()
            
            
            out_dir = output_root / dataset_base / model_base
            out_path = out_dir / f"{lang_suffix}_counterfactuals_passat{args.k}.json"

            if out_path.exists():
                print(f"[SKIP] Counterfactuals already exist → {out_path}")
                continue

            trunc_path = input_trunc_root / dataset_base / model_base / f"{lang_suffix}_trunc.json"
            if not trunc_path.exists():
                print(f"[SKIP] Missing: {trunc_path}")
                continue

            print(f"\n--- {dataset_base} | {lang_norm} ---")
            trunc_data = load_json(trunc_path)
            if args.max_examples is not None:
                trunc_data = trunc_data[: args.max_examples]

            out_items: List[Dict[str, Any]] = []
            selected_count = 0

            for ex in trunc_data:
                gold = ex.get("gold_answer", None)
                idx = ex.get("idx", None)

                mode_obj = ex.get(MODE, {})
                dir_obj = mode_obj.get(DIRECTION, {})
                ratio_obj = dir_obj.get(args.ratio, {})
                if not ratio_obj:
                    continue

                responses = ratio_obj.get("response", [])
                prompt = ratio_obj.get("prompt", "")

                if isinstance(responses, str):
                    responses = [responses]
                if not responses:
                    continue

                # Correct under pass@k?
                if not any(compare_answers(pred, gold) for pred in responses[: args.k]):
                    continue

                # Get original question
                orig_q = ex.get("question", "")
                if not orig_q:
                    orig_q = extract_question_from_prompt(prompt)
                orig_q = (orig_q or "").strip()
                if not orig_q:
                    continue

                # (1) numeric edit
                seed = args.numeric_seed_base
                if idx is not None:
                    try:
                        seed = seed + int(idx) * 10007
                    except Exception:
                        pass

                numedit_q = ""
                try:
                    numr = single_number_edit(
                        orig_q,
                        seed=seed,
                        preferred=args.numeric_preferred,
                    )
                    numedit_q = numr.edited_question
                except Exception as e:
                    # keep empty string if failed; still record the example
                    numedit_q = ""

                # (2) paraphrase/reorder (Gemini)
                para_q = ""
                if client is not None:
                    aime_strict_math = ("aime" in dataset_base.lower())
                    para_res = paraphrase_reorder_with_gemini(
                        orig_q,
                        client,
                        language_name=LANG_NAME.get(lang_norm, lang_norm),
                        model_name=args.gemini_model,
                        max_retries=2,
                        aime_strict_math=aime_strict_math,
                    )
                    if para_res.get("ok"):
                        para_q = para_res.get("edited_question", "").strip()
                    else:
                        para_q = ""

                    if args.throttle > 0:
                        time.sleep(args.throttle)

                out_items.append(
                    {
                        "idx": idx,
                        "gold_answer": gold,
                        "orig_question": orig_q,
                        "numedit": numedit_q,
                        "paraphrase": para_q,
                    }
                )

                selected_count += 1
                if args.max_selected is not None and selected_count >= args.max_selected:
                    break

            out_dir = output_root / dataset_base / model_base
            out_path = out_dir / f"{lang_suffix}_counterfactuals_passat{args.k}.json"
            
            
            result = {
                "meta": {
                    "dataset": dataset_name,
                    "model": args.model_name,
                    "language": lang_norm,
                    "mode": MODE,
                    "direction": DIRECTION,
                    "ratio": args.ratio,
                    "pass_k": args.k,
                    "num_total_selected": len(out_items)
                },
                "items": out_items,
            }

            save_json(result, out_path)
            print(
                f"[OK] Saved {len(out_items)} items "
                f"(correct under pass@{args.k}) → {out_path}"
            )

        print(f"########## DONE DATASET: {dataset_name} ##########\n")

    print("All done.")


if __name__ == "__main__":
    main()
