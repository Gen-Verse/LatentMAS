#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


# ---------------------------------------------------------------------
# Language normalization + answer prefixes
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


def normalize_lang_key(lang: str) -> str:
    """
    Normalize language key to canonical upper-case form, e.g. 'en' -> 'EN'.
    """
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

    Returns:
        (numeric_value_or_None, canonical_string)

    Steps:
      1) Try number immediately before a '}' (e.g. "3}")
      2) Fallback: first number with optional commas/decimal
      3) Remove commas before float()
      4) Canonicalize string: strip leading zeros, remove commas, and
         represent integers without '.0'
    """
    if ans is None:
        return None, ""

    s = str(ans)

    # 1) number just before a '}' (allow commas/decimals)
    m = re.search(r"([-+]?\d[\d,]*(?:\.\d+)?)(?=\s*})", s)
    if not m:
        # 2) fallback: first number anywhere
        m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", s)

    if not m:
        return None, ""

    num_str_raw = m.group(0)        # e.g. "033", "57,500", "12.0"
    num_str_clean = num_str_raw.replace(",", "")

    try:
        val = float(num_str_clean)
    except ValueError:
        return None, ""

    # 3) Canonical string:
    #    - integers → "33"
    #    - non-integers → shortest decimal form
    if abs(val - int(val)) < 1e-12:
        canon = str(int(val))
    else:
        canon = str(val).rstrip("0").rstrip(".")

    return val, canon


def compare_answers(pred_raw: Optional[str], gold_raw) -> bool:
    """
    Compare predicted answer string and gold answer.

    1) numeric comparison if both parse
    2) otherwise normalized string comparison
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


def gold_in_text(gold_raw, text: str) -> bool:
    """
    Check whether the gold numeric answer appears *anywhere* in the reasoning
    text, numerically.

    Strategy:
      1) Parse gold_raw into a numeric value (via normalize_answer).
      2) Scan the text for all numeric substrings (allowing commas, decimals).
      3) For each found number, normalize by removing commas and compare as float.

    This handles:
      - gold "033" vs "33" in text
      - gold 57500 vs "57,500" in text
    """
    gold_num, gold_str = normalize_answer(gold_raw)
    if gold_num is None:
        return False

    # Find all numbers like 57,500, 33, -12.5 etc.
    for m in re.finditer(r"[-+]?\d[\d,]*(?:\.\d+)?", text):
        token = m.group(0)
        token_clean = token.replace(",", "")
        try:
            val = float(token_clean)
        except ValueError:
            continue

        if abs(val - gold_num) < 1e-6:
            return True

    return False


# --------------------- think extraction from prompt --------------------- #

def extract_think_from_prompt(prompt: str) -> str:
    """
    Given a prompt of the form
        ... <｜Assistant｜><think>\n ...thinking... \n</think>\n\nThe answer is: ...
    return the substring between <think> and </think>.

    If tags are missing, returns "".
    """
    if not prompt:
        return ""

    start_tag = "<think>"
    end_tag = "</think>"

    start = prompt.find(start_tag)
    if start == -1:
        return ""
    start += len(start_tag)

    end = prompt.find(end_tag, start)
    if end == -1:
        end = len(prompt)

    return prompt[start:end]


# --------------------- main aggregation --------------------- #

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
        default="accuracy_trunc",
        help="Where to save aggregated accuracy JSON.",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="1,5,10",
        help="Comma separated pass@k values, e.g. '1,5,10'.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optionally limit number of examples per dataset/lang (debug).",
    )

    args = parser.parse_args()

    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    lang_list_raw = [l.strip() for l in args.languages.split(",") if l.strip()]
    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]

    input_trunc_root = Path(args.input_trunc_root)
    output_root = Path(args.output_root)

    model_base = args.model_name.split("/")[-1]

    # how many debug examples to print per dataset/lang
    DEBUG_LIMIT = 10

    print("=== Settings ===")
    print(f"Model:      {args.model_name}")
    print(f"Datasets:   {dataset_list}")
    print(f"Languages:  {lang_list_raw}")
    print(f"Pass@k:     {ks}")
    print("================\n")

    for dataset_name in dataset_list:
        dataset_base = dataset_name.split("/")[-1]
        print(f"########## DATASET: {dataset_name} ##########")

        for lang in lang_list_raw:
            lang_norm = normalize_lang_key(lang)
            lang_suffix = lang_norm.lower()

            trunc_path = (
                input_trunc_root / dataset_base / model_base / f"{lang_suffix}_trunc.json"
            )

            if not trunc_path.exists():
                print(f"[SKIP] No truncation file for {dataset_base}, {lang_norm}: {trunc_path}")
                continue

            print(f"\n--- {dataset_base} | {lang_norm} ---")
            print(f"Loading trunc file: {trunc_path}")
            trunc_data = load_json(trunc_path)

            if args.max_examples is not None:
                trunc_data = trunc_data[: args.max_examples]

            if not trunc_data:
                print(f"[WARN] Empty truncation data for {dataset_base}, {lang_norm}")
                continue

            # Probe modes/directions/ratios from first example
            example0 = trunc_data[0]
            modes: List[str] = []
            for m in ["normal", "hack"]:
                if m in example0:
                    modes.append(m)
            if not modes:
                print(f"[WARN] No 'normal'/'hack' in truncation data for {dataset_base}, {lang_norm}")
                continue

            directions = set()
            ratios = set()
            for mode in modes:
                mode_obj = example0.get(mode, {})
                for direction_name, dir_obj in mode_obj.items():
                    directions.add(direction_name)
                    for ratio_key in dir_obj.keys():
                        ratios.add(ratio_key)

            directions = sorted(list(directions))
            ratios = sorted(ratios, key=lambda x: int(x.rstrip("%")))

            print(f"Modes:      {modes}")
            print(f"Directions: {directions}")
            print(f"Ratios:     {ratios}")

            # Prepare result structure for this dataset+lang
            acc_result: Dict[str, Any] = {
                "dataset": dataset_name,
                "model": args.model_name,
                "language": lang_norm,
                "ratios": ratios,
                "directions": directions,
                "modes": modes,
                "ks": ks,
                "metrics": {}  # metrics[k][mode][direction][ratio] = {...}
            }

            for k in ks:
                key_k = f"pass_at_{k}"
                acc_result["metrics"][key_k] = {}
                for mode in modes:
                    acc_result["metrics"][key_k][mode] = {}
                    for direction in directions:
                        acc_result["metrics"][key_k][mode][direction] = {}
                        for ratio_key in ratios:
                            acc_result["metrics"][key_k][mode][direction][ratio_key] = {
                                "pass": 0.0,
                                "num_total": 0,
                                "num_correct": 0,
                                "num_correct_with_gold_in_steps": 0,
                                "fraction_gold_in_steps_among_correct": None,
                            }

            # counter for debug prints for this dataset+lang
            debug_count = 0

            # Aggregate over examples
            for trunc_ex in trunc_data:
                idx_val = int(trunc_ex.get("idx", 0))
                gold = trunc_ex.get("gold_answer", None)

                for mode in modes:
                    mode_trunc = trunc_ex.get(mode, {})
                    if not mode_trunc:
                        continue

                    for direction in directions:
                        dir_obj = mode_trunc.get(direction, {})
                        if not dir_obj:
                            continue

                        for ratio_key in ratios:
                            ratio_data = dir_obj.get(ratio_key, None)
                            if not ratio_data:
                                continue

                            prompt = ratio_data.get("prompt", "")
                            responses = ratio_data.get("response", [])

                            if isinstance(responses, str):
                                responses = [responses]

                            if not responses:
                                continue

                            # Extract truncated reasoning from the *prompt* itself
                            think_text = extract_think_from_prompt(prompt)

                            # For each k, count total
                            for k in ks:
                                key_k = f"pass_at_{k}"
                                m_entry = acc_result["metrics"][key_k][mode][direction][ratio_key]
                                m_entry["num_total"] += 1

                            # Now check correctness and whether gold is already in think_text
                            for k in ks:
                                key_k = f"pass_at_{k}"
                                m_entry = acc_result["metrics"][key_k][mode][direction][ratio_key]

                                correct = False
                                for pred in responses[:k]:
                                    if compare_answers(pred, gold):
                                        correct = True
                                        break

                                if not correct:
                                    continue

                                m_entry["num_correct"] += 1

                                gold_in_steps = gold_in_text(gold, think_text)
                                if gold_in_steps:
                                    m_entry["num_correct_with_gold_in_steps"] += 1
                                # else:
                                #     # DEBUG: correct but gold not in think_text
                                #     # print only for ratio=100% to inspect why
                                #     if (
                                #         ratio_key == "100%"
                                #         and debug_count < DEBUG_LIMIT
                                #     ):
                                #         debug_count += 1
                                #         print("\n[DEBUG] Correct answer but gold NOT found in 100% think_text")
                                #         print(f"  dataset   : {dataset_base}")
                                #         print(f"  language  : {lang_norm}")
                                #         print(f"  mode      : {mode}")
                                #         print(f"  direction : {direction}")
                                #         print(f"  idx       : {idx_val}")
                                #         print(f"  gold      : {gold}")
                                #         print("  last 300 chars of think_text:")
                                #         print("----------------------------------------")
                                #         print(think_text[-300:])
                                #         print("----------------------------------------")
                                #         print(f"  responses (first {k}): {responses[:k]}")
                                #         print("----------------------------------------")

            # Finalize metrics
            for k in ks:
                key_k = f"pass_at_{k}"
                for mode in modes:
                    for direction in directions:
                        for ratio_key in ratios:
                            m_entry = acc_result["metrics"][key_k][mode][direction][ratio_key]
                            total = m_entry["num_total"]
                            num_correct = m_entry["num_correct"]
                            num_correct_gold = m_entry["num_correct_with_gold_in_steps"]

                            if total > 0:
                                m_entry["pass"] = num_correct / total
                            else:
                                m_entry["pass"] = 0.0

                            if num_correct > 0:
                                frac = num_correct_gold / num_correct
                                m_entry["fraction_gold_in_steps_among_correct"] = frac
                            else:
                                m_entry["fraction_gold_in_steps_among_correct"] = None

            # Save for this dataset+lang
            out_dir = output_root / dataset_base / model_base
            out_path = out_dir / f"{lang_suffix}_trunc_accuracy.json"
            save_json(acc_result, out_path)
            print(f"[OK] Saved truncation accuracy for {dataset_base}, {lang_norm} → {out_path}")

        print(f"########## DONE DATASET: {dataset_name} ##########\n")

    print("All datasets & languages completed.")


if __name__ == "__main__":
    main()
