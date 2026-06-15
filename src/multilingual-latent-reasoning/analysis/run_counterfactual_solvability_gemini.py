#!/usr/bin/env python
import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from memorization_test_helper import get_client

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Lineesha Kamana"
__email__ = "lpk5305@psu.edu, hthakur@uccs.edu"
__status__ = "prototype"

# -----------------------
# JSON I/O
# -----------------------
def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -----------------------
# Counterfactual file format (list OR {meta, items})
# -----------------------
def load_counterfactual_items(obj: Any) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if isinstance(obj, list):
        return None, obj
    if isinstance(obj, dict) and "items" in obj:
        return obj.get("meta", {}), obj["items"]
    raise ValueError("Unrecognized counterfactual JSON format (expected list or {meta, items}).")

# -----------------------
# Lang utils
# -----------------------
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

_LANG_NAME_MAP = {
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

def language_name(lang_norm: str) -> str:
    return _LANG_NAME_MAP.get(lang_norm, lang_norm)


_FINAL_RE = re.compile(r"(?m)^FINAL_ANSWER:\s*(.+?)\s*$")

def extract_final_answer(text: str) -> Optional[str]:
    if not text:
        return None
    m = None
    for m in _FINAL_RE.finditer(text):
        pass  # take last occurrence if multiple
    if not m:
        return None
    ans = m.group(1).strip()
    return ans if ans else None

# -----------------------
# Prompting
# -----------------------
_SOLVE_PROMPT = r"""
You are given a grade-school math word problem.

Language constraint:
- Write your solution in the SAME language as the problem.
- The problem language is: {language_name}. Do not translate.

You may write intermediate steps.
Hard requirement:
- End your response with a SINGLE final line in this exact format:
FINAL_ANSWER: <answer>

Rules for <answer>:
- Provide only the final numeric value (or a simplified number).
- Do not wrap it in LaTeX, do not add units, and do not add extra words.
- Do not output anything after the FINAL_ANSWER line.

Problem:
<<<
{problem}
>>>
""".strip()

def build_solve_prompt(problem: str, lang_name: str) -> str:
    return _SOLVE_PROMPT.format(language_name=lang_name, problem=problem)

# -----------------------
# Call Gemini with retries
# -----------------------

def gemini_generate_text(
    client,
    model: str,
    prompt: str,
    *,
    max_retries: int = 6,
    base_sleep: float = 1.0,
    jitter: float = 0.3
) -> str:
    last_text = ""
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt
            )
            last_text = getattr(resp, "text", None) or ""
            return last_text
        except Exception as e:
            sleep = base_sleep * (2 ** attempt)
            sleep = sleep * (1.0 + random.uniform(-jitter, jitter))
            print(f"[WARN] Gemini call failed (attempt {attempt+1}/{max_retries}): {e}")
            print(f"       sleeping {sleep:.2f}s ...")
            time.sleep(sleep)
    return last_text

def gemini_generate_until_parsed(
    client,
    model: str,
    prompt: str,
    *,
    max_parse_retries: int = 3,
    max_call_retries: int = 6,
    base_sleep: float = 1.0,
    jitter: float = 0.3,
    sleep_between_parse_retries: float = 0.2,
    add_reminder_on_retry: bool = True,
) -> Tuple[str, Optional[str], int]:
    """
    Call Gemini and retry if FINAL_ANSWER cannot be parsed.

    Returns:
      raw_text, parsed_answer_or_None, num_parse_attempts_used
    """
    last_raw = ""
    last_ans = None

    for parse_attempt in range(1, max_parse_retries + 1):
        # Optionally make the instruction stronger on retries
        prompt_i = prompt
        if add_reminder_on_retry and parse_attempt > 1:
            prompt_i = (
                prompt
                + "\n\nREMINDER: The LAST line must be exactly:\nFINAL_ANSWER: <answer>\n"
                + "Do not put anything after it.\n"
            )

        raw = gemini_generate_text(
            client=client,
            model=model,
            prompt=prompt_i,
            max_retries=max_call_retries,
            base_sleep=base_sleep,
            jitter=jitter,
        )
        ans = extract_final_answer(raw)

        last_raw, last_ans = raw, ans
        if ans is not None:
            return raw, ans, parse_attempt

        # parse failed -> retry
        if sleep_between_parse_retries > 0:
            time.sleep(sleep_between_parse_retries)

    return last_raw, last_ans, max_parse_retries

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name (e.g., gemini-2.0-flash, gemini-1.5-pro).",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default="juletxara/mgsm",
        help="Comma-separated list of dataset names (dir base is the suffix after '/').",
    )
    
    parser.add_argument(
        "--languages",
        type=str,
        default="EN,FR,DE,ZH,JA,RU,ES,SW,BN,TE,TH",
        # "EN,FR,DE,ZH,JA,RU,ES,SW,BN,TE,TH"
        help="Comma-separated list of language keys.",
    )

    parser.add_argument(
        "--model_name_for_paths",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        help="The LRM model used when generating counterfactuals (used for folder names), "
             "e.g., deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )

    parser.add_argument(
        "--counterfactual_root",
        type=str,
        default="./counterfactuals",
        help="Root where <lang>_counterfactuals_passat10.json are saved.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./counterfactuals_solvability_gemini",
        help="Root to write solvability outputs.",
    )

    parser.add_argument("--max_items", type=int, default=None, help="Limit items per file (debug).")
    parser.add_argument("--overwrite", action="store_true")

    # Gemini settings
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--sleep_between", type=float, default=0., help="Optional fixed sleep between requests.")
    
    parser.add_argument("--max_parse_retries", type=int, default=3,
                    help="Retry generation if FINAL_ANSWER can't be parsed.")
    parser.add_argument("--sleep_between_parse_retries", type=float, default=0.2)


    args = parser.parse_args()

    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    lang_list = [l.strip() for l in args.languages.split(",") if l.strip()]

    cf_root = Path(args.counterfactual_root)
    out_root = Path(args.output_root)

    model_base = args.model_name_for_paths.split("/")[-1]

    client = get_client()

    for dataset_name in dataset_list:
        dataset_base = dataset_name.split("/")[-1]
        print(f"\n########## DATASET: {dataset_base} ##########")

        for lang in lang_list:
            lang_norm = normalize_lang_key(lang)
            lang_suffix = lang_norm.lower()
            lang_fullname = language_name(lang_norm)

            cf_path = cf_root / dataset_base / model_base / f"{lang_suffix}_counterfactuals_passat10.json"
            if not cf_path.exists():
                print(f"[SKIP] Missing counterfactual file: {cf_path}")
                continue

            out_dir = out_root / dataset_base / model_base
            out_path = out_dir / f"{lang_suffix}_gemini_solvability.json"
            if out_path.exists() and not args.overwrite:
                print(f"[SKIP] Output exists: {out_path}")
                continue

            meta_cf, items_cf = load_counterfactual_items(load_json(cf_path))
            if args.max_items is not None:
                items_cf = items_cf[: args.max_items]

            # Filter: require both numedit and paraphrase non-empty
            filtered: List[Dict[str, Any]] = []
            for it in items_cf:
                numedit_q = (it.get("numedit") or "").strip()
                para_q = (it.get("paraphrase") or "").strip()
                if not numedit_q or not para_q:
                    continue
                filtered.append(it)

            print(f"[{dataset_base} | {lang_norm}] Loaded {len(items_cf)} items; kept {len(filtered)} with both edits.")

            results: List[Dict[str, Any]] = []

            # Evaluate each item with 3 prompts
            for j, it in enumerate(filtered):
                idx = it.get("idx", None)
                try:
                    idx_int = int(idx)
                except Exception:
                    continue

                orig_q = (it.get("orig_question") or "").strip()
                numedit_q = (it.get("numedit") or "").strip()
                para_q = (it.get("paraphrase") or "").strip()

                if not orig_q:
                    # In your pipeline, orig_question should exist; but guard anyway
                    continue

                record = {
                    "idx": idx_int,
                    "gold_answer": it.get("gold_answer", None),
                    "questions": {
                        "orig": orig_q,
                        "numedit": numedit_q,
                        "paraphrase": para_q,
                    },
                    "gemini": {
                        "orig": {"raw": None, "final_answer": None, "parsed_ok": False},
                        "numedit": {"raw": None, "final_answer": None, "parsed_ok": False},
                        "paraphrase": {"raw": None, "final_answer": None, "parsed_ok": False},
                    },
                }

                for key, qtext in [("orig", orig_q), ("numedit", numedit_q), ("paraphrase", para_q)]:
                    prompt = build_solve_prompt(qtext, lang_fullname)
                    
                    
                    raw, ans, parse_tries = gemini_generate_until_parsed(
                        client=client,
                        model=args.gemini_model,
                        prompt=prompt,
                        max_parse_retries=args.max_parse_retries,
                        max_call_retries=args.max_retries,
                        sleep_between_parse_retries=args.sleep_between_parse_retries,
                    )

                    record["gemini"][key]["raw"] = raw
                    record["gemini"][key]["final_answer"] = ans
                    record["gemini"][key]["parsed_ok"] = (ans is not None)
                    record["gemini"][key]["num_parse_tries"] = parse_tries


                    if args.sleep_between > 0:
                        time.sleep(args.sleep_between)

                results.append(record)

                if (j + 1) % 10 == 0:
                    print(f"  progress: {j+1}/{len(filtered)}")

            out_obj = {
                "meta": {
                    "dataset": dataset_name,
                    "dataset_base": dataset_base,
                    "model_base_for_paths": model_base,
                    "language": lang_norm,
                    "language_name": lang_fullname,
                    "gemini_model": args.gemini_model,
                    "num_items_loaded": len(items_cf),
                    "num_items_used": len(results),
                    "filter": "require numedit != '' AND paraphrase != ''",
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                },
                "counterfactual_meta": meta_cf,
                "items": results,
            }

            save_json(out_obj, out_path)
            print(f"[OK] Saved → {out_path}")

    print("\nAll done.")

if __name__ == "__main__":
    main()
