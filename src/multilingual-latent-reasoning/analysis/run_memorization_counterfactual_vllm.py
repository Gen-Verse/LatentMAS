#!/usr/bin/env python
import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from vllm import LLM, SamplingParams

# You already have this in your project
from helper import build_truncated_think_block

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
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

def normalize_lang_key(lang: str) -> str:
    if not lang:
        return "EN"
    key = lang.strip()
    lower = key.lower()
    return _LANG_NORMALIZE.get(lower, key.upper())


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def replace_question_by_exact_match(
    prompt: str,
    orig_question: str,
    new_question: str,
    *,
    allow_fallback_ws: bool = True,
    allow_fallback_quote: bool = True,
    on_fail: str = "return",  # "return" | "raise"
) -> str:
    """
    Safest replacement: find orig_question inside prompt and replace with new_question.

    - First tries exact substring match.
    - Optionally falls back to normalized whitespace match.
    - Optionally falls back to quote-normalized match (curly vs straight quotes).
    """

    if not prompt:
        return prompt
    if not orig_question:
        if on_fail == "raise":
            raise ValueError("orig_question is empty; cannot replace safely.")
        return prompt

    # 1) exact match
    pos = prompt.find(orig_question)
    if pos != -1:
        return prompt[:pos] + new_question + prompt[pos + len(orig_question):]

    # 2) fallback: normalize quotes (curly quotes -> straight quotes)
    if allow_fallback_quote:
        def norm_quotes(x: str) -> str:
            return (x or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')

        p2 = norm_quotes(prompt)
        oq2 = norm_quotes(orig_question)
        nq2 = norm_quotes(new_question)

        pos2 = p2.find(oq2)
        if pos2 != -1:
            # apply replacement on the quote-normalized prompt, but return the modified normalized prompt
            # (this is usually fine because quotes differ only stylistically)
            return p2[:pos2] + nq2 + p2[pos2 + len(oq2):]

    # 3) fallback: whitespace-normalized approximate match
    # We locate orig_question by searching for its whitespace-normalized form within a whitespace-normalized prompt,
    # then map back by doing a conservative regex replacement in the original prompt.
    if allow_fallback_ws:
        oq_ws = _normalize_ws(orig_question)
        if oq_ws:
            # regex that matches orig_question with flexible whitespace
            pattern = re.escape(orig_question.strip())
            pattern = re.sub(r"\\\s+", r"\\s+", pattern)  # turn escaped spaces into \s+
            # Additionally, make any whitespace in orig_question match flexible whitespace
            pattern = re.sub(r"(?:\\ |\\\n|\\\t)+", r"\\s+", pattern)

            try:
                m = re.search(pattern, prompt, flags=re.DOTALL)
                if m:
                    return prompt[:m.start()] + new_question + prompt[m.end():]
            except re.error:
                pass

    if on_fail == "raise":
        raise ValueError("Could not find orig_question inside prompt; replacement aborted.")
    return prompt

# -----------------------
# Build three setups
# -----------------------
def build_prompt_orig_trace(base_prompt_with_edited_q: str, orig_full_response: str, think_lang: str) -> str:
    # Provide the original full trace (ratio=100%) and then only sample answer
    think_block = build_truncated_think_block(orig_full_response, think_lang, 1.0, reverse=False)
    return base_prompt_with_edited_q + think_block

def build_prompt_no_trace(base_prompt_with_edited_q: str, orig_full_response: str, think_lang: str) -> str:
    # Provide an empty trace via ratio=0%
    # (uses orig_full_response only as a container for truncation; content becomes empty)
    think_block = build_truncated_think_block(orig_full_response, think_lang, 0.0, reverse=False)
    return base_prompt_with_edited_q + think_block

def build_prompt_new_trace(base_prompt_with_edited_q: str) -> str:
    # No appended think block; model will generate ...</think>... and then answer itself
    return base_prompt_with_edited_q

# -----------------------
# Parse counterfactual file format (list OR {meta, items})
# -----------------------
def load_counterfactual_items(obj: Any) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if isinstance(obj, list):
        return None, obj
    if isinstance(obj, dict) and "items" in obj:
        return obj.get("meta", {}), obj["items"]
    raise ValueError("Unrecognized counterfactual JSON format (expected list or {meta, items}).")

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="juletxara/mgsm",
        help="Comma-separated list of dataset names.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="EN,FR,DE,ZH,JA,RU,ES,SW,BN,TE,TH",
        help="Comma-separated list of language keys.",
    )

    # Inputs:
    parser.add_argument(
        "--counterfactual_root",
        type=str,
        default="./counterfactuals",
        help="Root where <lang>_counterfactuals_passat10.json are saved.",
    )
    parser.add_argument(
        "--orig_results_root",
        type=str,
        default="../results",
        help="Root where original full-CoT results are stored (lang_result.json).",
    )

    # Outputs:
    parser.add_argument(
        "--output_root",
        type=str,
        default="./memorization_eval",
        help="Root to write evaluation outputs.",
    )

    # Sampling
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10, help="Number of samples per setup (vLLM n).")

    # Generation length
    parser.add_argument(
        "--max_tokens_answer_only",
        type=int,
        default=10,
        help="Max tokens for setups that only need the final answer (orig_trace/no_trace).",
    )
    parser.add_argument(
        "--max_tokens_new_trace",
        type=int,
        default=4096,
        help="Max tokens for new_trace setup (needs to include reasoning + answer).",
    )

    parser.add_argument("--cache_dir", type=str, default="../cache")
    parser.add_argument("--gpu_mem_util", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=16500)
    parser.add_argument("--max_num_seqs", type=int, default=100)

    # Debug/skip
    parser.add_argument("--max_items", type=int, default=None, help="Limit items per file (debug).")
    parser.add_argument("--overwrite", type=bool, default=False, help="Overwrite output if exists.")
    args = parser.parse_args()

    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    lang_list = [l.strip() for l in args.languages.split(",") if l.strip()]

    model_base = args.model_name.split("/")[-1]
    cf_root = Path(args.counterfactual_root)
    orig_root = Path(args.orig_results_root)
    out_root = Path(args.output_root)

    # vLLM sampling params:
    if "deepseek" in args.model_name.lower():
        sp_answer = SamplingParams(
            temperature=0.6, top_p=0.95, max_tokens=args.max_tokens_answer_only,
            seed=args.seed, n=args.k
        )
        sp_newtrace = SamplingParams(
            temperature=0.6, top_p=0.95, max_tokens=args.max_tokens_new_trace,
            seed=args.seed, n=args.k
        )
    else:
        raise ValueError("Not supported models!")

    # Load model once
    print("\nLoading model once…")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=args.gpu_mem_util,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        seed=args.seed,
        disable_custom_all_reduce=True,
        download_dir=args.cache_dir,
    )

    # You said: only forward direction + hack mode
    MODE = "hack"
    DIRECTION = "forward"

    for dataset_name in dataset_list:
        dataset_base = dataset_name.split("/")[-1]
        print(f"\n########## DATASET: {dataset_name} ##########")

        for lang in lang_list:
            lang_norm = normalize_lang_key(lang)
            lang_suffix = lang_norm.lower()
            think_lang = lang_norm  # thinking language equals prompt language

            # ---- input paths
            cf_path = cf_root / dataset_base / model_base / f"{lang_suffix}_counterfactuals_passat10.json"
            if not cf_path.exists():
                print(f"[SKIP] Missing counterfactual file: {cf_path}")
                raise ValueError("Missing counterfactual file")

            orig_path = orig_root / dataset_base / model_base / f"{lang_suffix}_result.json"
            if not orig_path.exists():
                print(f"[SKIP] Missing original full-CoT results: {orig_path}")
                raise ValueError("Missing original full-CoT results")

            # ---- output path
            out_dir = out_root / dataset_base / model_base
            out_path = out_dir / f"{lang_suffix}_mem_eval_k{args.k}.json"
            if out_path.exists() and not args.overwrite:
                print(f"[SKIP] Output exists: {out_path}")
                continue

            # ---- load
            meta_cf, items_cf = load_counterfactual_items(load_json(cf_path))
            orig_results = load_json(orig_path)

            # Map idx -> orig entry (hack prompt/response)
            idx2entry: Dict[int, Dict[str, Any]] = {}
            for e in orig_results:
                try:
                    idx2entry[int(e.get("idx"))] = e
                except Exception:
                    continue

            if args.max_items is not None:
                items_cf = items_cf[: args.max_items]

            # Build prompts in three batches to use different max_tokens
            prompts_answer: List[str] = []   # orig_trace and no_trace
            meta_answer: List[Dict[str, Any]] = []

            prompts_newtrace: List[str] = [] # new_trace
            meta_newtrace: List[Dict[str, Any]] = []

            # Keep a result skeleton
            results: List[Dict[str, Any]] = []

            def add_answer_prompt(p: str, info: Dict[str, Any]):
                prompts_answer.append(p)
                meta_answer.append(info)

            def add_newtrace_prompt(p: str, info: Dict[str, Any]):
                prompts_newtrace.append(p)
                meta_newtrace.append(info)

            for it in items_cf:
                idx = it.get("idx")
                gold = it.get("gold_answer")
                orig_q = (it.get("orig_question") or "").strip()
                numedit_q = (it.get("numedit") or "").strip()
                para_q = (it.get("paraphrase") or "").strip()
                
                
                # Require both counterfactuals to exist
                if not numedit_q or not para_q:
                    continue

                if idx is None:
                    continue
                try:
                    idx_int = int(idx)
                except Exception:
                    continue
                if idx_int not in idx2entry:
                    continue

                orig_entry = idx2entry[idx_int]
                hack_obj = orig_entry.get(MODE, {})
                base_prompt = hack_obj.get("prompt", "")
                orig_full_resp = hack_obj.get("response", "")

                if not base_prompt or not orig_full_resp:
                    continue

                rec = {
                    "idx": idx_int,
                    "gold_answer": gold,
                    "orig_question": orig_q,
                    "variants": {},  # filled below
                }

                def handle_variant(variant_name: str, edited_q: str):
                    if not edited_q:
                        return

                    # Replace the question in the original hack prompt                    
                    edited_prompt = replace_question_by_exact_match(
                                        base_prompt,
                                        orig_question=orig_q,     # from counterfactual item
                                        new_question=edited_q,
                                        on_fail="raise",          # I recommend "raise" so you don't silently run wrong prompts
                                    )

                    # Setup (1): original trace reused
                    p1 = build_prompt_orig_trace(edited_prompt, orig_full_resp, think_lang)
                    add_answer_prompt(p1, {
                        "idx": idx_int, "variant": variant_name, "setup": "orig_trace"
                    })

                    # Setup (3): no trace (empty think)
                    p3 = build_prompt_no_trace(edited_prompt, orig_full_resp, think_lang)
                    add_answer_prompt(p3, {
                        "idx": idx_int, "variant": variant_name, "setup": "no_trace"
                    })

                    # Setup (2): new trace generated
                    p2 = build_prompt_new_trace(edited_prompt)
                    add_newtrace_prompt(p2, {
                        "idx": idx_int, "variant": variant_name, "setup": "new_trace"
                    })

                    rec["variants"][variant_name] = {
                        "edited_question": edited_q,
                        "responses": {
                            "orig_trace": None,
                            "new_trace": None,
                            "no_trace": None,
                        }
                    }

                handle_variant("numedit", numedit_q)
                handle_variant("paraphrase", para_q)

                if rec["variants"]:
                    results.append(rec)

            if not results:
                print(f"[WARN] No runnable items for {dataset_base} {lang_norm}.")
                save_json({"meta": {"empty": True}, "items": []}, out_path)
                continue

            print(f"\n[{dataset_base} | {lang_norm}] Items: {len(results)}")
            print(f"Answer-only prompts (orig_trace + no_trace): {len(prompts_answer)}")
            print(f"New-trace prompts: {len(prompts_newtrace)}")

            # Run vLLM generation in two batches
            # 1) answer-only
            if prompts_answer:
                out_answer = llm.generate(prompts_answer, sp_answer, use_tqdm=True)
                for resp, info in zip(out_answer, meta_answer):
                    texts = [o.text for o in resp.outputs]
                    idx_i = info["idx"]
                    variant = info["variant"]
                    setup = info["setup"]

                    # write back
                    for rec in results:
                        if rec["idx"] == idx_i and variant in rec["variants"]:
                            rec["variants"][variant]["responses"][setup] = {
                                "prompt": resp.prompt,
                                "samples": texts,
                            }
                            break

            # 2) new-trace
            if prompts_newtrace:
                out_new = llm.generate(prompts_newtrace, sp_newtrace, use_tqdm=True)
                for resp, info in zip(out_new, meta_newtrace):
                    texts = [o.text for o in resp.outputs]
                    idx_i = info["idx"]
                    variant = info["variant"]
                    setup = info["setup"]

                    for rec in results:
                        if rec["idx"] == idx_i and variant in rec["variants"]:
                            rec["variants"][variant]["responses"][setup] = {
                                "prompt": resp.prompt,
                                "samples": texts,  # includes reasoning + answer
                            }
                            break

            out_obj = {
                "meta": {
                    "dataset": dataset_name,
                    "dataset_base": dataset_base,
                    "model": args.model_name,
                    "model_base": model_base,
                    "language": lang_norm,
                    "mode": MODE,
                    "direction": DIRECTION,
                    "counterfactual_file": str(cf_path),
                    "orig_results_file": str(orig_path),
                    "k": args.k,
                    "seed": args.seed,
                    "max_tokens_answer_only": args.max_tokens_answer_only,
                    "max_tokens_new_trace": args.max_tokens_new_trace,
                    "num_items": len(results),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                },
                "items": results,
            }

            save_json(out_obj, out_path)
            print(f"[OK] Saved → {out_path}")

    print("\nAll done.")

if __name__ == "__main__":
    main()



"""


NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=6,7 \
nohup python -u run_memorization_counterfactual_vllm.py \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --datasets juletxara/mgsm \
  --languages EN,FR,DE,ZH,JA,RU,ES,SW,BN,TE,TH \
  --cache_dir ../cache > 32b.txt 2>&1 &

"""