import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from helper import (
    extract_think_sentences,
    normalize_lang_key,
    get_answer_prefix,
)

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Lineesha Kamana"
__email__ = "lpk5305@psu.edu, hthakur@uccs.edu"
__status__ = "prototype"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_gold_first_token_id(tokenizer, gold_answer) -> Optional[int]:
    if gold_answer is None:
        return None
    text = str(gold_answer).strip()
    if not text:
        return None
    enc = tokenizer(text, add_special_tokens=False, return_tensors=None)
    ids = enc["input_ids"]
    if not ids:
        return None
    return ids[0]


def compute_logprob_and_rank_all_layers(
    hidden_states,
    lm_head,
    gold_token_id: int,
    last_token_index: int,
    final_norm=None,
):
    n_hs = len(hidden_states)
    per_layer = {}

    for layer_idx in range(n_hs):
        h = hidden_states[layer_idx][0, last_token_index, :]

        if final_norm is not None and layer_idx != n_hs - 1:
            h = final_norm(h)

        logits = lm_head(h)
        logits = logits.to(torch.float32)
        log_probs = torch.log_softmax(logits, dim=-1)

        lp = float(log_probs[gold_token_id].item())

        logits_cpu = logits.detach().cpu()
        target_logit = logits_cpu[gold_token_id]
        rank = int((logits_cpu > target_logit).sum().item()) + 1

        per_layer[f"layer_{layer_idx}"] = {
            "logprob_gold_first": lp,
            "rank_gold_first": float(rank),
        }

    return per_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--model_name", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--dataset_name", type=str,
                        default="juletxara/mgsm")
    parser.add_argument("--languages", type=str,
                        default="EN")
    parser.add_argument("--mode", type=str,
                        choices=["normal", "hack", "both"],
                        default="both")
    parser.add_argument("--input_results_root", type=str,
                        default="results")
    parser.add_argument("--output_results_root", type=str,
                        default="logitlens")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    dataset_base = args.dataset_name.split("/")[-1]
    model_base = args.model_name.split("/")[-1]
    lang_list = [x.strip() for x in args.languages.split(",") if x.strip()]
    modes_to_use = ["normal", "hack"] if args.mode == "both" else [args.mode]
    is_aime = "aime" in args.dataset_name.lower()

    print(f"Loading tokenizer & model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        cache_dir=args.cache_dir
    )
    model.eval()

    lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
    final_norm = model.model.norm

    input_root = Path(args.input_results_root)
    output_root = Path(args.output_results_root)

    # -------------------------------
    # LOOP OVER LANGUAGES
    # -------------------------------
    for lang in lang_list:
        lang_norm = normalize_lang_key(lang)
        lang_suffix = lang_norm.lower()

        # ==== SKIP IF RESULT ALREADY EXISTS ====
        out_dir = output_root / dataset_base / model_base
        out_path = out_dir / f"{lang_suffix}_logitlens_suffix_alllayers.json"

        if out_path.exists():
            print(f"[SKIP] Logit-lens output already exists for {lang_norm}: {out_path}")
            continue

        # Load original full-CoT
        in_path = input_root / dataset_base / model_base / f"{lang_suffix}_result.json"
        if not in_path.exists():
            print(f"[WARN] No full-CoT results for {lang_norm}: {in_path}")
            continue

        print(f"\n=== Language {lang_norm} ===")
        full_results = load_json(in_path)
        if args.max_examples is not None:
            full_results = full_results[: args.max_examples]

        all_stats = []
        answer_prefix = get_answer_prefix(lang_norm)

        for ex_idx, entry in enumerate(full_results):
            q = entry.get("question", "")
            gold = entry.get("gold_answer", None)
            idx_val = int(entry.get("idx", ex_idx))

            gold_first_id = get_gold_first_token_id(tokenizer, gold)
            if gold_first_id is None:
                continue

            ex_stat = {
                "idx": idx_val,
                "question": q,
                "gold_answer": gold,
                "per_mode": {},
            }

            for kind in modes_to_use:
                obj = entry.get(kind, {})
                if not obj:
                    continue

                base_prompt = obj["prompt"]
                base_response = obj["response"]

                think_sentences = extract_think_sentences(base_response, lang_norm)
                if not think_sentences:
                    continue

                num_steps = len(think_sentences)
                mode_stats = []

                if not is_aime:
                    # MGSM: every step
                    step_ratios = [(i + 1) / num_steps for i in range(num_steps)]
                    step_indices = list(range(num_steps))
                else:
                    # AIME: percentages 0%, 5%, … 100%
                    step_ratios = [i * 0.05 for i in range(21)]
                    step_indices = [
                        max(0, min(num_steps, int(round(r * num_steps)))) - 1 for r in step_ratios
                    ]

                # Evaluate each chosen step
                for r, step_id in zip(step_ratios, step_indices):
                    if step_id >= 0:
                        partial_think = "".join(think_sentences[: step_id + 1])
                        step_text = think_sentences[step_id]
                    else:
                        partial_think = ""
                        step_text = ""

                    context_text = (
                        base_prompt
                        + partial_think
                        + "\n</think>\n\n"
                        + answer_prefix
                    )

                    enc = tokenizer(
                        context_text, return_tensors="pt", add_special_tokens=False
                    )
                    input_ids = enc["input_ids"].to(args.device)
                    attention_mask = enc["attention_mask"].to(args.device)
                    seq_len = int(attention_mask.sum(1)[0].item())
                    last_token_index = seq_len - 1

                    with torch.no_grad():
                        outputs = model(
                            input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                        )

                    per_layer = compute_logprob_and_rank_all_layers(
                        outputs.hidden_states,
                        lm_head,
                        gold_first_id,
                        last_token_index,
                        final_norm,
                    )

                    mode_stats.append(
                        {
                            "step_index": step_id,
                            "num_steps": num_steps,
                            "step_ratio": r,
                            "step_text": step_text,
                            "per_layer": per_layer,
                        }
                    )

                if mode_stats:
                    ex_stat["per_mode"][kind] = mode_stats

            if ex_stat["per_mode"]:
                all_stats.append(ex_stat)

        save_json(all_stats, out_path)
        print(f"[OK] Saved logit-lens dynamics for {lang_norm} → {out_path}")

    print("\nAll languages completed.")


if __name__ == "__main__":
    main()


"""

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 \
python run_logitlens_dynamics.py \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --dataset_name shanchen/aime_2024_multilingual \
  --languages EN \
  --mode both \

"""