import argparse
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import load_mgsm
from methods import default_agents
from models import ModelWrapper
from prompts import (
    build_agent_message_hierarchical_latent_mas,
    build_agent_message_sequential_latent_mas,
    get_assistant_think_prefill,
)
from utils import auto_device, extract_gsm8k_answer, normalize_answer, set_seed
from helper import normalize_lang_key
from run_latent_mas_agent_similarity import (
    compute_logitlens_for_trace,
    cosine_by_step_layer,
    latent_reasoning_emergence,
)


def encode_prompts(model: ModelWrapper, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
    encoded = model.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    tokens_batch: List[List[str]] = []
    for ids_row, mask_row in zip(input_ids, attention_mask):
        active_ids = ids_row[mask_row.bool()].tolist()
        tokens_batch.append(model.tokenizer.convert_ids_to_tokens(active_ids))
    return input_ids, attention_mask, tokens_batch


def build_args(args: argparse.Namespace, lang: str) -> SimpleNamespace:
    return SimpleNamespace(
        method="latent_mas",
        model_name=args.model_name,
        task="mgsm",
        mgsm_lang=lang,
        prompt=args.prompt,
        text_mas_context_length=-1,
        think=False,
        latent_space_realign=args.latent_space_realign,
        use_vllm=False,
        enable_prefix_caching=False,
        use_second_HF_model=False,
        device=args.device,
        device2=args.device2,
        max_new_tokens=args.max_new_tokens,
    )


def first_mgsm_items(lang: str, max_examples: int) -> List[Dict]:
    out = []
    for idx, item in enumerate(load_mgsm(split="test", lang=lang)):
        if idx >= max_examples:
            break
        item = dict(item)
        item["idx"] = idx
        out.append(item)
    return out


def build_messages(args: argparse.Namespace, method_args: SimpleNamespace, role: str, question: str):
    if args.prompt == "hierarchical":
        return build_agent_message_hierarchical_latent_mas(
            role=role,
            question=question,
            context="",
            method="latent_mas",
            args=method_args,
        )
    return build_agent_message_sequential_latent_mas(
        role=role,
        question=question,
        context="",
        method="latent_mas",
        args=method_args,
    )


def run_one_example(model: ModelWrapper, args: argparse.Namespace, lang: str, item: Dict) -> Dict:
    method_args = build_args(args, lang)
    past_kv = None
    agents_out = {}
    final_text = ""

    for agent in default_agents():
        messages = build_messages(args, method_args, agent.role, item["question"])
        prompt = model.render_chat(messages, add_generation_prompt=True)

        if agent.role == "judger":
            think_prefill = get_assistant_think_prefill(method_args)
            if think_prefill:
                prompt = f"{prompt}{think_prefill}"
        input_ids, attention_mask, tokens_batch = encode_prompts(model, [prompt])

        if agent.role == "judger":
            hidden, _ = model.forward_last_hidden_by_layer(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_kv if args.latent_steps > 0 else None,
            )
            trace = hidden[:, None, :, :]
            generated_batch, _ = model.generate_text_batch(
                input_ids,
                attention_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                past_key_values=past_kv if args.latent_steps > 0 else None,
            )
            final_text = generated_batch[0].strip()
            if think_prefill:
                final_text = f"{think_prefill}{final_text}"
        else:
            past_kv, trace = model.generate_latent_batch_with_trace(
                input_ids,
                attention_mask=attention_mask,
                latent_steps=args.latent_steps,
                past_key_values=past_kv,
            )

        agents_out[agent.role] = {
            "name": agent.name,
            "prompt": prompt,
            "input_tokens": tokens_batch[0],
            "hidden": trace.squeeze(0).detach().to(torch.float16).cpu().numpy(),
            "logitlens": compute_logitlens_for_trace(model, trace, item["gold"]),
        }
        if agent.role == "judger":
            agents_out[agent.role]["output"] = final_text

    pred = normalize_answer(extract_gsm8k_answer(final_text))
    gold = normalize_answer(item["gold"])
    return {
        "idx": int(item["idx"]),
        "lang": lang,
        "lang_norm": normalize_lang_key(lang),
        "question": item["question"],
        "gold": item["gold"],
        "prediction": pred,
        "raw_prediction": final_text,
        "correct": bool(pred == gold) if pred and gold else False,
        "agents": agents_out,
    }


def summarize_language(examples: List[Dict], rank_threshold: int, layer_strategy: str) -> Dict:
    per_agent_scores: Dict[str, List[float]] = {a.role: [] for a in default_agents()}
    per_problem = []

    for ex in examples:
        per_problem.append(
            {
                "idx": ex["idx"],
                "correct": ex["correct"],
                "prediction": ex["prediction"],
                "gold": ex["gold"],
            }
        )
        for role, agent in ex["agents"].items():
            emergence = latent_reasoning_emergence(agent["logitlens"], rank_threshold, layer_strategy)
            agent["emergence"] = emergence
            per_agent_scores[role].append(emergence["latent_reasoning_score"])

    agent_avg = {
        role: float(np.mean(vals)) if vals else 0.0
        for role, vals in per_agent_scores.items()
    }
    return {
        "accuracy": float(np.mean([ex["correct"] for ex in examples])) if examples else 0.0,
        "correct": int(sum(ex["correct"] for ex in examples)),
        "total": len(examples),
        "latent_reasoning_score": float(np.mean(list(agent_avg.values()))) if agent_avg else 0.0,
        "agent_latent_reasoning_score": agent_avg,
        "per_problem": per_problem,
    }


def cosine_between_examples(a: Dict, b: Dict) -> float:
    values = []
    for role in a["agents"].keys():
        if role not in b["agents"]:
            continue
        ah = a["agents"][role]["hidden"]
        bh = b["agents"][role]["hidden"]
        if ah.shape != bh.shape:
            continue
        values.append(float(cosine_by_step_layer(ah, bh).mean()))
    return float(np.mean(values)) if values else float("nan")


def build_all_pairs_cosine(traces: Dict[str, List[Dict]], langs: List[str]) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    matrix = np.eye(len(langs), dtype=np.float32)
    nested: Dict[str, Dict[str, float]] = {lang: {} for lang in langs}
    for i, lang_a in enumerate(langs):
        for j, lang_b in enumerate(langs):
            if j < i:
                matrix[i, j] = matrix[j, i]
                nested[lang_a][lang_b] = nested[lang_b][lang_a]
                continue
            vals = []
            examples_a = {ex["idx"]: ex for ex in traces[lang_a]}
            examples_b = {ex["idx"]: ex for ex in traces[lang_b]}
            for idx in sorted(set(examples_a) & set(examples_b)):
                vals.append(cosine_between_examples(examples_a[idx], examples_b[idx]))
            val = float(np.nanmean(vals)) if vals else float("nan")
            matrix[i, j] = val
            nested[lang_a][lang_b] = val
    return matrix, nested


def jsonable_summary(summary: Dict, cosine_nested: Dict[str, Dict[str, float]]) -> Dict:
    return {
        "languages": summary["languages"],
        "language_summary": summary["language_summary"],
        "cosine_similarity_matrix": cosine_nested,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--languages", type=str, default="bn,de,en,es,fr,ja,ru,sw,te,th,zh")
    parser.add_argument("--prompt", choices=["sequential", "hierarchical"], default="sequential")
    parser.add_argument("--latent_steps", type=int, default=3)
    parser.add_argument("--max_examples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--emergence_rank_threshold", type=int, default=1000)
    parser.add_argument(
        "--emergence_layer_strategy",
        choices=["best_layer", "final_layer"],
        default="final_layer",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="multilingual-latent-reasoning/results_latent_mas_mgsm_batch")
    args = parser.parse_args()

    set_seed(args.seed)
    model_args = build_args(args, "en")
    model = ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=model_args)

    langs = [x.strip().lower() for x in args.languages.split(",") if x.strip()]
    traces: Dict[str, List[Dict]] = {}
    for lang in langs:
        print(f"=== {lang} ===")
        traces[lang] = []
        for item in first_mgsm_items(lang, args.max_examples):
            print(f"  idx={item['idx']}")
            traces[lang].append(run_one_example(model, args, lang, item))

    language_summary = {
        lang: summarize_language(traces[lang], args.emergence_rank_threshold, args.emergence_layer_strategy)
        for lang in langs
    }
    cosine_matrix, cosine_nested = build_all_pairs_cosine(traces, langs)

    out_dir = Path(args.out_dir) / args.model_name.split("/")[-1] / f"mgsm_first{args.max_examples}_{args.prompt}"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "model": args.model_name,
            "languages": langs,
            "prompt": args.prompt,
            "latent_steps": args.latent_steps,
            "max_examples": args.max_examples,
            "emergence_rank_threshold": args.emergence_rank_threshold,
            "emergence_layer_strategy": args.emergence_layer_strategy,
            "cosine_definition": "Average across common example indices, agents, latent steps, and layers.",
        },
        "languages": langs,
        "traces": traces,
        "language_summary": language_summary,
        "cosine_similarity_matrix": cosine_matrix,
    }
    with (out_dir / "latent_mas_mgsm_batch_traces.pkl").open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary_json = {
        "meta": payload["meta"],
        **jsonable_summary(payload, cosine_nested),
    }
    with (out_dir / "latent_mas_mgsm_batch_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print("\nLanguage averages:")
    for lang in langs:
        row = language_summary[lang]
        print(
            lang,
            "acc=", row["accuracy"],
            "lrs=", row["latent_reasoning_score"],
        )
    print(f"[OK] wrote {out_dir}")


if __name__ == "__main__":
    main()
