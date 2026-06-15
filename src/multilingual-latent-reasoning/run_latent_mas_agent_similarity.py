import argparse
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]  # repo root (src/<pkg>/<file>.py)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import load_mgsm
from methods import default_agents
from models import ModelWrapper
from prompts import (
    build_agent_message_hierarchical_latent_mas,
    build_agent_message_sequential_latent_mas,
)
from utils import auto_device, set_seed
from helper import normalize_lang_key

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Lineesha Kamana"
__email__ = "lpk5305@psu.edu, hthakur@uccs.edu"
__status__ = "prototype"


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), eps))


def cosine_by_step_layer(ref: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    if ref.shape != tgt.shape:
        raise ValueError(f"Shape mismatch: ref={ref.shape}, tgt={tgt.shape}")
    sims = np.zeros(ref.shape[:2], dtype=np.float32)
    for step_idx in range(ref.shape[0]):
        for layer_idx in range(ref.shape[1]):
            sims[step_idx, layer_idx] = cosine_sim(ref[step_idx, layer_idx], tgt[step_idx, layer_idx])
    return sims


def get_gold_first_token_id(model: ModelWrapper, gold: str) -> int:
    enc = model.tokenizer(str(gold).strip(), add_special_tokens=False, return_tensors=None)
    ids = enc["input_ids"]
    if not ids:
        raise ValueError(f"Could not tokenize gold answer: {gold!r}")
    return int(ids[0])


def compute_logitlens_for_trace(model: ModelWrapper, trace: torch.Tensor, gold: str) -> Dict:
    gold_id = get_gold_first_token_id(model, gold)
    lm_head = model.model.lm_head if hasattr(model.model, "lm_head") else model.model.get_output_embeddings()
    step_count = trace.shape[1]
    layer_count = trace.shape[2]
    logprob = np.zeros((step_count, layer_count), dtype=np.float32)
    rank = np.zeros((step_count, layer_count), dtype=np.float32)

    for step_idx in range(step_count):
        for layer_idx in range(layer_count):
            h = trace[0, step_idx, layer_idx, :].to(model.device)
            logits = lm_head(h).to(torch.float32)
            log_probs = torch.log_softmax(logits, dim=-1)
            target_logit = logits[gold_id]
            logprob[step_idx, layer_idx] = float(log_probs[gold_id].item())
            rank[step_idx, layer_idx] = float((logits > target_logit).sum().item() + 1)

    return {
        "gold_first_token_id": gold_id,
        "gold_first_token": model.tokenizer.decode([gold_id]),
        "logprob_gold_first": logprob,
        "rank_gold_first": rank,
    }


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


def first_mgsm_item(lang: str) -> Dict:
    return next(iter(load_mgsm(split="test", lang=lang)))


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


def collect_language_traces(
    model: ModelWrapper,
    args: argparse.Namespace,
    lang: str,
) -> Dict:
    method_args = build_args(args, lang)
    item = first_mgsm_item(lang)
    past_kv = None
    agents_out = {}

    for agent in default_agents():
        if args.prompt == "hierarchical":
            messages = build_agent_message_hierarchical_latent_mas(
                role=agent.role,
                question=item["question"],
                context="",
                method="latent_mas",
                args=method_args,
            )
        else:
            messages = build_agent_message_sequential_latent_mas(
                role=agent.role,
                question=item["question"],
                context="",
                method="latent_mas",
                args=method_args,
            )

        prompt = model.render_chat(messages, add_generation_prompt=True)
        input_ids, attention_mask, tokens_batch = encode_prompts(model, [prompt])

        if agent.role == "judger":
            hidden, _ = model.forward_last_hidden_by_layer(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_kv if args.latent_steps > 0 else None,
            )
            trace = hidden[:, None, :, :]
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

    return {
        "lang": lang,
        "lang_norm": normalize_lang_key(lang),
        "question": item["question"],
        "gold": item["gold"],
        "agents": agents_out,
    }


def latent_reasoning_emergence(logitlens: Dict, rank_threshold: int, layer_strategy: str) -> Dict:
    ranks = logitlens["rank_gold_first"]
    if layer_strategy == "final_layer":
        step_ranks = ranks[:, -1]
    elif layer_strategy == "best_layer":
        # Skip embedding layer 0. Logit-lens projections from raw embeddings can
        # look spuriously good and are not a meaningful reasoning layer.
        layer_ranks = ranks[:, 1:] if ranks.shape[1] > 1 else ranks
        step_ranks = layer_ranks.min(axis=1)
    else:
        raise ValueError(f"Unsupported layer_strategy: {layer_strategy}")

    emerged = np.where(step_ranks <= rank_threshold)[0]

    if len(emerged) == 0:
        return {
            "emergence_step": None,
            "latent_reasoning_score": 0.0,
            "rank_threshold": rank_threshold,
            "layer_strategy": layer_strategy,
        }

    emergence_step = int(emerged[0])
    step_count = max(int(len(step_ranks)), 1)
    score = 1.0 - (float(emergence_step) / float(step_count))
    return {
        "emergence_step": emergence_step,
        "latent_reasoning_score": float(score),
        "rank_threshold": rank_threshold,
        "layer_strategy": layer_strategy,
    }


def summarize_against_english(ref: Dict, tgt: Dict, rank_threshold: int, layer_strategy: str) -> Dict:
    agent_summaries = {}
    for role, tgt_agent in tgt["agents"].items():
        ref_hidden = ref["agents"][role]["hidden"]
        tgt_hidden = tgt_agent["hidden"]
        sims = cosine_by_step_layer(ref_hidden, tgt_hidden)
        emergence = latent_reasoning_emergence(tgt_agent["logitlens"], rank_threshold, layer_strategy)
        agent_summaries[role] = {
            "shape": list(tgt_hidden.shape),
            "mean_cosine": float(sims.mean()),
            "last_layer_mean_cosine": float(sims[:, -1].mean()),
            "final_step_mean_cosine": float(sims[-1, :].mean()),
            "final_step_last_layer_cosine": float(sims[-1, -1]),
            "cosine_by_step_layer": sims,
            "logitlens": tgt_agent["logitlens"],
            "emergence": emergence,
        }
    cosine_values = [v["mean_cosine"] for v in agent_summaries.values()]
    reasoning_values = [v["emergence"]["latent_reasoning_score"] for v in agent_summaries.values()]
    return {
        "lang": tgt["lang"],
        "lang_norm": tgt["lang_norm"],
        "mean_cosine_to_english": float(np.mean(cosine_values)),
        "latent_reasoning_score": float(np.mean(reasoning_values)),
        "agents": agent_summaries,
    }


def jsonable_summary(summary: Dict) -> Dict:
    out = {
        "lang": summary["lang"],
        "lang_norm": summary["lang_norm"],
        "mean_cosine_to_english": summary["mean_cosine_to_english"],
        "latent_reasoning_score": summary["latent_reasoning_score"],
        "agents": {},
    }
    for role, data in summary["agents"].items():
        out["agents"][role] = {
            "shape": data["shape"],
            "mean_cosine": data["mean_cosine"],
            "last_layer_mean_cosine": data["last_layer_mean_cosine"],
            "final_step_mean_cosine": data["final_step_mean_cosine"],
            "final_step_last_layer_cosine": data["final_step_last_layer_cosine"],
            "final_step_last_layer_gold_logprob": float(data["logitlens"]["logprob_gold_first"][-1, -1]),
            "final_step_last_layer_gold_rank": float(data["logitlens"]["rank_gold_first"][-1, -1]),
            "best_gold_rank": float(data["logitlens"]["rank_gold_first"].min()),
            "best_gold_logprob": float(data["logitlens"]["logprob_gold_first"].max()),
            "emergence_step": data["emergence"]["emergence_step"],
            "latent_reasoning_score": data["emergence"]["latent_reasoning_score"],
            "rank_threshold": data["emergence"]["rank_threshold"],
            "emergence_layer_strategy": data["emergence"]["layer_strategy"],
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--languages", type=str, default="bn,de,en,es,fr,ja,ru,sw,te,th,zh")
    parser.add_argument("--ref_lang", type=str, default="en")
    parser.add_argument("--prompt", choices=["sequential", "hierarchical"], default="sequential")
    parser.add_argument("--latent_steps", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--emergence_rank_threshold", type=int, default=10)
    parser.add_argument(
        "--emergence_layer_strategy",
        choices=["best_layer", "final_layer"],
        default="best_layer",
        help="Use the best logit-lens layer per latent step, or only the final layer.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="multilingual-latent-reasoning/results_latent_mas_agents")
    args = parser.parse_args()

    set_seed(args.seed)
    model_args = build_args(args, args.ref_lang)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=False, args=model_args)

    langs = [x.strip().lower() for x in args.languages.split(",") if x.strip()]
    if args.ref_lang.lower() not in langs:
        langs = [args.ref_lang.lower()] + langs

    traces = {}
    for lang in langs:
        print(f"=== collecting {lang} ===")
        traces[lang] = collect_language_traces(model, args, lang)

    ref = traces[args.ref_lang.lower()]
    summaries = {}
    for lang, trace in traces.items():
        summaries[lang] = summarize_against_english(
            ref,
            trace,
            args.emergence_rank_threshold,
            args.emergence_layer_strategy,
        )

    out_dir = Path(args.out_dir) / args.model_name.split("/")[-1] / f"mgsm_first_{args.prompt}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "latent_agent_traces.pkl").open("wb") as f:
        pickle.dump(
            {
                "meta": {
                    "model": args.model_name,
                    "prompt": args.prompt,
                    "latent_steps": args.latent_steps,
                    "ref_lang": args.ref_lang.lower(),
                    "languages": langs,
                },
                "traces": traces,
                "summaries": summaries,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    json_summary = {
        "meta": {
            "model": args.model_name,
            "prompt": args.prompt,
            "latent_steps": args.latent_steps,
            "ref_lang": args.ref_lang.lower(),
            "languages": langs,
            "cosine_definition": "mean cosine similarity to English across agents, latent steps, and layers",
            "latent_reasoning_score_definition": (
                "Mean across agents of 1 - emergence_step / max_step, where emergence_step is the first "
                "latent step whose final-layer gold first-token rank is <= emergence_rank_threshold. "
                "If the gold token never emerges, that agent score is 0."
            ),
            "emergence_rank_threshold": args.emergence_rank_threshold,
            "emergence_layer_strategy": args.emergence_layer_strategy,
        },
        "summaries": {lang: jsonable_summary(summary) for lang, summary in summaries.items()},
    }
    with (out_dir / "latent_agent_similarity_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_summary, f, ensure_ascii=False, indent=2)

    for lang in langs:
        row = json_summary["summaries"][lang]
        print(
            lang,
            "latent_reasoning_score=", row["latent_reasoning_score"],
            "mean_cosine_to_english=", row["mean_cosine_to_english"],
        )
    print(f"[OK] wrote {out_dir}")


if __name__ == "__main__":
    main()
