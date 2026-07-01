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
    build_agent_messages_hierarchical_text_mas,
    build_agent_messages_sequential_text_mas,
    get_assistant_think_prefill,
)
from utils import auto_device, set_seed
from helper import extract_think_sentences, normalize_lang_key
from run_latent_mas_agent_similarity import (
    compute_logitlens_for_trace,
    cosine_by_step_layer,
    jsonable_summary,
    latent_reasoning_emergence,
)

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"



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
        method="text_mas",
        model_name=args.model_name,
        task="mgsm",
        mgsm_lang=lang,
        prompt=args.prompt,
        text_mas_context_length=args.text_mas_context_length,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )


def hidden_trace_for_text(
    model: ModelWrapper,
    prompt: str,
    response: str,
    lang: str,
    max_steps: int,
) -> torch.Tensor:
    think_sentences = extract_think_sentences(response, lang)
    if not think_sentences:
        think_sentences = [response]
    if max_steps > 0 and len(think_sentences) > max_steps:
        idxs = np.linspace(0, len(think_sentences) - 1, max_steps).round().astype(int).tolist()
        units = [think_sentences[i] for i in idxs]
    else:
        units = think_sentences

    traces = []
    running = ""
    for unit in units:
        running += unit
        text = prompt + running
        input_ids, attention_mask, _ = encode_prompts(model, [text])
        hidden, _ = model.forward_last_hidden_by_layer(input_ids, attention_mask=attention_mask)
        traces.append(hidden)

    if not traces:
        input_ids, attention_mask, _ = encode_prompts(model, [prompt + response])
        hidden, _ = model.forward_last_hidden_by_layer(input_ids, attention_mask=attention_mask)
        traces.append(hidden)

    return torch.cat([h[:, None, :, :] for h in traces], dim=1)


def collect_language_traces(model: ModelWrapper, args: argparse.Namespace, lang: str) -> Dict:
    method_args = build_args(args, lang)
    item = first_mgsm_item(lang)
    contexts = ""
    agents_out = {}

    for agent in default_agents():
        if args.prompt == "hierarchical":
            messages = build_agent_messages_hierarchical_text_mas(
                role=agent.role,
                question=item["question"],
                context=contexts,
                method="text_mas",
                args=method_args,
            )
        else:
            messages = build_agent_messages_sequential_text_mas(
                role=agent.role,
                question=item["question"],
                context=contexts,
                method="text_mas",
                args=method_args,
            )

        prompt = model.render_chat(messages, add_generation_prompt=True)
        think_prefill = get_assistant_think_prefill(method_args)
        if think_prefill:
            prompt = f"{prompt}{think_prefill}"
        input_ids, attention_mask, tokens_batch = encode_prompts(model, [prompt])

        generated_texts, _ = model.generate_text_batch(
            input_ids,
            attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        response = generated_texts[0].strip()
        if think_prefill:
            response = f"{think_prefill}{response}"

        trace = hidden_trace_for_text(
            model,
            prompt,
            response[len(think_prefill):] if think_prefill and response.startswith(think_prefill) else response,
            normalize_lang_key(lang),
            args.max_trace_steps,
        )

        agents_out[agent.role] = {
            "name": agent.name,
            "prompt": prompt,
            "output": response,
            "input_tokens": tokens_batch[0],
            "hidden": trace.squeeze(0).detach().to(torch.float16).cpu().numpy(),
            "logitlens": compute_logitlens_for_trace(model, trace, item["gold"]),
        }

        if agent.role != "judger":
            contexts += f"[{agent.name}]:\n{response}\n\n"

    return {
        "lang": lang,
        "lang_norm": normalize_lang_key(lang),
        "question": item["question"],
        "gold": item["gold"],
        "agents": agents_out,
    }


def resample_steps(hidden: np.ndarray, target_steps: int) -> np.ndarray:
    if hidden.shape[0] == target_steps:
        return hidden
    idxs = np.linspace(0, hidden.shape[0] - 1, target_steps).round().astype(int)
    return hidden[idxs]


def summarize_against_english(ref: Dict, tgt: Dict, rank_threshold: int, layer_strategy: str) -> Dict:
    agent_summaries = {}
    for role, tgt_agent in tgt["agents"].items():
        ref_hidden = ref["agents"][role]["hidden"]
        tgt_hidden = tgt_agent["hidden"]
        target_steps = min(ref_hidden.shape[0], tgt_hidden.shape[0])
        ref_aligned = resample_steps(ref_hidden, target_steps)
        tgt_aligned = resample_steps(tgt_hidden, target_steps)
        sims = cosine_by_step_layer(ref_aligned, tgt_aligned)
        emergence = latent_reasoning_emergence(tgt_agent["logitlens"], rank_threshold, layer_strategy)
        agent_summaries[role] = {
            "shape": list(tgt_hidden.shape),
            "aligned_shape": list(tgt_aligned.shape),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--languages", type=str, default="bn,de,en,es,fr,ja,ru,sw,te,th,zh")
    parser.add_argument("--ref_lang", type=str, default="en")
    parser.add_argument("--prompt", choices=["sequential", "hierarchical"], default="sequential")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_trace_steps", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--text_mas_context_length", type=int, default=-1)
    parser.add_argument("--emergence_rank_threshold", type=int, default=10)
    parser.add_argument(
        "--emergence_layer_strategy",
        choices=["best_layer", "final_layer"],
        default="final_layer",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="multilingual-latent-reasoning/results_text_mas_agents")
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

    with (out_dir / "text_agent_traces.pkl").open("wb") as f:
        pickle.dump(
            {
                "meta": vars(args),
                "traces": traces,
                "summaries": summaries,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    json_summary = {
        "meta": {
            **vars(args),
            "cosine_definition": (
                "Mean cosine similarity to English across agents, sampled text reasoning steps, and layers. "
                "Agent traces with different lengths are linearly resampled to a shared step count."
            ),
            "reasoning_score_definition": (
                "Mean across agents of 1 - emergence_step / step_count, where emergence_step is the first "
                "sampled text-reasoning step whose selected-layer gold first-token rank is <= threshold."
            ),
        },
        "summaries": {lang: jsonable_summary(summary) for lang, summary in summaries.items()},
    }
    with (out_dir / "text_agent_similarity_summary.json").open("w", encoding="utf-8") as f:
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
