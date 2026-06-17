import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import load_mgsm
from helper import extract_think_sentences, normalize_lang_key
from methods import default_agents
from models import ModelWrapper
from prompts import (
    build_agent_messages_hierarchical_text_mas,
    build_agent_messages_sequential_text_mas,
    get_assistant_think_prefill,
)
from run_latent_mas_agent_similarity import (
    compute_logitlens_for_trace,
    cosine_by_step_layer,
    latent_reasoning_emergence,
)
from utils import auto_device, extract_gsm8k_answer, normalize_answer, set_seed


SHARED_STAGES = {
    "shared_after_planner": ["planner"],
    "shared_after_critic": ["planner", "critic"],
    "shared_after_refiner": ["planner", "critic", "refiner"],
    "shared_with_judger": ["planner", "critic", "refiner", "judger"],
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


def build_args(args: argparse.Namespace, lang: str) -> SimpleNamespace:
    return SimpleNamespace(
        method="text_mas",
        model_name=args.model_name,
        task="mgsm",
        mgsm_lang=lang,
        prompt=args.prompt,
        text_mas_context_length=args.text_mas_context_length,
        think=False,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        latent_space_realign=False,
        use_vllm=False,
        enable_prefix_caching=False,
        use_second_HF_model=False,
    )


def first_mgsm_items(lang: str, max_examples: int) -> List[Dict]:
    out = []
    for idx, item in enumerate(load_mgsm(split="test", lang=lang)):
        if max_examples >= 0 and idx >= max_examples:
            break
        item = dict(item)
        item["idx"] = idx
        out.append(item)
    return out


def build_messages(args: argparse.Namespace, method_args: SimpleNamespace, role: str, question: str, context: str):
    if args.prompt == "hierarchical":
        return build_agent_messages_hierarchical_text_mas(
            role=role,
            question=question,
            context=context,
            method="text_mas",
            args=method_args,
        )
    return build_agent_messages_sequential_text_mas(
        role=role,
        question=question,
        context=context,
        method="text_mas",
        args=method_args,
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
        input_ids, attention_mask, _ = encode_prompts(model, [prompt + running])
        hidden, _ = model.forward_last_hidden_by_layer(input_ids, attention_mask=attention_mask)
        traces.append(hidden)

    if not traces:
        input_ids, attention_mask, _ = encode_prompts(model, [prompt + response])
        hidden, _ = model.forward_last_hidden_by_layer(input_ids, attention_mask=attention_mask)
        traces.append(hidden)

    return torch.cat([h[:, None, :, :] for h in traces], dim=1)


def run_one_example(model: ModelWrapper, args: argparse.Namespace, lang: str, item: Dict) -> Dict:
    method_args = build_args(args, lang)
    context = ""
    agents_out = {}
    final_text = ""

    for agent in default_agents():
        messages = build_messages(args, method_args, agent.role, item["question"], context)
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

        response_for_trace = (
            response[len(think_prefill):]
            if think_prefill and response.startswith(think_prefill)
            else response
        )
        trace = hidden_trace_for_text(
            model,
            prompt,
            response_for_trace,
            normalize_lang_key(lang),
            args.max_trace_steps,
        )

        agents_out[agent.role] = {
            "name": agent.name,
            "prompt": prompt,
            "input_tokens": tokens_batch[0],
            "output": response,
            "hidden": trace.squeeze(0).detach().to(torch.float16).cpu().numpy(),
            "logitlens": compute_logitlens_for_trace(model, trace, item["gold"]),
        }

        if agent.role != "judger":
            context += f"[{agent.name}]:\n{response}\n\n"
        else:
            final_text = response

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


def resample_steps(hidden: np.ndarray, target_steps: int) -> np.ndarray:
    if hidden.shape[0] == target_steps:
        return hidden
    idxs = np.linspace(0, hidden.shape[0] - 1, target_steps).round().astype(int)
    return hidden[idxs]


def cosine_between_agents(a: Dict, b: Dict) -> float:
    ah = a["hidden"]
    bh = b["hidden"]
    target_steps = min(ah.shape[0], bh.shape[0])
    if target_steps <= 0:
        return float("nan")
    return float(cosine_by_step_layer(resample_steps(ah, target_steps), resample_steps(bh, target_steps)).mean())


def cosine_between_examples(a: Dict, b: Dict) -> float:
    values = []
    for role in a["agents"].keys():
        if role in b["agents"]:
            values.append(cosine_between_agents(a["agents"][role], b["agents"][role]))
    return float(np.nanmean(values)) if values else float("nan")


def build_all_pairs_cosine(traces: Dict[str, List[Dict]], langs: List[str]) -> Dict[str, Dict[str, float]]:
    nested: Dict[str, Dict[str, float]] = {lang: {} for lang in langs}
    by_lang = {lang: {ex["idx"]: ex for ex in traces.get(lang, [])} for lang in langs}
    for lang_a in langs:
        for lang_b in langs:
            vals = []
            for idx in sorted(set(by_lang[lang_a]) & set(by_lang[lang_b])):
                vals.append(cosine_between_examples(by_lang[lang_a][idx], by_lang[lang_b][idx]))
            nested[lang_a][lang_b] = float(np.nanmean(vals)) if vals else float("nan")
    return nested


def build_example_pair_cosines(traces: Dict[str, List[Dict]], langs: List[str]) -> Dict[int, Dict[str, Dict[str, float]]]:
    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    idxs = sorted({ex["idx"] for lang in langs for ex in traces.get(lang, [])})
    by_lang = {lang: {ex["idx"]: ex for ex in traces.get(lang, [])} for lang in langs}
    for idx in idxs:
        out[idx] = {lang: {} for lang in langs}
        for lang_a in langs:
            ex_a = by_lang[lang_a].get(idx)
            for lang_b in langs:
                ex_b = by_lang[lang_b].get(idx)
                out[idx][lang_a][lang_b] = (
                    cosine_between_examples(ex_a, ex_b)
                    if ex_a is not None and ex_b is not None
                    else float("nan")
                )
    return out


def agent_metrics(agent: Dict, rank_threshold: int, layer_strategy: str) -> Dict:
    emergence = agent.get("emergence")
    if emergence is None:
        emergence = latent_reasoning_emergence(agent["logitlens"], rank_threshold, layer_strategy)
        agent["emergence"] = emergence
    ranks = agent["logitlens"]["rank_gold_first"]
    logprobs = agent["logitlens"]["logprob_gold_first"]
    return {
        "shape": "x".join(str(x) for x in agent["hidden"].shape),
        "final_step_last_layer_gold_logprob": float(logprobs[-1, -1]),
        "final_step_last_layer_gold_rank": float(ranks[-1, -1]),
        "best_gold_rank": float(ranks.min()),
        "best_gold_logprob": float(logprobs.max()),
        "emergence_step": emergence["emergence_step"],
        "latent_reasoning_score": emergence["latent_reasoning_score"],
        "rank_threshold": emergence["rank_threshold"],
        "emergence_layer_strategy": emergence["layer_strategy"],
    }


def metric_keys() -> List[str]:
    return [
        "shape",
        "final_step_last_layer_gold_logprob",
        "final_step_last_layer_gold_rank",
        "best_gold_rank",
        "best_gold_logprob",
        "emergence_step",
        "latent_reasoning_score",
        "rank_threshold",
        "emergence_layer_strategy",
    ]


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
            metrics = agent_metrics(agent, rank_threshold, layer_strategy)
            per_agent_scores[role].append(metrics["latent_reasoning_score"])

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


def select_layer_values(matrix: np.ndarray, layer_strategy: str) -> np.ndarray:
    if layer_strategy == "final_layer":
        return matrix[:, -1]
    if layer_strategy == "best_layer":
        return matrix.min(axis=1)
    if layer_strategy == "mean_layer":
        return matrix.mean(axis=1)
    raise ValueError(f"Unsupported shared LRS layer strategy: {layer_strategy}")


def concat_shared_path(example: Dict, roles: List[str], key: str, layer_strategy: str) -> np.ndarray:
    pieces = []
    for role in roles:
        matrix = np.asarray(example["agents"][role]["logitlens"][key], dtype=np.float32)
        pieces.append(select_layer_values(matrix, layer_strategy))
    return np.concatenate(pieces, axis=0)


def shared_emergence_score(rank_path: np.ndarray, threshold: int) -> Dict:
    emerged = np.where(rank_path <= threshold)[0]
    if len(emerged) == 0:
        return {
            "shared_emergence_position": None,
            "shared_latent_reasoning_score": 0.0,
            "shared_emerged": False,
        }
    pos = int(emerged[0])
    return {
        "shared_emergence_position": pos,
        "shared_latent_reasoning_score": float(1.0 - (pos / float(len(rank_path)))),
        "shared_emerged": True,
    }


def shared_path_summary(rank_path: np.ndarray, logprob_path: np.ndarray) -> Dict:
    final_pos = int(len(rank_path) - 1)
    return {
        "shared_path_len": int(len(rank_path)),
        "shared_best_rank": float(np.min(rank_path)),
        "shared_best_rank_position": int(np.argmin(rank_path)),
        "shared_final_rank": float(rank_path[final_pos]),
        "shared_best_logprob": float(np.max(logprob_path)),
        "shared_best_logprob_position": int(np.argmax(logprob_path)),
        "shared_final_logprob": float(logprob_path[final_pos]),
        "shared_logprob_gain": float(logprob_path[final_pos] - np.max(logprob_path)),
    }


def build_shared_lrs_rows(
    traces: Dict[str, List[Dict]],
    langs: List[str],
    thresholds: List[int],
    layer_strategy: str,
) -> List[Dict]:
    rows = []
    for lang in langs:
        for ex in traces.get(lang, []):
            for stage, roles in SHARED_STAGES.items():
                rank_path = concat_shared_path(ex, roles, "rank_gold_first", layer_strategy)
                logprob_path = concat_shared_path(ex, roles, "logprob_gold_first", layer_strategy)
                base = {
                    "lang": lang,
                    "idx": ex["idx"],
                    "stage": stage,
                    "roles": "+".join(roles),
                    "correct": ex["correct"],
                    "prediction": ex["prediction"],
                    "gold": ex["gold"],
                    "shared_lrs_layer_strategy": layer_strategy,
                }
                base.update(shared_path_summary(rank_path, logprob_path))
                for threshold in thresholds:
                    row = dict(base)
                    row["shared_rank_threshold"] = threshold
                    row.update(shared_emergence_score(rank_path, threshold))
                    rows.append(row)
    return rows


def safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 3 or a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return float("nan")
    if method == "spearman":
        return float(a[mask].rank(method="average").corr(b[mask].rank(method="average"), method="pearson"))
    return float(a[mask].corr(b[mask], method=method))


def shared_lrs_correlation_rows(shared_rows: List[Dict]) -> List[Dict]:
    df = pd.DataFrame(shared_rows)
    if df.empty:
        return []
    df["correct_float"] = df["correct"].astype(str).str.lower().isin(("true", "1")).astype(float)
    df["shared_best_rank_score"] = 1.0 / (1.0 + np.log1p(pd.to_numeric(df["shared_best_rank"], errors="coerce")))
    df["shared_final_rank_score"] = 1.0 / (1.0 + np.log1p(pd.to_numeric(df["shared_final_rank"], errors="coerce")))
    score_cols = [
        "shared_latent_reasoning_score",
        "shared_emerged",
        "shared_best_rank_score",
        "shared_final_rank_score",
        "shared_best_logprob",
        "shared_final_logprob",
        "shared_logprob_gain",
    ]
    rows = []
    for (stage, threshold), group in df.groupby(["stage", "shared_rank_threshold"], sort=False):
        lang_group = group.groupby("lang", as_index=False).mean(numeric_only=True)
        for score in score_cols:
            rows.append(
                {
                    "stage": stage,
                    "shared_rank_threshold": int(threshold),
                    "score": score,
                    "example_pearson": safe_corr(group[score], group["correct_float"], "pearson"),
                    "example_spearman": safe_corr(group[score], group["correct_float"], "spearman"),
                    "language_pearson": safe_corr(lang_group[score], lang_group["correct_float"], "pearson"),
                    "language_spearman": safe_corr(lang_group[score], lang_group["correct_float"], "spearman"),
                    "score_mean_correct": float(group.loc[group["correct_float"] == 1.0, score].mean()),
                    "score_mean_wrong": float(group.loc[group["correct_float"] == 0.0, score].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out["accuracy_aligned"] = out["language_pearson"] > 0
    out["language_abs_pearson"] = out["language_pearson"].abs()
    out["example_abs_pearson"] = out["example_pearson"].abs()
    out = out.sort_values(
        ["accuracy_aligned", "language_abs_pearson", "example_abs_pearson"],
        ascending=[False, False, False],
    )
    return out.to_dict(orient="records")


def shared_lrs_language_summary_rows(shared_rows: List[Dict]) -> List[Dict]:
    df = pd.DataFrame(shared_rows)
    if df.empty:
        return []
    df["correct_float"] = df["correct"].astype(str).str.lower().isin(("true", "1")).astype(float)
    rows = []
    for (stage, threshold, lang), group in df.groupby(["stage", "shared_rank_threshold", "lang"], sort=False):
        rows.append(
            {
                "stage": stage,
                "shared_rank_threshold": int(threshold),
                "lang": lang,
                "accuracy": float(group["correct_float"].mean()),
                "total": int(len(group)),
                "shared_latent_reasoning_score": float(group["shared_latent_reasoning_score"].mean()),
                "shared_emergence_rate": float(group["shared_emerged"].mean()),
                "mean_shared_best_rank": float(group["shared_best_rank"].mean()),
                "mean_shared_final_rank": float(group["shared_final_rank"].mean()),
                "mean_shared_logprob_gain": float(group["shared_logprob_gain"].mean()),
                "corr_problem_shared_lrs_with_correct": safe_corr(
                    group["shared_latent_reasoning_score"],
                    group["correct_float"],
                    "pearson",
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: Dict, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def partial_example_fieldnames() -> List[str]:
    fields = [
        "lang",
        "idx",
        "correct",
        "prediction",
        "gold",
        "latent_reasoning_score",
        "question",
        "raw_prediction",
    ]
    for agent in default_agents():
        for key in metric_keys():
            fields.append(f"{agent.role}_{key}")
    return fields


def partial_agent_fieldnames() -> List[str]:
    return [
        "lang",
        "idx",
        "role",
        "agent_name",
        "correct",
        "prediction",
        "gold",
        *metric_keys(),
    ]


def partial_rows_for_example(ex: Dict, rank_threshold: int, layer_strategy: str) -> Tuple[Dict, List[Dict]]:
    role_scores = []
    example_row = {
        "lang": ex["lang"],
        "idx": ex["idx"],
        "correct": ex["correct"],
        "prediction": ex["prediction"],
        "gold": ex["gold"],
        "question": ex["question"],
        "raw_prediction": ex["raw_prediction"],
    }
    agent_rows = []
    for role, agent in ex["agents"].items():
        metrics = agent_metrics(agent, rank_threshold, layer_strategy)
        role_scores.append(metrics["latent_reasoning_score"])
        for key, value in metrics.items():
            example_row[f"{role}_{key}"] = value
        agent_rows.append(
            {
                "lang": ex["lang"],
                "idx": ex["idx"],
                "role": role,
                "agent_name": agent["name"],
                "correct": ex["correct"],
                "prediction": ex["prediction"],
                "gold": ex["gold"],
                **metrics,
            }
        )
    example_row["latent_reasoning_score"] = float(np.mean(role_scores)) if role_scores else 0.0
    return example_row, agent_rows


def checkpoint_example(out_dir: Path, ex: Dict, rank_threshold: int, layer_strategy: str) -> None:
    example_row, agent_rows = partial_rows_for_example(ex, rank_threshold, layer_strategy)
    append_csv(out_dir / "text_agent_similarity_examples.partial.csv", example_row, partial_example_fieldnames())
    agent_fields = partial_agent_fieldnames()
    for row in agent_rows:
        append_csv(out_dir / "text_agent_similarity_agent_examples.partial.csv", row, agent_fields)


def checkpoint_language_trace(out_dir: Path, meta: Dict, lang: str, examples: List[Dict]) -> None:
    shard_dir = out_dir / "trace_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    with (shard_dir / f"{lang}.pkl").open("wb") as f:
        pickle.dump(
            {
                "meta": meta,
                "lang": lang,
                "traces": {lang: examples},
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def prepare_checkpoint_files(out_dir: Path, keep_existing: bool) -> None:
    if keep_existing:
        return
    for rel in [
        "text_agent_similarity_examples.partial.csv",
        "text_agent_similarity_agent_examples.partial.csv",
    ]:
        path = out_dir / rel
        if path.exists():
            path.unlink()


def write_cosine_matrix_csv(path: Path, langs: List[str], cosine_nested: Dict[str, Dict[str, float]]) -> None:
    rows = []
    for lang in langs:
        row = {"lang": lang}
        for other in langs:
            row[f"cosine_to_{other}"] = cosine_nested[lang][other]
        rows.append(row)
    write_csv(path, rows)


def write_language_summary_csv(path: Path, langs: List[str], language_summary: Dict[str, Dict], cosine_nested: Dict[str, Dict[str, float]]) -> None:
    rows = []
    for lang in langs:
        summary = language_summary[lang]
        row = {
            "lang": lang,
            "accuracy": summary["accuracy"],
            "correct": summary["correct"],
            "total": summary["total"],
            "latent_reasoning_score": summary["latent_reasoning_score"],
        }
        for role, score in summary["agent_latent_reasoning_score"].items():
            row[f"{role}_latent_reasoning_score"] = score
        for other in langs:
            row[f"cosine_to_{other}"] = cosine_nested[lang][other]
        rows.append(row)
    write_csv(path, rows)


def write_example_csvs(
    out_dir: Path,
    traces: Dict[str, List[Dict]],
    langs: List[str],
    rank_threshold: int,
    layer_strategy: str,
) -> None:
    example_cosines = build_example_pair_cosines(traces, langs)
    by_lang = {lang: {ex["idx"]: ex for ex in traces.get(lang, [])} for lang in langs}
    example_rows = []
    agent_rows = []
    for lang in langs:
        for ex in traces.get(lang, []):
            idx = ex["idx"]
            role_scores = []
            row = {
                "lang": lang,
                "idx": idx,
                "correct": ex["correct"],
                "prediction": ex["prediction"],
                "gold": ex["gold"],
                "question": ex["question"],
                "raw_prediction": ex["raw_prediction"],
            }
            for other in langs:
                row[f"cosine_to_{other}"] = example_cosines[idx][lang][other]
            for role, agent in ex["agents"].items():
                metrics = agent_metrics(agent, rank_threshold, layer_strategy)
                role_scores.append(metrics["latent_reasoning_score"])
                for key, value in metrics.items():
                    row[f"{role}_{key}"] = value
                agent_row = {
                    "lang": lang,
                    "idx": idx,
                    "role": role,
                    "agent_name": agent["name"],
                    "correct": ex["correct"],
                    "prediction": ex["prediction"],
                    "gold": ex["gold"],
                    **metrics,
                }
                for other in langs:
                    other_ex = by_lang[other].get(idx)
                    agent_row[f"cosine_to_{other}"] = (
                        cosine_between_agents(agent, other_ex["agents"][role])
                        if other_ex is not None and role in other_ex["agents"]
                        else float("nan")
                    )
                agent_rows.append(agent_row)
            row["latent_reasoning_score"] = float(np.mean(role_scores)) if role_scores else 0.0
            example_rows.append(row)
    write_csv(out_dir / "text_agent_similarity_examples.csv", example_rows)
    write_csv(out_dir / "text_agent_similarity_agent_examples.csv", agent_rows)


def jsonable_payload(payload: Dict, cosine_nested: Dict[str, Dict[str, float]], shared_rows: List[Dict]) -> Dict:
    return {
        "meta": payload["meta"],
        "languages": payload["languages"],
        "language_summary": payload["language_summary"],
        "cosine_similarity_matrix": cosine_nested,
        "shared_lrs_summary": shared_lrs_language_summary_rows(shared_rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--languages", type=str, default="bn,de,en,es,fr,ja,ru,sw,te,th,zh")
    parser.add_argument("--prompt", choices=["sequential", "hierarchical"], default="sequential")
    parser.add_argument("--max_examples", type=int, default=5, help="Examples per language. Use -1 for all MGSM test examples.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_trace_steps", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--text_mas_context_length", type=int, default=-1)
    parser.add_argument("--emergence_rank_threshold", type=int, default=1000)
    parser.add_argument(
        "--emergence_layer_strategy",
        choices=["best_layer", "final_layer"],
        default="final_layer",
    )
    parser.add_argument(
        "--shared_lrs_thresholds",
        type=str,
        default="1,5,10,25,50,100,250,500,1000,2500,5000,10000",
    )
    parser.add_argument(
        "--shared_lrs_layer_strategy",
        choices=["final_layer", "best_layer", "mean_layer"],
        default="final_layer",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="src/multilingual-latent-reasoning/results_text_mas_agents")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=1, help="Write partial CSVs and trace shard every N examples. Use 0 to disable.")
    parser.add_argument("--keep_existing_partials", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    model_args = build_args(args, "en")
    model = ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=model_args)

    langs = [x.strip().lower() for x in args.languages.split(",") if x.strip()]
    example_label = "all" if args.max_examples < 0 else f"first{args.max_examples}"
    run_name = args.run_name or f"mgsm_{example_label}_{args.prompt}_text_mas_csv"
    out_dir = Path(args.out_dir) / args.model_name.split("/")[-1] / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_checkpoint_files(out_dir, args.keep_existing_partials)

    shared_thresholds = [int(x.strip()) for x in args.shared_lrs_thresholds.split(",") if x.strip()]
    meta = {
        "model": args.model_name,
        "method": "text_mas",
        "languages": langs,
        "prompt": args.prompt,
        "max_examples": args.max_examples,
        "max_new_tokens": args.max_new_tokens,
        "max_trace_steps": args.max_trace_steps,
        "emergence_rank_threshold": args.emergence_rank_threshold,
        "emergence_layer_strategy": args.emergence_layer_strategy,
        "shared_lrs_thresholds": shared_thresholds,
        "shared_lrs_layer_strategy": args.shared_lrs_layer_strategy,
        "cosine_definition": "Average across common example indices, agents, sampled text-reasoning steps, and layers. Variable-length text traces are resampled to a shared step count per comparison.",
        "shared_lrs_definition": "Cumulative shared TextMAS path over planner, critic, refiner, and judger logit-lens rank trajectories. LRS = 1 - first_emergence_position / shared_path_length.",
        "checkpoint_every": args.checkpoint_every,
    }

    traces: Dict[str, List[Dict]] = {}
    for lang in langs:
        print(f"=== {lang} ===", flush=True)
        traces[lang] = []
        for item_num, item in enumerate(first_mgsm_items(lang, args.max_examples), start=1):
            print(f"  idx={item['idx']}", flush=True)
            ex = run_one_example(model, args, lang, item)
            traces[lang].append(ex)
            if args.checkpoint_every > 0:
                checkpoint_example(out_dir, ex, args.emergence_rank_threshold, args.emergence_layer_strategy)
            if args.checkpoint_every > 0 and item_num % args.checkpoint_every == 0:
                checkpoint_language_trace(out_dir, meta, lang, traces[lang])
                print(f"  [checkpoint] wrote partial rows and {lang} trace shard through idx={item['idx']}", flush=True)
        if args.checkpoint_every > 0:
            checkpoint_language_trace(out_dir, meta, lang, traces[lang])
            print(f"  [checkpoint] finalized {lang} trace shard", flush=True)

    language_summary = {
        lang: summarize_language(traces[lang], args.emergence_rank_threshold, args.emergence_layer_strategy)
        for lang in langs
    }
    cosine_nested = build_all_pairs_cosine(traces, langs)
    shared_rows = build_shared_lrs_rows(
        traces,
        langs,
        shared_thresholds,
        args.shared_lrs_layer_strategy,
    )
    shared_corr_rows = shared_lrs_correlation_rows(shared_rows)
    shared_language_rows = shared_lrs_language_summary_rows(shared_rows)

    payload = {
        "meta": meta,
        "languages": langs,
        "traces": traces,
        "language_summary": language_summary,
        "cosine_similarity_matrix": cosine_nested,
        "shared_lrs_rows": shared_rows,
        "shared_lrs_correlations": shared_corr_rows,
    }
    with (out_dir / "text_mas_mgsm_batch_traces.pkl").open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary_json = jsonable_payload(payload, cosine_nested, shared_rows)
    with (out_dir / "text_mas_mgsm_batch_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)
    with (out_dir / "text_agent_similarity_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    write_language_summary_csv(out_dir / "text_agent_similarity_language_summary.csv", langs, language_summary, cosine_nested)
    write_cosine_matrix_csv(out_dir / "text_agent_similarity_cosine_matrix.csv", langs, cosine_nested)
    write_example_csvs(out_dir, traces, langs, args.emergence_rank_threshold, args.emergence_layer_strategy)
    write_csv(out_dir / "shared_lrs_problem_rows.csv", shared_rows)
    write_csv(out_dir / "shared_lrs_language_summary.csv", shared_language_rows)
    write_csv(out_dir / "shared_lrs_score_correlations.csv", shared_corr_rows)
    if shared_corr_rows:
        pd.DataFrame(shared_corr_rows).head(25).to_csv(out_dir / "shared_lrs_top25.csv", index=False)

    print("\nLanguage averages:")
    for lang in langs:
        row = language_summary[lang]
        print(lang, "acc=", row["accuracy"], "lrs=", row["latent_reasoning_score"])
    if shared_corr_rows:
        print("\nTop shared LRS/candidate correlations:")
        print(pd.DataFrame(shared_corr_rows).head(15).to_string(index=False))
    print(f"[OK] wrote {out_dir}")


if __name__ == "__main__":
    main()
