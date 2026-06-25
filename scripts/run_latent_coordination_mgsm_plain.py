#!/usr/bin/env python3
<<<<<<< HEAD
"""Run latent_coordination agents on MGSM without modifying latent_coordination.

This runner is intentionally external to src/latent_coordination. It loads MGSM,
uses the existing TranslationAgent and ReasoningAgent classes, and scores with
the repo's boxed-answer parser.
=======
"""Run the prototype latent_coordination agents on MGSM without editing that package.

This is intentionally a thin adapter around src/latent_coordination:
  - loads MGSM via the existing data.py loader
  - runs the existing ReasoningAgent and/or TranslationAgent
  - optionally runs a second pass with the first pass latent_state injected
  - scores with the existing GSM/MGSM answer extractor

Modes:
  reasoning_only       one ReasoningAgent call per problem
  reasoning_latent_2pass first ReasoningAgent call, then a second call with latent_state
  translate_reason     TranslationAgent processes question, ReasoningAgent solves text
  translate_reason_latent TranslationAgent processes question and passes latent_state
>>>>>>> feat/unsloth-llamacpp-pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data import load_mgsm  # noqa: E402
from latent_coordination.agents.base_agent import AgentConfig, AgentTask  # noqa: E402
from latent_coordination.agents.specialized_agents import ReasoningAgent, TranslationAgent  # noqa: E402
from utils import extract_gsm8k_answer, normalize_answer  # noqa: E402


def iter_examples(lang: str, max_examples: int) -> Iterable[Dict]:
    for idx, item in enumerate(load_mgsm(split="test", lang=lang)):
        if max_examples >= 0 and idx >= max_examples:
            break
        row = dict(item)
        row["idx"] = idx
        row["lang"] = lang
        yield row


def make_reasoning_agent(args: argparse.Namespace) -> ReasoningAgent:
    cfg = AgentConfig(
        agent_id="mgsm_reasoner",
        model_id=args.model_name,
        role="reasoning",
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        hidden_dim=args.hidden_dim,
        dtype=args.dtype,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=True,
    )
    return ReasoningAgent(
        cfg,
        reasoning_layer=args.reasoning_layer,
        n_reasoning_components=args.n_reasoning_components,
    )


def make_translation_agent(args: argparse.Namespace) -> TranslationAgent:
    cfg = AgentConfig(
        agent_id="mgsm_translator",
        model_id=args.model_name,
        role="translation",
        device=args.device,
        max_new_tokens=args.translation_max_new_tokens,
        hidden_dim=args.hidden_dim,
        dtype=args.dtype,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=True,
    )
    return TranslationAgent(cfg, steer_layer=args.translation_layer)


def score_text(text: str, gold: str) -> tuple[str | None, bool]:
    pred = normalize_answer(extract_gsm8k_answer(text))
    gold_norm = normalize_answer(gold)
    return pred, bool(pred and gold_norm and pred == gold_norm)


<<<<<<< HEAD
def run_reasoning_only(reasoner: ReasoningAgent, item: Dict, args: argparse.Namespace) -> Dict:
    task = AgentTask(
=======
def run_reasoning_only(agent: ReasoningAgent, item: Dict, args: argparse.Namespace) -> Dict:
    base_task = AgentTask(
>>>>>>> feat/unsloth-llamacpp-pipeline
        task_id=f"mgsm_{item['lang']}_{item['idx']}",
        query=item["question"],
        context=args.context,
        target_language=item["lang"],
    )
<<<<<<< HEAD
    response = reasoner.process(task)
    pred, ok = score_text(response.output_text, item["gold"])
    return {
        "lang": item["lang"],
        "idx": item["idx"],
        "mode": args.mode,
        "correct": ok,
        "prediction": pred,
        "gold": normalize_answer(item["gold"]),
        "first_pass_correct": ok,
        "first_pass_prediction": pred,
        "question": item["question"],
        "raw_prediction": response.output_text,
        "first_pass_output": response.output_text,
        "translated_question": "",
        "translation_output": "",
        "used_second_pass": False,
        "used_translation": False,
        "used_translation_latent": False,
        "first_elapsed_ms": response.elapsed_ms,
        "translation_elapsed_ms": 0.0,
        "final_elapsed_ms": response.elapsed_ms,
    }


def run_reasoning_latent_2pass(reasoner: ReasoningAgent, item: Dict, args: argparse.Namespace) -> Dict:
    first_task = AgentTask(
        task_id=f"mgsm_{item['lang']}_{item['idx']}_first",
        query=item["question"],
        context=args.context,
        target_language=item["lang"],
    )
    first = reasoner.process(first_task)
    second_task = AgentTask(
        task_id=f"mgsm_{item['lang']}_{item['idx']}_latent2",
        query=item["question"],
        context=first.output_text if args.pass_text_context else "",
        latent_state=first.latent_state,
        target_language=item["lang"],
    )
    final = reasoner.process(second_task)
    first_pred, first_ok = score_text(first.output_text, item["gold"])
=======

    first = agent.process(base_task)
    final = first
    first_pred, first_ok = score_text(first.output_text, item["gold"])

    if args.mode == "reasoning_latent_2pass":
        second_task = AgentTask(
            task_id=f"mgsm_{item['lang']}_{item['idx']}_latent2",
            query=item["question"],
            context=first.output_text if args.pass_text_context else "",
            latent_state=first.latent_state,
            target_language=item["lang"],
        )
        final = agent.process(second_task)

>>>>>>> feat/unsloth-llamacpp-pipeline
    pred, ok = score_text(final.output_text, item["gold"])
    return {
        "lang": item["lang"],
        "idx": item["idx"],
        "mode": args.mode,
        "correct": ok,
        "prediction": pred,
        "gold": normalize_answer(item["gold"]),
        "first_pass_correct": first_ok,
        "first_pass_prediction": first_pred,
        "question": item["question"],
        "raw_prediction": final.output_text,
        "first_pass_output": first.output_text,
        "translated_question": "",
        "translation_output": "",
<<<<<<< HEAD
        "used_second_pass": True,
        "used_translation": False,
        "used_translation_latent": True,
=======
        "used_second_pass": args.mode == "reasoning_latent_2pass",
        "used_translation": False,
        "used_translation_latent": False,
>>>>>>> feat/unsloth-llamacpp-pipeline
        "first_elapsed_ms": first.elapsed_ms,
        "translation_elapsed_ms": 0.0,
        "final_elapsed_ms": final.elapsed_ms,
    }


def run_translate_reason(
    translator: TranslationAgent,
    reasoner: ReasoningAgent,
    item: Dict,
    args: argparse.Namespace,
) -> Dict:
<<<<<<< HEAD
    target_language = item["lang"] if args.translation_target_language == "same" else args.translation_target_language
=======
    target_language = (
        item["lang"]
        if args.translation_target_language == "same"
        else args.translation_target_language
    )
>>>>>>> feat/unsloth-llamacpp-pipeline
    translation_task = AgentTask(
        task_id=f"mgsm_{item['lang']}_{item['idx']}_translate",
        query=item["question"],
        target_language=target_language,
    )
    translation = translator.process(translation_task)
    translated_question = translation.output_text.strip() or item["question"]
<<<<<<< HEAD
    latent_state = translation.latent_state if args.mode == "translate_reason_latent" else None

=======

    latent_state = (
        translation.latent_state
        if args.mode == "translate_reason_latent"
        else None
    )
>>>>>>> feat/unsloth-llamacpp-pipeline
    context = ""
    if args.include_original_question:
        context = f"Original question ({item['lang']}): {item['question']}"

    reasoning_task = AgentTask(
        task_id=f"mgsm_{item['lang']}_{item['idx']}_reason",
        query=translated_question,
        context=context,
        latent_state=latent_state,
        target_language=target_language,
    )
    reasoning = reasoner.process(reasoning_task)
<<<<<<< HEAD
    first_pred, first_ok = score_text(translation.output_text, item["gold"])
    pred, ok = score_text(reasoning.output_text, item["gold"])
=======

    pred, ok = score_text(reasoning.output_text, item["gold"])
    first_pred, first_ok = score_text(translation.output_text, item["gold"])
>>>>>>> feat/unsloth-llamacpp-pipeline
    return {
        "lang": item["lang"],
        "idx": item["idx"],
        "mode": args.mode,
        "correct": ok,
        "prediction": pred,
        "gold": normalize_answer(item["gold"]),
        "first_pass_correct": first_ok,
        "first_pass_prediction": first_pred,
        "question": item["question"],
        "raw_prediction": reasoning.output_text,
        "first_pass_output": translation.output_text,
        "translated_question": translated_question,
        "translation_output": translation.output_text,
        "used_second_pass": True,
        "used_translation": True,
        "used_translation_latent": args.mode == "translate_reason_latent",
        "first_elapsed_ms": translation.elapsed_ms,
        "translation_elapsed_ms": translation.elapsed_ms,
        "final_elapsed_ms": reasoning.elapsed_ms,
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B")
    parser.add_argument("--languages", default="bn,de,en,es,fr,ja,ru,sw,te,th,zh")
    parser.add_argument("--max_examples", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--hidden_dim", type=int, default=2560)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--translation_max_new_tokens", type=int, default=512)
    parser.add_argument("--reasoning_layer", type=int, default=-2)
    parser.add_argument("--translation_layer", type=int, default=-1)
    parser.add_argument("--n_reasoning_components", type=int, default=16)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument(
        "--mode",
<<<<<<< HEAD
        choices=["reasoning_only", "reasoning_latent_2pass", "translate_reason", "translate_reason_latent"],
        default="reasoning_only",
    )
    parser.add_argument("--pass_text_context", action="store_true")
=======
        choices=[
            "reasoning_only",
            "reasoning_latent_2pass",
            "translate_reason",
            "translate_reason_latent",
        ],
        default="reasoning_only",
    )
    parser.add_argument(
        "--pass_text_context",
        action="store_true",
        help="In 2-pass mode, pass first output as text context in addition to latent_state.",
    )
>>>>>>> feat/unsloth-llamacpp-pipeline
    parser.add_argument("--context", default="")
    parser.add_argument(
        "--translation_target_language",
        default="same",
<<<<<<< HEAD
        help="Use 'same' to keep each MGSM language as target; set 'en' only for an English-pivot ablation.",
=======
        help="Use 'same' to keep each MGSM language as the translation target; set e.g. 'en' only for an English-pivot ablation.",
>>>>>>> feat/unsloth-llamacpp-pipeline
    )
    parser.add_argument("--include_original_question", action="store_true")
    parser.add_argument("--out_dir", default="results/latent_coordination_mgsm_plain")
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    langs = [x.strip().lower() for x in args.languages.split(",") if x.strip()]
    example_label = "all" if args.max_examples < 0 else f"first{args.max_examples}"
    model_safe = args.model_name.split("/")[-1]
    run_name = args.run_name or f"mgsm_{example_label}_{args.mode}"
    out_dir = Path(args.out_dir) / model_safe / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    reasoner = make_reasoning_agent(args)
<<<<<<< HEAD
    translator = make_translation_agent(args) if args.mode in {"translate_reason", "translate_reason_latent"} else None
=======
    translator = (
        make_translation_agent(args)
        if args.mode in {"translate_reason", "translate_reason_latent"}
        else None
    )
>>>>>>> feat/unsloth-llamacpp-pipeline
    rows: List[Dict] = []

    for lang in langs:
        print(f"=== {lang} ===", flush=True)
        for item in iter_examples(lang, args.max_examples):
            print(f"  idx={item['idx']}", flush=True)
<<<<<<< HEAD
            if args.mode == "reasoning_only":
                row = run_reasoning_only(reasoner, item, args)
            elif args.mode == "reasoning_latent_2pass":
                row = run_reasoning_latent_2pass(reasoner, item, args)
            else:
                assert translator is not None
=======
            if translator is None:
                row = run_reasoning_only(reasoner, item, args)
            else:
>>>>>>> feat/unsloth-llamacpp-pipeline
                row = run_translate_reason(translator, reasoner, item, args)
            rows.append(row)
            write_csv(out_dir / "examples.partial.csv", rows)

    write_csv(out_dir / "examples.csv", rows)
    summary_rows = []
    for lang in langs:
        group = [r for r in rows if r["lang"] == lang]
        if not group:
            continue
        correct = sum(1 for r in group if r["correct"])
<<<<<<< HEAD
        summary_rows.append({"lang": lang, "accuracy": correct / len(group), "correct": correct, "total": len(group)})
=======
        summary_rows.append(
            {
                "lang": lang,
                "accuracy": correct / len(group),
                "correct": correct,
                "total": len(group),
            }
        )
>>>>>>> feat/unsloth-llamacpp-pipeline
    write_csv(out_dir / "language_summary.csv", summary_rows)

    meta = {
        "model_name": args.model_name,
        "mode": args.mode,
        "languages": langs,
        "max_examples": args.max_examples,
        "max_new_tokens": args.max_new_tokens,
        "translation_max_new_tokens": args.translation_max_new_tokens,
        "translation_target_language": args.translation_target_language,
        "device": args.device,
        "dtype": args.dtype,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
