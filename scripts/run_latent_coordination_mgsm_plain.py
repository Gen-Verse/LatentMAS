#!/usr/bin/env python3
"""Run latent_coordination agents on MGSM without modifying latent_coordination.

This runner is intentionally external to src/latent_coordination. It loads MGSM,
uses the existing TranslationAgent and ReasoningAgent classes, and scores with
the repo's boxed-answer parser.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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
from latent_coordination.latent_space.universal_space import UniversalLatentSpace  # noqa: E402
from latent_coordination.orchestration.router import AdaptiveOrchestrator, RoutingPlan  # noqa: E402
from utils import extract_gsm8k_answer, normalize_answer  # noqa: E402


def iter_examples(lang: str, max_examples: int, start_idx: int = 0) -> Iterable[Dict]:
    yielded = 0
    for idx, item in enumerate(load_mgsm(split="test", lang=lang)):
        if idx < start_idx:
            continue
        if max_examples >= 0 and yielded >= max_examples:
            break
        row = dict(item)
        row["idx"] = idx
        row["lang"] = lang
        yielded += 1
        yield row


_MGSM_CACHE: Dict[str, List[Dict]] = {}


def get_mgsm_item(lang: str, idx: int) -> Dict:
    if lang not in _MGSM_CACHE:
        _MGSM_CACHE[lang] = [dict(item) for item in load_mgsm(split="test", lang=lang)]
    item = dict(_MGSM_CACHE[lang][idx])
    item["idx"] = idx
    item["lang"] = lang
    return item


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
    return TranslationAgent(
        cfg,
        steer_layer=args.translation_layer,
        sfr_threshold=args.sfr_threshold,
    )


def get_universal_space(
    args: argparse.Namespace,
    translator: TranslationAgent,
    reasoner: ReasoningAgent,
) -> UniversalLatentSpace:
    cached = getattr(args, "_universal_space", None)
    if cached is not None:
        return cached

    universal_space = UniversalLatentSpace(
        universal_dim=args.universal_dim,
        device=args.device,
    )
    universal_space.register_agent(translator.agent_id, hidden_dim=args.hidden_dim)
    universal_space.register_agent(reasoner.agent_id, hidden_dim=args.hidden_dim)
    if args.uls_adapter_dir:
        universal_space.load_adapters(args.uls_adapter_dir)
    args._universal_space = universal_space
    return universal_space


def score_text(text: str, gold: str) -> tuple[str | None, bool]:
    pred = normalize_answer(extract_gsm8k_answer(text))
    gold_norm = normalize_answer(gold)
    return pred, bool(pred and gold_norm and pred == gold_norm)


def numbers_in_text(text: str) -> List[str]:
    return re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text or "")


def translation_looks_contaminated(source: str, translation: str, args: argparse.Namespace) -> tuple[bool, str]:
    """Detect translator outputs that are likely to poison downstream reasoning."""
    text = translation or ""
    lowered = text.lower()
    if not text.strip():
        return True, "empty_translation"

    answer_markers = [
        "answer is",
        "the answer",
        "final answer",
        "therefore",
        "so,",
        "so ",
        "boxed",
        "\\boxed",
        "উত্তর",
        "respuesta",
        "réponse",
        "ответ",
        "答案",
        "答え",
        "జవాబు",
        "คำตอบ",
    ]
    if any(marker in lowered for marker in answer_markers):
        return True, "answer_or_reasoning_marker"

    source_numbers = set(numbers_in_text(source))
    output_numbers = set(numbers_in_text(text))
    extra_numbers = output_numbers - source_numbers
    if len(extra_numbers) > args.max_extra_translation_numbers:
        return True, f"extra_numbers={sorted(extra_numbers)}"

    source_len = max(len(source), 1)
    if len(text) > source_len * args.max_translation_length_ratio:
        return True, "too_long"

    return False, "ok"


def run_translator_latent_only(
    translator: TranslationAgent,
    reasoner: ReasoningAgent,
    item: Dict,
    args: argparse.Namespace,
    *,
    gated: bool = False,
) -> Dict:
    target_language = item["lang"] if args.translation_target_language == "same" else args.translation_target_language
    translation_task = AgentTask(
        task_id=f"mgsm_{item['lang']}_{item['idx']}_translate_latent",
        query=item["question"],
        target_language=target_language,
    )
    translation = translator.process(translation_task)
    contaminated, gate_reason = translation_looks_contaminated(
        item["question"],
        translation.output_text,
        args,
    )
    use_latent = not (gated and contaminated)
    latent_state = translation.latent_state if use_latent else None
    used_uls_adapter = False
    if latent_state is not None and (args.use_uls_transfer or args.uls_adapter_dir):
        universal_space = get_universal_space(args, translator, reasoner)
        latent_state = universal_space.transfer(
            translator.agent_id,
            reasoner.agent_id,
            latent_state,
            norm_match=True,
        )
        used_uls_adapter = True
    reasoning_task = AgentTask(
        task_id=f"mgsm_{item['lang']}_{item['idx']}_latent_reason",
        query=item["question"],
        context=args.context,
        latent_state=latent_state,
        target_language=target_language,
    )
    reasoning = reasoner.process(reasoning_task)
    first_pred, first_ok = score_text(translation.output_text, item["gold"])
    pred, ok = score_text(reasoning.output_text, item["gold"])
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
        "translated_question": "",
        "translation_output": translation.output_text,
        "used_second_pass": True,
        "used_translation": True,
        "used_translation_latent": use_latent,
        "used_uls_adapter": used_uls_adapter,
        "used_uls_transfer": bool(latent_state is not None and (args.use_uls_transfer or args.uls_adapter_dir)),
        "uls_adapter_dir": args.uls_adapter_dir,
        "translation_contaminated": contaminated,
        "translation_gate_reason": gate_reason,
        "first_elapsed_ms": translation.elapsed_ms,
        "translation_elapsed_ms": translation.elapsed_ms,
        "final_elapsed_ms": reasoning.elapsed_ms,
    }


def run_latent_consensus_verify(
    translator: TranslationAgent,
    reasoner: ReasoningAgent,
    item: Dict,
    args: argparse.Namespace,
) -> Dict:
    """Compare baseline and latent-only candidates, then verify disagreements."""
    baseline = run_reasoning_only(reasoner, item, args)
    latent = run_translator_latent_only(translator, reasoner, item, args, gated=True)
    baseline_pred = baseline["prediction"]
    latent_pred = latent["prediction"]

    if baseline_pred == latent_pred or not latent_pred:
        chosen = baseline
        verifier_text = ""
        verifier_pred = baseline_pred
        used_verifier = False
    else:
        target_language = item["lang"] if args.translation_target_language == "same" else args.translation_target_language
        verifier_prompt = (
            "Solve the original math problem and choose the correct candidate answer. "
            "Ignore any candidate that comes from a mistranslation or changes the story. "
            "Output only a short solution and a boxed final answer.\n\n"
            f"Original problem:\n{item['question']}\n\n"
            f"Candidate A: {baseline_pred}\n"
            f"Candidate B: {latent_pred}\n"
        )
        verifier_task = AgentTask(
            task_id=f"mgsm_{item['lang']}_{item['idx']}_latent_verify",
            query=verifier_prompt,
            target_language=target_language,
        )
        verifier = reasoner.process(verifier_task)
        verifier_text = verifier.output_text
        verifier_pred, _ = score_text(verifier_text, item["gold"])
        chosen = dict(latent if verifier_pred == latent_pred else baseline)
        chosen["raw_prediction"] = verifier_text if verifier_pred else chosen["raw_prediction"]
        chosen["prediction"] = verifier_pred or chosen["prediction"]
        chosen["correct"] = bool(verifier_pred and verifier_pred == normalize_answer(item["gold"]))
        used_verifier = True

    chosen = dict(chosen)
    chosen["mode"] = args.mode
    chosen["baseline_prediction"] = baseline_pred
    chosen["latent_prediction"] = latent_pred
    chosen["verifier_prediction"] = verifier_pred
    chosen["used_verifier"] = used_verifier
    chosen["verifier_output"] = verifier_text
    chosen["translation_contaminated"] = latent.get("translation_contaminated", "")
    chosen["translation_gate_reason"] = latent.get("translation_gate_reason", "")
    return chosen


def run_english_anchor_latent(
    translator: TranslationAgent,
    reasoner: ReasoningAgent,
    item: Dict,
    args: argparse.Namespace,
) -> Dict:
    """Inject hidden states from the parallel English MGSM question, not English text."""
    target_language = item["lang"] if args.translation_target_language == "same" else args.translation_target_language
    anchor_item = get_mgsm_item(args.anchor_lang, item["idx"])
    t0 = translator._start_timer()
    translator._ensure_model_loaded()
    hs = translator.extract_hidden_states(
        anchor_item["question"],
        layer_ids=[args.translation_layer],
    )
    anchor_latent = hs.get(args.translation_layer)
    anchor_elapsed = translator._stop_timer(t0)
    reasoning_task = AgentTask(
        task_id=f"mgsm_{item['lang']}_{item['idx']}_english_anchor_latent",
        query=item["question"],
        context=args.context,
        latent_state=anchor_latent,
        target_language=target_language,
    )
    reasoning = reasoner.process(reasoning_task)
    pred, ok = score_text(reasoning.output_text, item["gold"])
    return {
        "lang": item["lang"],
        "idx": item["idx"],
        "mode": args.mode,
        "correct": ok,
        "prediction": pred,
        "gold": normalize_answer(item["gold"]),
        "first_pass_correct": "",
        "first_pass_prediction": "",
        "question": item["question"],
        "raw_prediction": reasoning.output_text,
        "first_pass_output": "",
        "translated_question": "",
        "translation_output": "",
        "anchor_lang": args.anchor_lang,
        "anchor_question": anchor_item["question"],
        "used_second_pass": True,
        "used_translation": False,
        "used_translation_latent": True,
        "translation_contaminated": False,
        "translation_gate_reason": "english_anchor_hidden_state",
        "first_elapsed_ms": anchor_elapsed,
        "translation_elapsed_ms": anchor_elapsed,
        "final_elapsed_ms": reasoning.elapsed_ms,
    }


def run_reasoning_only(reasoner: ReasoningAgent, item: Dict, args: argparse.Namespace) -> Dict:
    task = AgentTask(
        task_id=f"mgsm_{item['lang']}_{item['idx']}",
        query=item["question"],
        context=args.context,
        target_language=item["lang"],
    )
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
        "used_second_pass": True,
        "used_translation": False,
        "used_translation_latent": True,
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
    target_language = item["lang"] if args.translation_target_language == "same" else args.translation_target_language
    translation_task = AgentTask(
        task_id=f"mgsm_{item['lang']}_{item['idx']}_translate",
        query=item["question"],
        target_language=target_language,
    )
    translation = translator.process(translation_task)
    translated_question = translation.output_text.strip() or item["question"]
    latent_state = translation.latent_state if args.mode == "translate_reason_latent" else None

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
    first_pred, first_ok = score_text(translation.output_text, item["gold"])
    pred, ok = score_text(reasoning.output_text, item["gold"])
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


def run_orchestrated_translate_reason_latent(
    translator: TranslationAgent,
    reasoner: ReasoningAgent,
    item: Dict,
    args: argparse.Namespace,
) -> Dict:
    """Run the intended orchestrator path: original query + context + latent transfer."""
    target_language = item["lang"] if args.translation_target_language == "same" else args.translation_target_language
    router = AdaptiveOrchestrator(device=args.device, router_type="kmeans")
    router.register_agent(translator)
    router.register_agent(reasoner)
    universal_space = get_universal_space(args, translator, reasoner)
    task = AgentTask(
        task_id=f"mgsm_{item['lang']}_{item['idx']}_orchestrated",
        query=item["question"],
        context=args.context,
        target_language=target_language,
    )
    routing_plan = RoutingPlan(
        task_id=task.task_id,
        selected_agents=[translator.agent_id, reasoner.agent_id],
        execution_order=[translator.agent_id, reasoner.agent_id],
        estimated_cost=2.0,
        routing_confidence=1.0,
    )
    result = router.execute(task, routing_plan, universal_space)

    translation = result.agent_responses[0] if result.agent_responses else None
    reasoning = result.agent_responses[-1] if result.agent_responses else None
    final_text = result.final_output
    pred, ok = score_text(final_text, item["gold"])
    first_text = translation.output_text if translation is not None else ""
    first_pred, first_ok = score_text(first_text, item["gold"])
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
        "raw_prediction": final_text,
        "first_pass_output": first_text,
        "translated_question": "",
        "translation_output": first_text,
        "used_second_pass": True,
        "used_translation": True,
        "used_translation_latent": True,
        "used_uls_adapter": bool(args.uls_adapter_dir),
        "uls_adapter_dir": args.uls_adapter_dir,
        "first_elapsed_ms": translation.elapsed_ms if translation is not None else 0.0,
        "translation_elapsed_ms": translation.elapsed_ms if translation is not None else 0.0,
        "final_elapsed_ms": reasoning.elapsed_ms if reasoning is not None else 0.0,
        "orchestrator_elapsed_ms": result.total_elapsed_ms,
        "communication_cost_tokens": result.communication_cost_tokens,
        "communication_cost_latent": result.communication_cost_latent,
        "routing_order": "->".join(routing_plan.execution_order),
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
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--hidden_dim", type=int, default=2560)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--translation_max_new_tokens", type=int, default=512)
    parser.add_argument("--reasoning_layer", type=int, default=-2)
    parser.add_argument("--translation_layer", type=int, default=-1)
    parser.add_argument("--n_reasoning_components", type=int, default=16)
    parser.add_argument("--universal_dim", type=int, default=256)
    parser.add_argument(
        "--uls_adapter_dir",
        default="",
        help="Optional directory containing UniversalLatentSpace agent adapter checkpoints.",
    )
    parser.add_argument(
        "--use_uls_transfer",
        action="store_true",
        help="Route direct translator->reasoner latent handoff through ULS even without trained adapters.",
    )
    parser.add_argument("--sfr_threshold", type=float, default=0.3)
    parser.add_argument("--max_extra_translation_numbers", type=int, default=0)
    parser.add_argument("--max_translation_length_ratio", type=float, default=2.5)
    parser.add_argument("--anchor_lang", default="en")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument(
        "--mode",
        choices=[
            "reasoning_only",
            "reasoning_latent_2pass",
            "translate_reason",
            "translate_reason_latent",
            "orchestrated_translate_reason_latent",
            "translator_latent_only",
            "translator_latent_only_gated",
            "latent_consensus_verify",
            "english_anchor_latent",
        ],
        default="reasoning_only",
    )
    parser.add_argument("--pass_text_context", action="store_true")
    parser.add_argument("--context", default="")
    parser.add_argument(
        "--translation_target_language",
        default="same",
        help="Use 'same' to keep each MGSM language as target; set 'en' only for an English-pivot ablation.",
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
    translator = (
        make_translation_agent(args)
        if args.mode in {
            "translate_reason",
            "translate_reason_latent",
            "orchestrated_translate_reason_latent",
            "translator_latent_only",
            "translator_latent_only_gated",
            "latent_consensus_verify",
            "english_anchor_latent",
        }
        else None
    )
    rows: List[Dict] = []

    for lang in langs:
        print(f"=== {lang} ===", flush=True)
        for item in iter_examples(lang, args.max_examples, args.start_idx):
            print(f"  idx={item['idx']}", flush=True)
            if args.mode == "reasoning_only":
                row = run_reasoning_only(reasoner, item, args)
            elif args.mode == "reasoning_latent_2pass":
                row = run_reasoning_latent_2pass(reasoner, item, args)
            elif args.mode == "orchestrated_translate_reason_latent":
                assert translator is not None
                row = run_orchestrated_translate_reason_latent(translator, reasoner, item, args)
            elif args.mode == "translator_latent_only":
                assert translator is not None
                row = run_translator_latent_only(translator, reasoner, item, args, gated=False)
            elif args.mode == "translator_latent_only_gated":
                assert translator is not None
                row = run_translator_latent_only(translator, reasoner, item, args, gated=True)
            elif args.mode == "latent_consensus_verify":
                assert translator is not None
                row = run_latent_consensus_verify(translator, reasoner, item, args)
            elif args.mode == "english_anchor_latent":
                assert translator is not None
                row = run_english_anchor_latent(translator, reasoner, item, args)
            else:
                assert translator is not None
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
        summary_rows.append({"lang": lang, "accuracy": correct / len(group), "correct": correct, "total": len(group)})
    write_csv(out_dir / "language_summary.csv", summary_rows)

    meta = {
        "model_name": args.model_name,
        "mode": args.mode,
        "languages": langs,
        "max_examples": args.max_examples,
        "start_idx": args.start_idx,
        "max_new_tokens": args.max_new_tokens,
        "translation_max_new_tokens": args.translation_max_new_tokens,
        "translation_target_language": args.translation_target_language,
        "universal_dim": args.universal_dim,
        "uls_adapter_dir": args.uls_adapter_dir,
        "use_uls_transfer": args.use_uls_transfer,
        "sfr_threshold": args.sfr_threshold,
        "max_extra_translation_numbers": args.max_extra_translation_numbers,
        "max_translation_length_ratio": args.max_translation_length_ratio,
        "anchor_lang": args.anchor_lang,
        "device": args.device,
        "dtype": args.dtype,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
