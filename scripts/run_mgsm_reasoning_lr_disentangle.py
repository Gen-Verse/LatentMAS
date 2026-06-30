#!/usr/bin/env python3
"""Training-free L-R disentanglement pilot for single-agent MGSM reasoning.

This script keeps the language model frozen. For each target language it:

1. collects hidden states for parallel English/target MGSM questions,
2. estimates a target-language direction subspace from target - English states,
3. sweeps activation projection strengths on a calibration split,
4. evaluates the direct baseline and the calibrated intervention on heldout
   MGSM examples.

The intervention is a forward hook on middle transformer layers:

    h' = h - alpha * P_lang(h)

where P_lang is the projection onto the language-specific subspace.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data import load_mgsm  # noqa: E402
from latent_coordination.agents.base_agent import AgentConfig, AgentTask  # noqa: E402
from latent_coordination.agents.specialized_agents import ReasoningAgent  # noqa: E402
from utils import extract_gsm8k_answer, normalize_answer  # noqa: E402


@dataclass
class LanguageSubspace:
    lang: str
    layer_to_basis: Dict[int, torch.Tensor]
    n_components: int
    calibration_examples: int


_MGSM_CACHE: Dict[str, List[Dict]] = {}


def get_mgsm_items(lang: str) -> List[Dict]:
    if lang not in _MGSM_CACHE:
        rows = []
        for idx, item in enumerate(load_mgsm(split="test", lang=lang)):
            row = dict(item)
            row["idx"] = idx
            row["lang"] = lang
            rows.append(row)
        _MGSM_CACHE[lang] = rows
    return _MGSM_CACHE[lang]


def iter_mgsm_slice(lang: str, start_idx: int, count: int) -> Iterable[Dict]:
    items = get_mgsm_items(lang)
    stop = len(items) if count < 0 else min(len(items), start_idx + count)
    for idx in range(start_idx, stop):
        yield dict(items[idx])


def parse_layers(layer_spec: str, n_layers: int | None = None) -> List[int]:
    layers: List[int] = []
    for chunk in layer_spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            step = 1 if hi >= lo else -1
            layers.extend(range(lo, hi + step, step))
        else:
            layers.append(int(chunk))
    if n_layers is not None:
        resolved = []
        for layer in layers:
            resolved_layer = n_layers + layer if layer < 0 else layer
            if not (0 <= resolved_layer < n_layers):
                raise ValueError(f"Layer {layer} resolves outside model layer range 0..{n_layers - 1}")
            resolved.append(resolved_layer)
        return sorted(set(resolved))
    return sorted(set(layers))


def parse_floats(values: str) -> List[float]:
    return [float(x.strip()) for x in values.split(",") if x.strip()]


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
    return ReasoningAgent(cfg, reasoning_layer=args.reasoning_layer)


def model_layers(model: torch.nn.Module):
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError("Could not find transformer blocks at model.model.layers.")
    return layers


def prompt_for(agent: ReasoningAgent, question: str, lang: str) -> str:
    task = AgentTask(
        task_id=f"prompt_{lang}",
        query=question,
        target_language=lang,
    )
    return agent._build_cot_prompt(task)


def last_token_hidden_by_layer(
    agent: ReasoningAgent,
    prompt: str,
    layers: Sequence[int],
) -> Dict[int, torch.Tensor]:
    agent._ensure_model_loaded()
    inputs = agent._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(agent._device)
    blocks = model_layers(agent._model)
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def make_capture_hook(layer: int):
        def hook(_module, _module_inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden[:, -1, :].detach().float().cpu().squeeze(0)

        return hook

    for layer in layers:
        handles.append(blocks[layer].register_forward_hook(make_capture_hook(layer)))

    with torch.no_grad():
        try:
            agent._model(**inputs, output_hidden_states=False, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    missing = [layer for layer in layers if layer not in captured]
    if missing:
        raise RuntimeError(f"Failed to capture hidden states for layers: {missing}")
    return captured


def fit_language_subspace(
    agent: ReasoningAgent,
    lang: str,
    layers: Sequence[int],
    calibration_start_idx: int,
    calibration_examples: int,
    n_components: int,
) -> LanguageSubspace:
    if lang == "en":
        return LanguageSubspace(lang=lang, layer_to_basis={}, n_components=0, calibration_examples=0)

    diffs_by_layer: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
    target_items = get_mgsm_items(lang)
    english_items = get_mgsm_items("en")
    stop = min(len(target_items), len(english_items), calibration_start_idx + calibration_examples)
    for idx in range(calibration_start_idx, stop):
        target_prompt = prompt_for(agent, target_items[idx]["question"], lang)
        english_prompt = prompt_for(agent, english_items[idx]["question"], "en")
        target_hidden = last_token_hidden_by_layer(agent, target_prompt, layers)
        english_hidden = last_token_hidden_by_layer(agent, english_prompt, layers)
        for layer in layers:
            diffs_by_layer[layer].append(target_hidden[layer] - english_hidden[layer])

    layer_to_basis: Dict[int, torch.Tensor] = {}
    for layer, diffs in diffs_by_layer.items():
        matrix = torch.stack(diffs, dim=0)
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
        rank = min(n_components, matrix.shape[0], matrix.shape[1])
        if rank <= 0:
            continue
        _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        # [hidden_dim, rank], orthonormal basis for the language-contrast subspace.
        layer_to_basis[layer] = vh[:rank].T.contiguous()

    return LanguageSubspace(
        lang=lang,
        layer_to_basis=layer_to_basis,
        n_components=n_components,
        calibration_examples=max(0, stop - calibration_start_idx),
    )


def make_projection_hook(basis_cpu: torch.Tensor, alpha: float):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        basis = basis_cpu.to(device=hidden.device, dtype=torch.float32)
        hidden_float = hidden.float()
        projected = torch.matmul(torch.matmul(hidden_float, basis), basis.T)
        steered = (hidden_float - alpha * projected).to(dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    return hook


@contextmanager
def language_projection_hooks(
    agent: ReasoningAgent,
    subspace: LanguageSubspace,
    alpha: float,
):
    handles = []
    if alpha > 0 and subspace.layer_to_basis:
        layers = model_layers(agent._model)
        for layer, basis in subspace.layer_to_basis.items():
            handles.append(layers[layer].register_forward_hook(make_projection_hook(basis, alpha)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def score_text(text: str, gold: str) -> tuple[str | None, bool]:
    pred = normalize_answer(extract_gsm8k_answer(text))
    gold_norm = normalize_answer(gold)
    return pred, bool(pred and gold_norm and pred == gold_norm)


def run_reasoning(
    agent: ReasoningAgent,
    item: Dict,
    subspace: LanguageSubspace,
    alpha: float,
) -> Dict:
    task = AgentTask(
        task_id=f"mgsm_{item['lang']}_{item['idx']}_lr_alpha_{alpha}",
        query=item["question"],
        target_language=item["lang"],
    )
    with language_projection_hooks(agent, subspace, alpha):
        response = agent.process(task)
    pred, ok = score_text(response.output_text, item["gold"])
    return {
        "lang": item["lang"],
        "idx": item["idx"],
        "alpha": alpha,
        "correct": ok,
        "prediction": pred,
        "gold": normalize_answer(item["gold"]),
        "question": item["question"],
        "raw_prediction": response.output_text,
        "elapsed_ms": response.elapsed_ms,
        "n_reasoning_steps": response.metadata.get("n_reasoning_steps", ""),
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict], group_cols: Sequence[str]) -> List[Dict]:
    groups: Dict[tuple, List[Dict]] = {}
    for row in rows:
        key = tuple(row[col] for col in group_cols)
        groups.setdefault(key, []).append(row)
    out = []
    for key, group in sorted(groups.items()):
        correct = sum(1 for row in group if row["correct"])
        summary = {col: val for col, val in zip(group_cols, key)}
        summary.update({"accuracy": correct / len(group), "correct_count": correct, "total": len(group)})
        out.append(summary)
    return out


def choose_alpha(calibration_rows: List[Dict], lang: str, alphas: Sequence[float]) -> float:
    candidates = []
    for alpha in alphas:
        group = [r for r in calibration_rows if r["lang"] == lang and float(r["alpha"]) == alpha]
        if not group:
            continue
        correct = sum(1 for r in group if r["correct"])
        # tie-break toward the weakest intervention.
        candidates.append((correct / len(group), -abs(alpha), alpha))
    if not candidates:
        return 0.0
    return float(max(candidates)[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B")
    parser.add_argument("--languages", default="bn,de,en,es,fr,ja,ru,sw,te,th,zh")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--hidden_dim", type=int, default=2560)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--reasoning_layer", type=int, default=-2)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--calibration_start_idx", type=int, default=0)
    parser.add_argument("--calibration_examples", type=int, default=10)
    parser.add_argument("--eval_start_idx", type=int, default=10)
    parser.add_argument("--eval_examples", type=int, default=10)
    parser.add_argument("--layers", default="12-26")
    parser.add_argument("--n_components", type=int, default=8)
    parser.add_argument("--alphas", default="0,0.025,0.05,0.1,0.2")
    parser.add_argument("--out_dir", default="results/mgsm_reasoning_lr_disentangle")
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    langs = [x.strip().lower() for x in args.languages.split(",") if x.strip()]
    alphas = parse_floats(args.alphas)
    if 0.0 not in alphas:
        alphas = [0.0] + alphas

    model_safe = args.model_name.split("/")[-1]
    run_name = args.run_name or f"mgsm_calib{args.calibration_examples}_eval{args.eval_examples}_lr_disentangle"
    out_dir = Path(args.out_dir) / model_safe / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = make_reasoning_agent(args)
    agent._ensure_model_loaded()
    n_model_layers = len(model_layers(agent._model))
    layers = parse_layers(args.layers, n_model_layers)
    print(f"[setup] model_layers={n_model_layers} intervention_layers={layers}", flush=True)

    subspaces: Dict[str, LanguageSubspace] = {}
    calibration_rows: List[Dict] = []
    evaluation_rows: List[Dict] = []
    chosen_rows: List[Dict] = []

    for lang in langs:
        print(f"=== fitting {lang} ===", flush=True)
        subspace = fit_language_subspace(
            agent,
            lang,
            layers,
            args.calibration_start_idx,
            args.calibration_examples,
            args.n_components,
        )
        subspaces[lang] = subspace

        print(f"=== calibrating {lang} ===", flush=True)
        for item in iter_mgsm_slice(lang, args.calibration_start_idx, args.calibration_examples):
            print(f"  calib idx={item['idx']}", flush=True)
            for alpha in alphas:
                row = run_reasoning(agent, item, subspace, alpha)
                row["split"] = "calibration"
                row["chosen_alpha"] = ""
                calibration_rows.append(row)
                write_csv(out_dir / "calibration_examples.partial.csv", calibration_rows)

        chosen_alpha = choose_alpha(calibration_rows, lang, alphas)
        chosen_rows.append(
            {
                "lang": lang,
                "chosen_alpha": chosen_alpha,
                "calibration_examples": subspace.calibration_examples,
                "layers": args.layers,
                "resolved_layers": " ".join(str(x) for x in layers),
                "n_components": subspace.n_components,
            }
        )
        write_csv(out_dir / "chosen_interventions.csv", chosen_rows)

        print(f"=== evaluating {lang} chosen_alpha={chosen_alpha} ===", flush=True)
        for item in iter_mgsm_slice(lang, args.eval_start_idx, args.eval_examples):
            print(f"  eval idx={item['idx']}", flush=True)
            for label, alpha in (("baseline", 0.0), ("calibrated_lr_disentangle", chosen_alpha)):
                row = run_reasoning(agent, item, subspace, alpha)
                row["split"] = "eval"
                row["condition"] = label
                row["chosen_alpha"] = chosen_alpha
                evaluation_rows.append(row)
                write_csv(out_dir / "eval_examples.partial.csv", evaluation_rows)

    write_csv(out_dir / "calibration_examples.csv", calibration_rows)
    write_csv(out_dir / "eval_examples.csv", evaluation_rows)
    write_csv(out_dir / "chosen_interventions.csv", chosen_rows)
    write_csv(out_dir / "calibration_summary_by_lang_alpha.csv", summarize(calibration_rows, ["lang", "alpha"]))
    write_csv(out_dir / "eval_summary_by_lang_condition.csv", summarize(evaluation_rows, ["lang", "condition"]))
    write_csv(out_dir / "eval_summary_by_condition.csv", summarize(evaluation_rows, ["condition"]))

    meta = {
        "model_name": args.model_name,
        "languages": langs,
        "calibration_start_idx": args.calibration_start_idx,
        "calibration_examples": args.calibration_examples,
        "eval_start_idx": args.eval_start_idx,
        "eval_examples": args.eval_examples,
        "layers": args.layers,
        "resolved_layers": layers,
        "n_components": args.n_components,
        "alphas": alphas,
        "device": args.device,
        "dtype": args.dtype,
        "training_free": True,
        "notes": "No model weights or adapters are trained; only SVD language subspaces and per-language alpha choices are calibrated.",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
