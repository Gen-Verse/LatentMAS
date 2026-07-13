#!/usr/bin/env python
"""Run an AfriMGSM planner/solver/critic agentic pilot.

This standalone runner is intentionally separate from the CVAE coordination
pipeline. The current CVAE role library is translation/reasoning/safety, which
is not the cognitive decomposition we want to test for math. Here we compare:

1. single_solver
2. text_planner_solver_critic: plan -> solve -> critique -> revise, text handoff
3. latent_planner_solver_critic: plan -> solve -> critique -> revise, latent handoff

All roles share one loaded backbone to avoid tripling GPU memory. The role
distinction comes from prompts and, in the latent condition, captured hidden
states passed into the next role's prefill.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from latent_coordination.agents.base_agent import AgentConfig, AgentTask, BaseAgent
from latent_coordination.eval.correctness import load_afrimgsm_tasks, load_mgsm_tasks, score_mgsm


AFRIMGSM_LANGS = ["am", "ee", "ha", "ig", "rw", "ln", "lg", "om", "sn", "st", "sw", "tw", "wo", "xh", "yo", "zu"]
MGSM_LANGS = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]


def build_prompt(role: str, question: str, context: str = "") -> str:
    if role == "planner":
        return (
            "You are a math planning agent. Read the problem and produce a short, "
            "number-focused plan. Do not solve fully yet. Keep all quantities and "
            "operations explicit.\n\n"
            f"Problem:\n{question}\n\nPlan:"
        )
    if role == "solver":
        return (
            "You are a careful math solver. Solve the problem exactly. Use the "
            "available context or latent signal if provided. End with a final "
            "numeric answer in the form: Answer: <number>.\n\n"
            f"Problem:\n{question}\n"
            f"{context}\n\nSolution:"
        )
    if role == "critic":
        return (
            "You are a math critic. Check whether the proposed solution uses the "
            "right quantities and arithmetic. If wrong, state the correction. "
            "End with the corrected numeric answer in the form: Corrected answer: <number>.\n\n"
            f"Problem:\n{question}\n"
            f"{context}\n\nCritique:"
        )
    if role == "reviser":
        return (
            "You are the final math solver. Use the critique or latent feedback to "
            "produce the final answer. End with exactly: Answer: <number>.\n\n"
            f"Problem:\n{question}\n"
            f"{context}\n\nFinal solution:"
        )
    raise ValueError(f"unknown role: {role}")


class PromptRoleAgent(BaseAgent):
    def __init__(self, config: AgentConfig, prompt_role: str, capture_layer: int) -> None:
        super().__init__(config)
        self.prompt_role = prompt_role
        self.capture_layer = capture_layer

    def process(self, task: AgentTask):  # not used; run_role gives us prompt control.
        raise NotImplementedError("Use run_role() in this standalone runner.")

    def run_role(
        self,
        question: str,
        context: str = "",
        latent_state: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
    ):
        prompt = build_prompt(self.prompt_role, question, context)
        t0 = time.perf_counter()
        text, latent = self.generate_and_capture(
            prompt,
            latent_state=latent_state,
            injection_layer=self.capture_layer,
            capture_layer=self.capture_layer,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            do_sample=False,
        )
        return text.strip(), latent, (time.perf_counter() - t0) * 1000.0


@dataclass
class TaskItem:
    task_id: str
    lang: str
    idx: int
    question: str
    answer: float


def load_items(benchmark: str, languages: Iterable[str], n: int, start_idx: int = 0) -> List[TaskItem]:
    items: List[TaskItem] = []
    loader = load_afrimgsm_tasks if benchmark == "afrimgsm" else load_mgsm_tasks
    for lang in languages:
        requested_n = None if n < 0 else start_idx + n
        tasks = loader(language=lang, n=requested_n)
        selected = tasks[start_idx:] if n < 0 else tasks[start_idx:start_idx + n]
        for idx, t in enumerate(selected, start=start_idx):
            items.append(TaskItem(
                task_id=f"{benchmark}_{lang}_{idx}",
                lang=lang,
                idx=idx,
                question=t.get("question") or t.get("query") or "",
                answer=float(t["answer"]),
            ))
    return items


def normalize_modes(value: str) -> List[str]:
    if value == "all":
        return ["single_solver", "text_planner_solver_critic", "latent_planner_solver_critic"]
    return [m.strip() for m in value.split(",") if m.strip()]


def make_agents(args) -> Dict[str, PromptRoleAgent]:
    base_cfg = AgentConfig(
        agent_id="agent_solver",
        model_id=args.model_name,
        role="reasoning",
        device=args.device,
        hidden_dim=args.hidden_dim,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        latent_transfer_layer=args.latent_transfer_layer,
        max_time_s=args.max_time_s,
    )
    solver = PromptRoleAgent(base_cfg, "solver", args.latent_transfer_layer)
    solver._ensure_model_loaded()

    agents = {"solver": solver}
    for role in ["planner", "critic", "reviser"]:
        cfg = AgentConfig(
            agent_id=f"agent_{role}",
            model_id=args.model_name,
            role="reasoning",
            device=args.device,
            hidden_dim=args.hidden_dim,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
            latent_transfer_layer=args.latent_transfer_layer,
            max_time_s=args.max_time_s,
        )
        agent = PromptRoleAgent(cfg, role, args.latent_transfer_layer)
        agent._model = solver._model
        agent._tokenizer = solver._tokenizer
        agent._is_loaded = True
        agents[role] = agent
    return agents


def run_single_solver(item: TaskItem, agents: Dict[str, PromptRoleAgent], args) -> Dict:
    text, _, elapsed = agents["solver"].run_role(item.question, max_new_tokens=args.max_new_tokens)
    scored = score_mgsm(text, item.answer)
    return {
        "output_text": text,
        "prediction": scored.predicted,
        "correct": scored.is_correct,
        "elapsed_ms": elapsed,
        "planner_text": "",
        "solver_text": text,
        "critic_text": "",
        "final_text": text,
    }


def run_text_psc(item: TaskItem, agents: Dict[str, PromptRoleAgent], args) -> Dict:
    plan, _, t_plan = agents["planner"].run_role(item.question, max_new_tokens=args.plan_tokens)
    sol_ctx = f"\nPlanner output:\n{plan}"
    sol, _, t_sol = agents["solver"].run_role(item.question, sol_ctx, max_new_tokens=args.max_new_tokens)
    crit_ctx = f"\nPlanner output:\n{plan}\n\nProposed solution:\n{sol}"
    crit, _, t_crit = agents["critic"].run_role(item.question, crit_ctx, max_new_tokens=args.critic_tokens)
    rev_ctx = f"\nPlanner output:\n{plan}\n\nInitial solution:\n{sol}\n\nCritique:\n{crit}"
    final, _, t_final = agents["reviser"].run_role(item.question, rev_ctx, max_new_tokens=args.max_new_tokens)
    scored = score_mgsm(final, item.answer)
    return {
        "output_text": final,
        "prediction": scored.predicted,
        "correct": scored.is_correct,
        "elapsed_ms": t_plan + t_sol + t_crit + t_final,
        "planner_text": plan,
        "solver_text": sol,
        "critic_text": crit,
        "final_text": final,
    }


def run_latent_psc(item: TaskItem, agents: Dict[str, PromptRoleAgent], args) -> Dict:
    plan, z_plan, t_plan = agents["planner"].run_role(item.question, max_new_tokens=args.plan_tokens)
    sol_ctx = "\nA latent planning signal is available. Use it to solve, but do not quote or translate it."
    sol, z_sol, t_sol = agents["solver"].run_role(
        item.question, sol_ctx, latent_state=z_plan, max_new_tokens=args.max_new_tokens
    )
    crit_ctx = "\nA latent solution signal is available. Check the arithmetic and infer the final answer."
    crit, z_crit, t_crit = agents["critic"].run_role(
        item.question, crit_ctx, latent_state=z_sol, max_new_tokens=args.critic_tokens
    )
    rev_ctx = "\nA latent critique signal is available. Produce the corrected final answer."
    final, _, t_final = agents["reviser"].run_role(
        item.question, rev_ctx, latent_state=z_crit, max_new_tokens=args.max_new_tokens
    )
    scored = score_mgsm(final, item.answer)
    return {
        "output_text": final,
        "prediction": scored.predicted,
        "correct": scored.is_correct,
        "elapsed_ms": t_plan + t_sol + t_crit + t_final,
        "planner_text": plan,
        "solver_text": sol,
        "critic_text": crit,
        "final_text": final,
    }


def write_outputs(rows: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "examples.csv", index=False)

    summary = (
        df.groupby("mode")["correct"]
        .agg(accuracy="mean", correct="sum", total="count")
        .reset_index()
    )
    summary.to_csv(out_dir / "summary_by_mode.csv", index=False)

    lang_summary = (
        df.groupby(["mode", "lang"])["correct"]
        .agg(accuracy="mean", correct="sum", total="count")
        .reset_index()
    )
    lang_summary.to_csv(out_dir / "summary_by_mode_lang.csv", index=False)

    wide = df.pivot_table(index=["task_id", "lang", "idx"], columns="mode", values="correct", aggfunc="first").reset_index()
    wide.to_csv(out_dir / "overlap_wide.csv", index=False)

    meta = {
        "summary_by_mode": summary.to_dict(orient="records"),
        "summary_by_mode_lang": lang_summary.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="CohereLabs/aya-expanse-8b")
    ap.add_argument("--benchmark", choices=["afrimgsm", "mgsm"], default="afrimgsm")
    ap.add_argument("--languages", default=None)
    ap.add_argument("--max_examples", type=int, default=10)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--modes", default="all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--hidden_dim", type=int, default=4096)
    ap.add_argument("--load_in_8bit", action="store_true", default=True)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--plan_tokens", type=int, default=192)
    ap.add_argument("--critic_tokens", type=int, default=256)
    ap.add_argument("--latent_transfer_layer", type=int, default=-4)
    ap.add_argument("--max_time_s", type=float, default=180.0)
    ap.add_argument("--out_dir", default="results/afrimgsm_planner_solver_critic")
    ap.add_argument("--run_name", default="aya_first10_planner_solver_critic")
    ap.add_argument("--checkpoint_every", type=int, default=10)
    args = ap.parse_args()

    default_langs = MGSM_LANGS if args.benchmark == "mgsm" else AFRIMGSM_LANGS
    languages = [x.strip() for x in (args.languages or ",".join(default_langs)).split(",") if x.strip()]
    modes = normalize_modes(args.modes)
    items = load_items(args.benchmark, languages, args.max_examples, args.start_idx)
    agents = make_agents(args)

    out_dir = Path(args.out_dir) / args.run_name
    rows: List[Dict] = []
    for n, item in enumerate(items, start=1):
        print(f"=== {item.lang} idx={item.idx} ===", flush=True)
        for mode in modes:
            if mode == "single_solver":
                result = run_single_solver(item, agents, args)
            elif mode == "text_planner_solver_critic":
                result = run_text_psc(item, agents, args)
            elif mode == "latent_planner_solver_critic":
                result = run_latent_psc(item, agents, args)
            else:
                raise ValueError(f"unknown mode: {mode}")
            row = {
                "mode": mode,
                "task_id": item.task_id,
                "lang": item.lang,
                "idx": item.idx,
                "gold": item.answer,
                "question": item.question,
                **result,
            }
            print(f"  {mode}: correct={row['correct']} pred={row['prediction']} gold={row['gold']}", flush=True)
            rows.append(row)
        if args.checkpoint_every > 0 and n % args.checkpoint_every == 0:
            write_outputs(rows, out_dir)
            print(f"  [checkpoint] wrote {len(rows)} rows to {out_dir}", flush=True)

    write_outputs(rows, out_dir)
    print("[OK] wrote", out_dir, flush=True)
    print(pd.read_csv(out_dir / "summary_by_mode.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
