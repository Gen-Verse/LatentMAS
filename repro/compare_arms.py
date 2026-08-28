#!/usr/bin/env python
"""Compare method arms on the prefix of problems ALL of them have finished.

Why not repro/analyze.py: that one parses a finished stdout log. These runs are
preempted constantly, so a "finished log" is rare and a partial log is the normal
case. The --checkpoint JSONL is written after every batch and already carries the
full agent trace (`input_tokens` per agent, plus each agent's `output` text), so
every number the paper compares on can be recovered from it *mid-run*.

The arms are only comparable over the same problems, so everything is reported on
the common prefix -- run.py's resume is a prefix skip, so record i is problem i in
every arm.

Wall-clock comes from `_elapsed_sec`, stamped on the last record of each batch and
carried across preemption by run.py's resume. It is only meaningful if every arm
ran on the same GPU model (all of these: L40S).

Usage:
    python repro/compare_arms.py                        # every mbppplus/4B arm
    python repro/compare_arms.py --glob 'repro/results/*gsm8k*.ckpt.jsonl'
"""
import argparse
import ast
import glob
import io
import json
import os
import re
from collections import defaultdict

ROLES = ["planner", "critic", "refiner", "judger", "singleagent"]


def load(path):
    """Records + the elapsed-seconds mark at each batch boundary."""
    recs, marks = [], []
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                break  # a preemption kill can truncate the final line
            el = rec.pop("_elapsed_sec", None)
            recs.append(rec)
            if el is not None:
                marks.append((len(recs), el))
    return recs, marks


def agents_of(rec):
    a = rec.get("agents")
    if isinstance(a, str):
        # older records round-tripped the trace through repr()
        try:
            a = ast.literal_eval(a)
        except (ValueError, SyntaxError):
            return []
    return a or []


def elapsed_at(marks, n):
    """Wall-clock after the last completed batch that ends at or before n."""
    best = None
    for count, el in marks:
        if count <= n:
            best = (count, el)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="repro/results/*mbppplus*.ckpt.jsonl")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default="repro/results/arm_comparison.json")
    ap.add_argument("--no-tokens", action="store_true",
                    help="skip loading the tokenizer (accuracy + timing only)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"no checkpoints match {args.glob}")

    arms = {}
    for p in paths:
        name = os.path.basename(p).replace(".ckpt.jsonl", "")
        recs, marks = load(p)
        if recs:
            arms[name] = (recs, marks, p)

    n = min(len(r) for r, _, _ in arms.values())
    print(f"{len(arms)} arms; common prefix = {n} problems "
          f"(per-arm: {', '.join(str(len(r)) for r, _, _ in arms.values())})\n")

    tok = None
    if not args.no_tokens:
        os.environ.setdefault("HF_HOME", f"/pub/{os.environ.get('USER','')}/hf")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    rows = []
    for name, (recs, marks, path) in sorted(arms.items()):
        pre = recs[:n]
        correct = sum(1 for r in pre if str(r.get("correct")) == "True")
        row = {"arm": name, "n": n, "correct": correct, "accuracy": correct / n}

        mark = elapsed_at(marks, n)
        if mark:
            row["timed_n"], row["elapsed_sec"] = mark
            row["sec_per_problem"] = mark[1] / mark[0]

        if tok is not None:
            gen = defaultdict(int)
            prompt = defaultdict(int)
            lat = defaultdict(int)
            for r in pre:
                for a in agents_of(r):
                    role = a.get("role", "?")
                    out = a.get("output") or ""
                    # the latent agents emit no text at all -- that is the claim
                    gen[role] += len(tok(out, add_special_tokens=False)["input_ids"]) if out else 0
                    lat[role] += a.get("latent_steps") or 0
                    # `input_tokens` / `input_ids` are the token LISTS, not counts
                    ids = a.get("input_ids") or a.get("input_tokens") or []
                    prompt[role] += len(ids)
            row["gen_tokens_per_problem"] = {k: round(v / n, 1) for k, v in gen.items()}
            row["gen_tokens_total_per_problem"] = round(sum(gen.values()) / n, 1)
            row["prompt_tokens_per_problem"] = {k: round(v / n, 1) for k, v in prompt.items()}
            row["prompt_tokens_total_per_problem"] = round(sum(prompt.values()) / n, 1)
            # latent steps are KV entries, not tokens: the paper's Token column
            # never counts them, so track them as their own third currency.
            row["latent_steps_per_problem"] = round(sum(lat.values()) / n, 1)
        rows.append(row)

    w = max(len(r["arm"]) for r in rows)
    head = (f"{'arm'.ljust(w)}  {'acc':>7}  {'sec/prob':>9}  {'gen tok':>8}  "
            f"{'prompt tok':>10}  {'latent':>6}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['arm'].ljust(w)}  {r['accuracy']*100:6.2f}%  "
              f"{r.get('sec_per_problem', float('nan')):9.1f}  "
              f"{r.get('gen_tokens_total_per_problem', float('nan')):8.1f}  "
              f"{r.get('prompt_tokens_total_per_problem', float('nan')):10.1f}  "
              f"{r.get('latent_steps_per_problem', float('nan')):6.1f}")

    if tok is not None:
        print("\ngenerated tokens per problem, by agent role "
              "(latent agents should be ~0 -- that is the mechanism):")
        for r in rows:
            parts = " ".join(f"{k}={v}" for k, v in sorted(r["gen_tokens_per_problem"].items())
                             if v > 0)
            print(f"  {r['arm'].ljust(w)}  {parts or '(none)'}")

    with io.open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"common_prefix": n, "arms": rows}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
