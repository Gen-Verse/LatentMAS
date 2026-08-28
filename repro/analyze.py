#!/usr/bin/env python
"""Extract accuracy + token accounting from a run.py stdout log.

run.py's final JSON reports accuracy and wall clock but no token counts, and the
paper's headline claim ("~50-80% fewer tokens") is a token claim. This recovers
the missing half by re-tokenizing every agent's [Output] block with the model's
own tokenizer.

Usage:
    python repro/analyze.py repro/results/*.log
"""
import argparse
import json
import re
import sys
from collections import defaultdict

PROBLEM_RE = re.compile(r"^={20} Problem #(\d+) ={20}$", re.M)
AGENT_RE = re.compile(r"^----- Agent: (.+?) \((.+?)\) -----$", re.M)
SEP = "-" * 46


def parse_agent_block(block):
    """block starts right after the agent header."""
    out = ""
    inp = ""
    latent = None
    if "[Latent Steps]" in block:
        m = re.search(r"^\[Latent Steps\]\n(\d+)$", block, re.M)
        if m:
            latent = int(m.group(1))
    # the prompt actually fed to the model, between [To Tokenize] and the next marker
    m = re.search(r"^\[To Tokenize\]\n(.*?)(?=^\[Latent Steps\]$|^\[Output\]$)", block, re.S | re.M)
    if m:
        inp = m.group(1)
    m = re.search(r"^\[Output\]\n(.*?)(?:\n" + re.escape(SEP) + r"\n|\Z)", block, re.S | re.M)
    if m:
        out = m.group(1)
    return inp, out, latent


def parse_log(path):
    text = open(path, encoding="utf-8", errors="replace").read()

    final = None
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and '"accuracy"' in line:
            try:
                final = json.loads(line)
                break
            except json.JSONDecodeError:
                pass

    problems = []
    parts = PROBLEM_RE.split(text)
    # parts = [preamble, idx, body, idx, body, ...]
    for i in range(1, len(parts), 2):
        body = parts[i + 1]
        agents = []
        ah = list(AGENT_RE.finditer(body))
        for j, m in enumerate(ah):
            end = ah[j + 1].start() if j + 1 < len(ah) else len(body)
            inp, out, latent = parse_agent_block(body[m.end():end])
            agents.append({"name": m.group(1), "role": m.group(2),
                           "input": inp, "output": out, "latent_steps": latent})
        ok = None
        mr = re.search(r"^Result: Pred=.*\| OK=(True|False)$", body, re.M)
        if mr:
            ok = mr.group(1) == "True"
        problems.append({"agents": agents, "correct": ok})
    return final, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default="repro/results/summary.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    rows = []
    for path in args.logs:
        final, problems = parse_log(path)
        if not problems:
            print(f"[skip] no problems parsed from {path}", file=sys.stderr)
            continue

        per_role = defaultdict(int)
        total_gen = 0
        total_prompt = 0
        for p in problems:
            for a in p["agents"]:
                n = len(tok(a["output"], add_special_tokens=False)["input_ids"])
                per_role[a["role"]] += n
                total_gen += n
                total_prompt += len(tok(a.get("input", ""), add_special_tokens=False)["input_ids"])

        n_prob = len(problems)
        graded = [p for p in problems if p["correct"] is not None]
        acc = (sum(p["correct"] for p in graded) / len(graded)) if graded else float("nan")
        lat = next((a["latent_steps"] for p in problems for a in p["agents"]
                    if a["latent_steps"] is not None), None)

        rows.append({
            "log": path.split("/")[-1],
            "method": (final or {}).get("method", "?"),
            "model": (final or {}).get("model", "?"),
            "n": n_prob,
            "acc_parsed": acc,
            "acc_reported": (final or {}).get("accuracy"),
            "latent_steps": lat,
            "gen_tokens_total": total_gen,
            "gen_tokens_per_problem": total_gen / n_prob,
            "prompt_tokens_total": total_prompt,
            "prompt_tokens_per_problem": total_prompt / n_prob,
            "compute_tokens_per_problem": (total_gen + total_prompt) / n_prob,
            "per_role": dict(per_role),
            "wall_sec": (final or {}).get("total_time_sec"),
            "sec_per_problem": (final or {}).get("time_per_sample_sec"),
        })

    print(f"{'method':<12} {'ls':>3} {'n':>5} {'acc':>7} {'gen tok':>9} {'prompt tok':>11} "
          f"{'gen+prompt':>11} {'sec/prob':>9}")
    print("-" * 82)
    for r in rows:
        acc = r["acc_reported"] if r["acc_reported"] is not None else r["acc_parsed"]
        sp = r["sec_per_problem"]
        ls = r["latent_steps"]
        print(f"{r['method']:<12} {(str(ls) if ls is not None else '-'):>3} {r['n']:>5} {acc:>7.4f} "
              f"{r['gen_tokens_per_problem']:>9.1f} {r['prompt_tokens_per_problem']:>11.1f} "
              f"{r['compute_tokens_per_problem']:>11.1f} {(f'{sp:.1f}' if sp else '?'):>9}")
    print()
    for r in rows:
        roles = "  ".join(f"{k}={v}" for k, v in sorted(r["per_role"].items()))
        print(f"  {r['method']:<12} ls={r['latent_steps']} per-role generated: {roles}")

    if len(rows) > 1:
        base = next((r for r in rows if r["method"] == "text_mas"), rows[0])
        print(f"\nRelative to {base['method']}:")
        for r in rows:
            if r is base:
                continue
            dt = 100 * (1 - r["gen_tokens_per_problem"] / base["gen_tokens_per_problem"])
            dc = 100 * (1 - r["compute_tokens_per_problem"] / base["compute_tokens_per_problem"])
            line = f"  {r['method']:<12} generated {dt:+.1f}%   generated+prompt {dc:+.1f}%"
            if r["sec_per_problem"] and base["sec_per_problem"]:
                line += f"   speedup {base['sec_per_problem'] / r['sec_per_problem']:.2f}x"
            print(line)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
