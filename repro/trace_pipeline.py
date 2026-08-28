#!/usr/bin/env python
"""Instrumented single-problem trace of the LatentMAS pipeline.

run.py prints prompts and outputs, but not the thing that actually matters:
the KV cache. This wraps ModelWrapper.generate_latent_batch /
generate_text_batch and prints, per agent, how the cache grows and where the
tokens do (and do not) get spent.

Usage:
  python repro/trace_pipeline.py --method latent_mas --latent_steps 10
  python repro/trace_pipeline.py --method latent_mas --latent_steps 0   # the released-log setting
  python repro/trace_pipeline.py --method text_mas
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import models
from models import ModelWrapper, _past_length
from methods.latent_mas import LatentMASMethod
from methods.text_mas import TextMASMethod
from methods.baseline import BaselineMethod
from data import load_gsm8k, load_mbppplus
from utils import set_seed


def hr(c="="):
    print(c * 78)


def instrument(model: ModelWrapper):
    """Wrap the two generation entry points to narrate cache state."""
    state = {"agent": 0}

    orig_latent = model.generate_latent_batch
    orig_text = model.generate_text_batch

    def generate_latent_batch(input_ids, attention_mask=None, *, latent_steps, past_key_values=None):
        state["agent"] += 1
        n_prompt = int(attention_mask.sum(dim=1)[0].item()) if attention_mask is not None else input_ids.shape[1]
        before = _past_length(past_key_values)
        past = orig_latent(
            input_ids, attention_mask=attention_mask,
            latent_steps=latent_steps, past_key_values=past_key_values,
        )
        after = _past_length(past)
        print(f"  KV cache in            : {before:6d} positions")
        print(f"  + prompt forward-passed: {n_prompt:6d} tokens   (prefill on top of the inherited cache)")
        print(f"  + latent steps         : {latent_steps:6d} positions (embedding-space, NO sampling, NO detokenization)")
        print(f"  KV cache out           : {after:6d} positions  (delta {after - before:+d})")
        print(f"  TEXT TOKENS EMITTED    : {0:6d}   <-- this agent produces no text at all")
        return past

    def generate_text_batch(input_ids, attention_mask=None, **kw):
        state["agent"] += 1
        past_kv = kw.get("past_key_values")
        inherited = _past_length(past_kv)
        n_prompt = int(attention_mask.sum(dim=1)[0].item()) if attention_mask is not None else input_ids.shape[1]
        gens, out_past = orig_text(input_ids, attention_mask=attention_mask, **kw)
        n_gen = len(model.tokenizer(gens[0], add_special_tokens=False)["input_ids"])
        print(f"  inherited KV cache     : {inherited:6d} positions "
              f"{'<-- the ENTIRE inter-agent channel' if inherited else '<-- NONE (cache was discarded!)'}")
        print(f"  + own prompt           : {n_prompt:6d} tokens")
        print(f"  => attends over        : {inherited + n_prompt:6d} positions before decoding starts")
        print(f"  TEXT TOKENS EMITTED    : {n_gen:6d}   <-- the only text the whole system produces")
        return gens, out_past

    model.generate_latent_batch = generate_latent_batch
    model.generate_text_batch = generate_text_batch
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="latent_mas", choices=["latent_mas", "text_mas", "baseline"])
    ap.add_argument("--model_name", default="Qwen/Qwen3-4B")
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k", "mbppplus"])
    ap.add_argument("--prompt", default="sequential", choices=["sequential", "hierarchical"])
    ap.add_argument("--latent_steps", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--generate_bs", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device2", default="cuda:1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--think", action="store_true", default=True)
    ap.add_argument("--latent_space_realign", action="store_true")
    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--text_mas_context_length", type=int, default=-1)
    ap.add_argument("--problem", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    print(f"loading {args.model_name} ...", flush=True)
    model = ModelWrapper(args.model_name, torch.device(args.device), use_vllm=False, args=args)

    item = list(load_gsm8k(split="test") if args.task == "gsm8k" else load_mbppplus(split="test"))[args.problem]

    hr()
    print(f"METHOD={args.method}  TASK={args.task}  PROMPT={args.prompt}  latent_steps={args.latent_steps}")
    hr()
    print("QUESTION:")
    print("  " + item["question"].strip()[:400].replace("\n", "\n  "))
    print(f"\nGOLD: {item['gold']}")
    hr()

    if args.method == "latent_mas":
        method = LatentMASMethod(model, latent_steps=args.latent_steps,
                                 judger_max_new_tokens=args.max_new_tokens,
                                 temperature=args.temperature, top_p=args.top_p,
                                 generate_bs=1, args=args)
    elif args.method == "text_mas":
        method = TextMASMethod(model, max_new_tokens_each=args.max_new_tokens,
                               temperature=args.temperature, top_p=args.top_p,
                               generate_bs=1, args=args)
    else:
        method = BaselineMethod(model, max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature, top_p=args.top_p,
                                generate_bs=1, use_vllm=False, args=args)

    instrument(model)

    # narrate agent boundaries by patching the agent list iteration point
    if hasattr(method, "agents"):
        for a in method.agents:
            print(f"  agent in pipeline: {a.name} ({a.role})")
        hr("-")

    import time
    t0 = time.time()
    res = method.run_batch([item])[0]
    dt = time.time() - t0

    hr()
    tok = model.tokenizer
    total_gen = 0
    for a in res.get("agents", []):
        n = len(tok(a.get("output", ""), add_special_tokens=False)["input_ids"]) if a.get("output") else 0
        total_gen += n
        print(f"  {a['name']:<9} ({a['role']:<8}) generated {n:6d} text tokens")
    print(f"  {'TOTAL':<9} {'':<11} generated {total_gen:6d} text tokens   in {dt:.1f}s")
    hr()
    print(f"PREDICTION: {res.get('prediction')}   GOLD: {res.get('gold')}   CORRECT: {res.get('correct')}")
    hr()


if __name__ == "__main__":
    main()
