import argparse
import json
from typing import Dict, List, Tuple

from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa
)
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.text_mas import TextMASMethod
from models import ModelWrapper
from utils import auto_device, set_seed
import time


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct


def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    if args.method == "latent_mas" and args.use_vllm: 
        results = method.run_batch_vllm(current_batch) 
    else:
        results = method.run_batch(current_batch)
    if len(results) > remaining:
        results = results[:remaining]
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def main():
    parser = argparse.ArgumentParser()

    # core args for experiments
    parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas"], required=True)
    parser.add_argument("--model_name", type=str, required=True, #choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-4B", "Qwen/Qwen3-14B"]
    )
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--task", choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus", 'humanevalplus', 'medqa', "custom"], default="gsm8k")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential")
    parser.add_argument("--custom_prompt_file", type=str, default=None, help="Path to custom prompt template(s). Supports baseline, text_mas, and latent_mas. Accepts plain text (baseline) or JSON with role-specific fields.")
    parser.add_argument("--custom_question", type=str, default=None, help="Custom question text when --task custom")
    parser.add_argument("--custom_question_file", type=str, default=None, help="Path to a file containing a custom question when --task custom")
    parser.add_argument("--custom_gold", type=str, default=None, help="Optional gold answer for custom task (string)")

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=10000)
    parser.add_argument("--latent_steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=20)
    parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
    parser.add_argument("--think", nargs="?", const="<think>\n", default=None, help="Manually add think token in the prompt for LatentMAS. Use --think for default '<think>\\n<brainstorm>\\n' or --think 'custom' for custom tokens")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--first_agent_text", action="store_true", help="First agent generates text instead of latent (useful for graph/structured reasoning)")
    parser.add_argument("--do_not_enforce_qwen", action="store_true", help="Disable Qwen-specific system message and model name validation (use non-Qwen models)")
    parser.add_argument("--seed", type=int, default=42)

    # for vllm support
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
    parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching in vLLM for latent_mas")
    parser.add_argument("--use_second_HF_model", action="store_true", help="Use a second HF model for latent generation in latent_mas")
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")

    args = parser.parse_args()
    
    args.custom_prompts = None
    args.custom_prompt_text = None  # kept for baseline compatibility
    args.custom_agents = None
    if args.custom_prompt_file:
        with open(args.custom_prompt_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
        if not raw_text.strip():
            raise ValueError("Custom prompt file is empty.")
        try:
            parsed = json.loads(raw_text)
            args.custom_prompts = parsed
            if isinstance(parsed, dict):
                # Optional baseline override keys
                args.custom_prompt_text = parsed.get("baseline") or parsed.get("user")
                # Parse custom agents if provided
                if "agents" in parsed and isinstance(parsed["agents"], list):
                    from methods import Agent
                    args.custom_agents = [
                        Agent(name=agent_dict.get("name", ""), role=agent_dict.get("role", ""))
                        for agent_dict in parsed["agents"]
                        if isinstance(agent_dict, dict) and "name" in agent_dict and "role" in agent_dict
                    ]
        except json.JSONDecodeError:
            args.custom_prompts = raw_text
            args.custom_prompt_text = raw_text

    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True 
        args.enable_prefix_caching = True
    
    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    
    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == 'latent_mas':
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs, 
            args=args,
        )

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []

    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split)
    elif args.task == "aime2024":
        dataset_iter = load_aime2024(split="train")
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split='train')
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split='test')
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split='test')
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split='test')
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split='test')
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split='test')
    elif args.task == "medqa":
        dataset_iter = load_medqa(split='test')
    elif args.task == "custom":
        if args.custom_question is None and args.custom_question_file is None:
            raise ValueError("For --task custom, provide --custom_question or --custom_question_file.")
        if args.custom_question_file:
            with open(args.custom_question_file, "r", encoding="utf-8") as f:
                custom_question = f.read()
        else:
            custom_question = args.custom_question
        gold = args.custom_gold.strip().lower() if args.custom_gold else ""
        dataset_iter = [
            {
                "question": custom_question.strip(),
                "solution": gold,
                "gold": gold,
            }
        ]
        if args.max_samples == 100:  # unchanged default
            args.max_samples = 1
    else:
        raise ValueError(f'no {args.task} support')
    
    if args.max_samples == -1:
        dataset_iter = list(dataset_iter)  
        args.max_samples = len(dataset_iter)

    progress = tqdm(total=args.max_samples)

    for item in dataset_iter:
        if processed >= args.max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                args.max_samples,
                args,
            )
            batch = []
            if processed >= args.max_samples:
                break

    if batch and processed < args.max_samples:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            progress,
            max_samples=args.max_samples,
            args=args,
        )
    progress.close()
    
    total_time = time.time() - start_time

    acc, correct = evaluate(preds)
    print(
        json.dumps(
            {
                "method": args.method,
                "model": args.model_name,
                "split": args.split,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "accuracy": acc,
                "correct": correct,
                "total_time_sec": round(total_time,4),
                "time_per_sample_sec": round(total_time / args.max_samples, 4),
            },
            ensure_ascii=False,
        )
    )



if __name__ == "__main__":
    main()
