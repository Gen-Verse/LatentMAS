import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"



def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_prompt(tokenizer, instruction, content):
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="juletxara/mgsm",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_language", type=str, default="TH")
    # Key used in instructions.json / hack_prefix.json, e.g. "EN", "DE", "FR"
    parser.add_argument("--think_language", type=str, default="TH")
    parser.add_argument(
        "--instructions_path",
        type=str,
        default="instructions.json",
    )
    parser.add_argument(
        "--hack_prefix_path",
        type=str,
        default="hack_prefix.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Path to the cache directory",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--example_idx",
        type=int,
        default=None,
        help="If set, only run this example index (debug). If None, run all.",
    )
    args = parser.parse_args()

    instructions = load_json(Path(args.instructions_path))
    hack_prefix = load_json(Path(args.hack_prefix_path))

    if args.prompt_language not in instructions:
        raise ValueError(f"{args.prompt_language} not found in instructions.json")
    if args.think_language not in hack_prefix:
        raise ValueError(f"{args.think_language} not found in hack_prefix.json")

    system_instruction, template = instructions[args.prompt_language]

    style = ''
    # -------- Load dataset (MGSM only for now) --------
    if "mgsm" in args.dataset_name.lower():
        ds = load_dataset(
            args.dataset_name,
            args.prompt_language.lower(),
            split="test",
            cache_dir=args.cache_dir,
        )
        args.max_tokens = 4096
        style = 'mgsm'
    elif "aime" in args.dataset_name.lower():
        ds = load_dataset(args.dataset_name, 
                          split=args.prompt_language.lower(), 
                          cache_dir=args.cache_dir)
        args.max_tokens = 16432
        style = 'aime'
    else:
        raise NotImplementedError(
            f"Dataset {args.dataset_name} not implemented yet (only MGSM for now)."
        )

    # If example_idx is specified, restrict to that single example (debug mode)
    if args.example_idx is not None:
        ds = ds.select([args.example_idx])

    # -------- Tokenizer --------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, trust_remote_code=True
    )

    # -------- Build all prompts (normal + hacked) --------
    all_prompts = []
    meta = []  # maps each prompt to (result_idx, "normal"/"hack")

    results = []

    if style == '':
        raise ValueError('The style cannot be empty.')
    for idx, ex in enumerate(ds):
        if style == 'mgsm':
            question = ex["question"]
            # MGSM numeric answer
            gold_answer = ex.get("answer_number", None)
        elif style == 'aime':
            question = ex["problem"]
            # AIME numeric answer
            gold_answer = ex.get("answer", None)

        content = template.format(question)
        base_prompt = make_prompt(tokenizer, system_instruction, content)

        # Manually add generation template with <think> tag
        if "deepseek" in args.model_name.lower():
            prompt = base_prompt + "<｜Assistant｜><think>\n"
        elif "qwen3" in args.model_name.lower():
            prompt = base_prompt + "<|im_start|>assistant\n<think>\n"
        elif "qwen2.5" in args.model_name.lower():
            # For Qwen2.5 we override with a simple system+user+assistant template
            prompt = (
                "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
                "You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                + content
                + "<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            raise ValueError(
                f"Unsupported model for vllm when manually adding <think> tag: {args.model_name}"
            )

        prompt_hack = prompt + f"{hack_prefix[args.think_language]}"

        # Prepare result entry for this example
        result_entry = {
            "idx": int(ex["idx"]) if "idx" in ex else int(idx),
            "question": question,
            "gold_answer": gold_answer,
            "normal": {},
            "hack": {},
        }
        results.append(result_entry)
        result_idx = len(results) - 1

        # Add prompts to global list + meta mapping
        all_prompts.append(prompt)
        meta.append({"result_idx": result_idx, "kind": "normal"})

        all_prompts.append(prompt_hack)
        meta.append({"result_idx": result_idx, "kind": "hack"})

    print(f"Total examples: {len(results)}")
    print(f"Total prompts to generate: {len(all_prompts)}")

    # -------- vLLM generation --------
    print("\nLoading model with vLLM...")

    if "deepseek" in args.model_name.lower():
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )

    extra_kw = {"download_dir": args.cache_dir}
    model = LLM(
        model=args.model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.90,
        dtype=torch.bfloat16,
        # distributed_executor_backend="mp",
        trust_remote_code=True,
        max_num_seqs=100,
        max_model_len=args.max_tokens,
        seed=args.seed,
        disable_custom_all_reduce=True,
        **extra_kw,
    )

    responses = model.generate(all_prompts, sampling_params, use_tqdm=True)

    # -------- Attach responses back to results --------
    for resp, m in zip(responses, meta):
        result_idx = m["result_idx"]
        kind = m["kind"]
        prompt_text = resp.prompt
        response_text = resp.outputs[0].text

        results[result_idx][kind] = {
            "prompt": prompt_text,
            "response": response_text,
        }

    # -------- Save results to JSON --------
    dataset_base = args.dataset_name.split("/")[-1]
    model_base = args.model_name.split("/")[-1]
    lang_suffix = args.prompt_language.lower()

    out_dir = Path("results") / dataset_base / model_base
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{lang_suffix}_result.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
