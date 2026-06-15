import argparse
import json
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

from helper import build_truncated_think_block

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Lineesha Kamana"
__email__ = "lpk5305@psu.edu, hthakur@uccs.edu"
__status__ = "prototype"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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

    parser.add_argument(
        "--languages",
        type=str,
        default="EN",
        help="Comma-separated list of language keys (e.g. 'EN,FR,DE,ZH').",
    )

    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10,
        help="Maximum number of tokens to generate for the ANSWER phase.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["normal", "hack", "both"],
        default="both",
        help="Which original traces to truncate and run: normal, hack, or both.",
    )
    parser.add_argument(
        "--input_results_root",
        type=str,
        default="results",
        help="Root directory where original full CoT results are stored.",
    )
    parser.add_argument(
        "--output_results_root",
        type=str,
        default="results_trunc",
        help="Root directory to store truncation results.",
    )
    parser.add_argument(
        "--example_idx",
        type=int,
        default=None,
        help="If set, only run this example index (debug). If None, run all.",
    )

    args = parser.parse_args()

    dataset_base = args.dataset_name.split("/")[-1]
    model_base = args.model_name.split("/")[-1]

    # --------------------------------------------------------------
    # Truncation ratios
    # --------------------------------------------------------------
    if "mgsm" in args.dataset_name.lower():
        # 0%, 10%, 20%, ..., 90%, 100%
        trunc_ratios = [0.0] + [k / 10.0 for k in range(1, 10)] + [1.0]
    elif "aime" in args.dataset_name.lower():
        # 0%, 5%, 10%, ..., 95%, 100%
        trunc_ratios = [0.0] + [k * 0.05 for k in range(1, 20)] + [1.0]
    else:
        raise NotImplementedError


    # --------------------------------------------------------------
    # Initialize vLLM once (with 10 samples per prompt)
    # --------------------------------------------------------------
    if "deepseek" in args.model_name.lower():
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=args.max_tokens,
            seed=args.seed,
            n=10,  # 10 samples per prompt
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
            n=10,  # 10 samples per prompt
        )

    print("\nLoading model once for all languages…")
    model = LLM(
        model=args.model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.90,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        max_num_seqs=100,
        max_model_len=16500,  # long enough to hold original CoT in prompt
        seed=args.seed,
        disable_custom_all_reduce=True,
        download_dir=args.cache_dir,
    )

    # --------------------------------------------------------------
    # Loop over languages
    # --------------------------------------------------------------
    lang_list = [l.strip() for l in args.languages.split(",") if l.strip()]
    print(f"Languages to run: {lang_list}")

    input_root = Path(args.input_results_root)
    output_root = Path(args.output_results_root)

    modes_to_use = ["normal", "hack"] if args.mode == "both" else [args.mode]

    for lang in lang_list:
        lang_suffix = lang.lower()
        think_lang = lang  # thinking language == prompt language

        # ----------------------------
        # Skip if output exists
        # ----------------------------
        out_dir = output_root / dataset_base / model_base
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.example_idx is not None:
            out_path = out_dir / f"{lang_suffix}_trunc_idx{args.example_idx}.json"
        else:
            out_path = out_dir / f"{lang_suffix}_trunc.json"

        if out_path.exists():
            print(f"[SKIP] Output already exists for lang={lang}: {out_path}")
            continue

        # ----------------------------
        # Load original full-CoT results
        # ----------------------------
        orig_path = (
            input_root / dataset_base / model_base / f"{lang_suffix}_result.json"
        )

        if not orig_path.exists():
            print(f"[WARN] No full-CoT original file for lang={lang}: {orig_path}")
            continue

        orig_results = load_json(orig_path)
        print(f"\n=== Language {lang} === Loaded {len(orig_results)} examples")

        if args.example_idx is not None:
            if args.example_idx < 0 or args.example_idx >= len(orig_results):
                raise IndexError(
                    f"example_idx {args.example_idx} out of range (0..{len(orig_results)-1})"
                )
            orig_results = [orig_results[args.example_idx]]

        # ----------------------------
        # Build prompts (forward + backward)
        # ----------------------------
        all_prompts = []
        meta = []  # each: {result_idx, kind, direction, ratio_key}
        trunc_results = []

        directions = [
            ("forward", False),   # reverse=False
            ("backward", True),  # reverse=True
        ]

        for idx, entry in enumerate(orig_results):
            q = entry.get("question", "")
            gold = entry.get("gold_answer", None)
            ex_idx = int(entry.get("idx", idx))

            trunc_entry = {
                "idx": ex_idx,
                "question": q,
                "gold_answer": gold,
                "normal": {
                    "forward": {},
                    "backward": {},
                },
                "hack": {
                    "forward": {},
                    "backward": {},
                },
            }
            trunc_results.append(trunc_entry)
            result_idx = len(trunc_results) - 1

            for kind in modes_to_use:
                obj = entry.get(kind, {})
                if not obj:
                    continue

                base_prompt = obj.get("prompt", "")
                base_response = obj.get("response", "")

                for direction_name, reverse_flag in directions:
                    for r in trunc_ratios:
                        ratio_key = f"{int(round(r * 100))}%"

                        # Build the truncated think block (forward / backward)
                        trunc_block = build_truncated_think_block(
                            base_response,
                            think_lang,
                            r,
                            reverse=reverse_flag,
                        )

                        # New prompt = original prompt + truncated block
                        full_prompt = base_prompt + trunc_block

                        all_prompts.append(full_prompt)
                        meta.append(
                            {
                                "result_idx": result_idx,
                                "kind": kind,
                                "direction": direction_name,  # "forward" / "backward"
                                "ratio_key": ratio_key,
                            }
                        )

        print(f"Total truncated prompts for lang={lang}: {len(all_prompts)}")

        if not all_prompts:
            print(f"[WARN] No prompts constructed for lang={lang}. Skipping.")
            continue

        # ----------------------------
        # Generate answers (10 samples per prompt)
        # ----------------------------
        responses = model.generate(all_prompts, sampling_params, use_tqdm=True)

        for resp, m in zip(responses, meta):
            ridx = m["result_idx"]
            kind = m["kind"]
            direction = m["direction"]
            ratio = m["ratio_key"]

            # Collect all n sampled outputs as a list
            texts = [o.text for o in resp.outputs]

            trunc_results[ridx][kind][direction][ratio] = {
                "prompt": resp.prompt,
                "response": texts,  # list[str]: [answer_1, answer_2, ..., answer_10]
            }

        # ----------------------------
        # Save
        # ----------------------------
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(trunc_results, f, ensure_ascii=False, indent=2)

        print(f"[OK] Saved truncation results for lang={lang}: {out_path}")

    print("\nAll languages completed.")


if __name__ == "__main__":
    main()
