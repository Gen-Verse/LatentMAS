import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from helper import normalize_lang_key

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="HF model name or local path."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="juletxara/mgsm",
        help="Full dataset name (used to derive dataset_base)."
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="EN",
        help="Comma-separated language keys, e.g. EN,FR,DE,..."
    )
    parser.add_argument(
        "--input_results_root",
        type=str,
        default="results_trunc",
        help="Root where truncated results are stored."
    )
    parser.add_argument(
        "--output_results_root",
        type=str,
        default="repr_hidden",
        help="Root where per-language hidden-state pickles will be stored."
    )
    parser.add_argument(
        "--input_file_suffix",
        type=str,
        default="trunc",
        help="Suffix in truncated JSON filenames: {lang_suffix}_{suffix}.json."
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional limit on number of examples per language."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model inference."
    )
    args = parser.parse_args()

    dataset_base = args.dataset_name.split("/")[-1]
    model_base = args.model_name.split("/")[-1]
    lang_list = [x.strip() for x in args.languages.split(",") if x.strip()]
    is_aime = "aime" in args.dataset_name.lower()

    print(f"Loading tokenizer & model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        cache_dir=args.cache_dir
    )
    model.eval()

    # Final norm for intermediate layers (as in your logit-lens code)
    final_norm = getattr(model.model, "norm", None)

    input_root = Path(args.input_results_root)
    output_root = Path(args.output_results_root)

    # Decide which percent checkpoints to use
    if is_aime:
        # AIME: 0%, 5%, ..., 100%
        step_percents = list(range(0, 101, 5))
    else:
        # MGSM/others: 0%, 10%, ..., 100%
        step_percents = list(range(0, 101, 10))

    # -------------------------------
    # LOOP OVER LANGUAGES
    # -------------------------------
    for lang in lang_list:
        lang_norm = normalize_lang_key(lang)
        lang_suffix = lang_norm.lower()
        
        
        # -------------------------------
        # CHECK IF OUTPUT ALREADY EXISTS
        # -------------------------------
        out_dir = output_root / dataset_base / model_base
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{lang_suffix}_hidden_states_repr.pkl"

        if out_path.exists():
            print(f"[SKIP] Hidden states already computed for {lang_norm}: {out_path}")
            continue
        # -------------------------------

        in_path = (
            input_root
            / dataset_base
            / model_base
            / f"{lang_suffix}_{args.input_file_suffix}.json"
        )
        if not in_path.exists():
            print(f"[WARN] No truncated results for {lang_norm}: {in_path}")
            continue

        print(f"\n=== Language {lang_norm} ===")
        full_results = load_json(in_path)
        if args.max_examples is not None:
            full_results = full_results[: args.max_examples]

        examples_data: List[Dict[str, Any]] = []

        for ex_idx, entry in enumerate(full_results):
            idx_val = int(entry.get("idx", ex_idx))

            hack_obj = entry.get("hack", {})
            forward_obj = hack_obj.get("forward", {})
            if not forward_obj:
                # no hack/forward data
                raise ValueError("No forward field")

            steps_list = []

            for p in step_percents:
                key = f"{p}%"
                if key not in forward_obj:
                    # this percentage may be missing for this example
                    raise ValueError(f"No {key} percentage")
                    

                prompt = forward_obj[key].get("prompt", None)
                if not prompt:
                    raise ValueError(f"No prompt")

                enc = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False
                )
                input_ids = enc["input_ids"].to(args.device)
                attention_mask = enc["attention_mask"].to(args.device)
                seq_len = int(attention_mask.sum(1)[0].item())
                last_token_index = seq_len - 1

                with torch.no_grad():
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )

                hidden_states = outputs.hidden_states  # tuple: (num_layers, 1, seq_len, d_model)
                n_hs = len(hidden_states)

                per_layer_vecs = []
                for layer_idx in range(n_hs):
                    h = hidden_states[layer_idx][0, last_token_index, :]  # (d_model,)

                    # Apply final_norm to intermediate layers
                    if final_norm is not None and layer_idx != n_hs - 1:
                        h = final_norm(h)

                    h_np = h.detach().to(torch.float16).cpu().numpy()
                    per_layer_vecs.append(h_np)

                hidden_arr = np.stack(per_layer_vecs, axis=0)  # (num_layers, hidden_dim)
                print(np.shape(hidden_arr))

                steps_list.append(
                    {
                        "percent": int(p),
                        "step_ratio": float(p) / 100.0,
                        "hidden": hidden_arr,
                    }
                )

            if steps_list:
                examples_data.append(
                    {
                        "idx": idx_val,
                        "steps": steps_list,
                    }
                )

        print(f"[OK] Collected hidden states for {lang_norm}: {len(examples_data)} examples")

        # Save ONE pickle per language
        out_dir = output_root / dataset_base / model_base
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{lang_suffix}_hidden_states_repr.pkl"

        data_to_save = {
            "meta": {
                "dataset": dataset_base,
                "model": model_base,
                "lang": lang_suffix,
                "is_aime": is_aime,
                "step_percents": step_percents,
            },
            "examples": examples_data,
        }

        with out_path.open("wb") as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[OK] Saved hidden-state representations for {lang_norm} → {out_path}")

    print("\nAll languages completed.")


if __name__ == "__main__":
    main()
