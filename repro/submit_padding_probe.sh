#!/bin/bash
# FINDING #5: Qwen3's tokenizer_config.json sets no padding_side, so HF defaults to
# RIGHT padding, and nothing in models.py overrides it. Under right padding every
# sequence shorter than the batch max continues generating from a pad token rather
# than from its real last prompt token.
#
# It fails SILENTLY: generate_text_batch decodes with skip_special_tokens=True, so
# the pads never appear in the output text.
#
# Do NOT "fix" this by setting padding_side="left" alone. generate_text_batch slices
#   sequences[idx, prompt_lengths[idx]:]
# where prompt_lengths = attention_mask.sum(dim=1) -- correct ONLY under right
# padding. Under left padding the generation starts at input_ids.shape[1] for every
# row, so that slice would return prompt tail, not generation.
#
# This probe quantifies the artifact before committing GPU-hours: same 30 samples,
# same seed, bs=1 (no padding possible) vs bs=20 (heavy padding). A large gap means
# batched numbers are not trustworthy and every arm must run at bs=1.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3-4B}
TASK=${TASK:-gsm8k}
NSAMP=${NSAMP:-30}
LSTEPS=${LSTEPS:-10}

for BS in 1 20; do
  sbatch --partition=free-gpu --gres=gpu:A100:1 --time=3:00:00 \
    --export=ALL,MODEL=$MODEL,TASK=$TASK,METHOD=latent_mas,PROMPT=sequential,\
LSTEPS=$LSTEPS,NSAMP=$NSAMP,BS=$BS,MAXNEW=4096,THINK=1 \
    repro/run_latentmas.sbatch
done
squeue -u "$USER" -o "%.10i %.12P %.26j %.8T %.10M %R"
