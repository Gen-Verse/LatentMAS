#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/Users/panli/GaTech Dropbox/Pan Li/multiagent_latent_space"
REPO="$WORKSPACE/LatentMAS"
LOG_DIR="$WORKSPACE/runs/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
PYTHON="/Users/panli/.local/miniforge3-latentmas/envs/latentmas/bin/python"

mkdir -p "$LOG_DIR"
export HF_HOME="/Users/panli/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

cd "$REPO"

for METHOD in baseline text_mas latent_mas; do
  EXTRA_ARGS=()
  if [[ "$METHOD" == "latent_mas" ]]; then
    EXTRA_ARGS=(--latent_steps 1)
  fi

  "$PYTHON" run.py \
    --method "$METHOD" \
    --model_name Qwen/Qwen3.5-0.8B \
    --model_backend auto \
    --task gsm8k \
    --max_samples 1 \
    --generate_bs 1 \
    --max_new_tokens 128 \
    --device cpu \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/${METHOD}_qwen35_0p8b_text_vlm_gsm8k_$STAMP.log"
done
