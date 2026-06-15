#!/usr/bin/env bash
set -euo pipefail

# === Config ===
SCRIPT="run.py"   # <- change if your file is named differently
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
# DATASET="juletxara/mgsm"
DATASET="shanchen/aime_2025_multilingual"

CACHE_DIR="./cache"
SEED=42
GPU_ID=1

# Languages = keys in instructions.json / hack_prefix.json
LANG_KEYS=(EN FR DE ZH JA RU ES SW BN TE TH)

for LANG in "${LANG_KEYS[@]}"; do
  echo "========================================"
  echo "Running language: ${LANG}"
  echo "========================================"

  NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=${GPU_ID} \
  python "${SCRIPT}" \
    --model_name "${MODEL}" \
    --dataset_name "${DATASET}" \
    --prompt_language "${LANG}" \
    --think_language "${LANG}" \
    --cache_dir "${CACHE_DIR}" \
    --seed ${SEED}
done
