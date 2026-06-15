#!/usr/bin/env bash
set -euo pipefail

###############################
# Config
###############################

SCRIPT="run_truncation.py"      # truncation generation script
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
CACHE_DIR="./cache"
SEED=42
GPU_ID=2,3
MAX_TOKENS=10

# All datasets to run
DATASETS=(
  "juletxara/mgsm"
  "shanchen/aime_2024_multilingual"
  "shanchen/aime_2025_multilingual"
)

# Languages (comma-separated list for Python)
LANG_KEYS="EN,FR,DE,ZH,JA,RU,ES,SW,BN,TE,TH"


###############################
# Main Loop
###############################

for DATASET in "${DATASETS[@]}"; do
  echo "========================================"
  echo " DATASET: ${DATASET}"
  echo " MODEL:   ${MODEL}"
  echo "========================================"

  echo " Running truncation for ALL LANGUAGES: ${LANG_KEYS}"
  echo "----------------------------------------"

  NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=${GPU_ID} \
  python "${SCRIPT}" \
      --model_name "${MODEL}" \
      --dataset_name "${DATASET}" \
      --languages "${LANG_KEYS}" \
      --cache_dir "${CACHE_DIR}" \
      --max_tokens ${MAX_TOKENS} \
      --seed ${SEED}

  echo "========================================"
  echo " Finished DATASET: ${DATASET}"
  echo "========================================"
  echo ""
done

echo "========================================"
echo " All truncation experiments finished!"
echo "========================================"
