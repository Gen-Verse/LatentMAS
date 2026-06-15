#!/usr/bin/env bash
set -euo pipefail

########################################
# Config
########################################

SCRIPT="run_logitlens_dynamics.py"

# All models you want to analyze
MODELS=(
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
  # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

# All datasets you want to analyze
DATASETS=(
  # "juletxara/mgsm"
  "shanchen/aime_2024_multilingual"
  "shanchen/aime_2025_multilingual"
)

# Languages = keys used in your JSON + helper.py
LANG_KEYS=(EN FR DE ZH JA RU ES SW BN TE TH)

# Turn LANG_KEYS into comma-separated string for --languages
LANG_CSV=$(IFS=,; echo "${LANG_KEYS[*]}")

CACHE_DIR="./cache"
SEED=42
GPU_ID=0,1

# Where original full-CoT results are stored
INPUT_RESULTS_ROOT="results"

# Where to store logit-lens outputs
OUTPUT_RESULTS_ROOT="logitlens"


########################################
# Main loop
########################################

echo "Running logit-lens dynamics for:"
echo "  MODELS:   ${MODELS[*]}"
echo "  DATASETS: ${DATASETS[*]}"
echo "  LANGS:    ${LANG_CSV}"
echo

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    echo "========================================"
    echo " MODEL:   ${MODEL}"
    echo " DATASET: ${DATASET}"
    echo " LANGS:   ${LANG_CSV}"
    echo "========================================"

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=${GPU_ID} \
    python "${SCRIPT}" \
      --model_name "${MODEL}" \
      --dataset_name "${DATASET}" \
      --languages "${LANG_CSV}" \
      --cache_dir "${CACHE_DIR}" \
      --input_results_root "${INPUT_RESULTS_ROOT}" \
      --output_results_root "${OUTPUT_RESULTS_ROOT}"

    echo "Finished logit-lens: model=${MODEL}, dataset=${DATASET}"
    echo
  done
done

echo "========================================"
echo " All logit-lens dynamics runs completed!"
echo "========================================"
