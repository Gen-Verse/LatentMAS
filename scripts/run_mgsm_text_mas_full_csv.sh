#!/usr/bin/env bash
set -euo pipefail

# Run TextMAS sequential analysis on the full MGSM test set for all languages.
# This writes TextMAS response/agent CSVs, hidden-state trace pickles, language
# cosine matrices, and shared latent-path LRS correlation CSVs.
#
# Usage:
#   bash scripts/run_mgsm_text_mas_full_csv.sh
#
# Optional overrides:
#   MODEL_NAME=Qwen/Qwen3-4B DEVICE=auto RUN_NAME=mgsm_all_sequential_text_mas_csv bash scripts/run_mgsm_text_mas_full_csv.sh

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
LANGUAGES="${LANGUAGES:-bn,de,en,es,fr,ja,ru,sw,te,th,zh}"
PROMPT="${PROMPT:-sequential}"
MAX_EXAMPLES="${MAX_EXAMPLES:--1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_TRACE_STEPS="${MAX_TRACE_STEPS:-12}"
DEVICE="${DEVICE:-auto}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TEXT_MAS_CONTEXT_LENGTH="${TEXT_MAS_CONTEXT_LENGTH:--1}"
SEED="${SEED:-42}"
EMERGENCE_RANK_THRESHOLD="${EMERGENCE_RANK_THRESHOLD:-1000}"
EMERGENCE_LAYER_STRATEGY="${EMERGENCE_LAYER_STRATEGY:-final_layer}"
SHARED_LRS_THRESHOLDS="${SHARED_LRS_THRESHOLDS:-1,5,10,25,50,100,250,500,1000,2500,5000,10000}"
SHARED_LRS_LAYER_STRATEGY="${SHARED_LRS_LAYER_STRATEGY:-final_layer}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
RUN_NAME="${RUN_NAME:-mgsm_all_${PROMPT}_text_mas_csv}"
OUT_DIR="${OUT_DIR:-src/multilingual-latent-reasoning/results_text_mas_agents}"

echo "================ Full MGSM TextMAS CSV run ================"
echo "  model             : ${MODEL_NAME}"
echo "  languages         : ${LANGUAGES}"
echo "  prompt            : ${PROMPT}"
echo "  device            : ${DEVICE}"
echo "  max examples      : ${MAX_EXAMPLES}"
echo "  max tokens        : ${MAX_NEW_TOKENS}"
echo "  max trace steps   : ${MAX_TRACE_STEPS}"
echo "  run_name          : ${RUN_NAME}"
echo "  out_dir           : ${OUT_DIR}"
echo "  checkpoint        : every ${CHECKPOINT_EVERY} example(s)"
echo "  shared thresholds : ${SHARED_LRS_THRESHOLDS}"
echo "==========================================================="

python src/multilingual-latent-reasoning/run_text_mas_mgsm_batch_analysis.py \
  --model_name "${MODEL_NAME}" \
  --languages "${LANGUAGES}" \
  --prompt "${PROMPT}" \
  --max_examples "${MAX_EXAMPLES}" \
  --device "${DEVICE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_trace_steps "${MAX_TRACE_STEPS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --text_mas_context_length "${TEXT_MAS_CONTEXT_LENGTH}" \
  --seed "${SEED}" \
  --emergence_rank_threshold "${EMERGENCE_RANK_THRESHOLD}" \
  --emergence_layer_strategy "${EMERGENCE_LAYER_STRATEGY}" \
  --shared_lrs_thresholds "${SHARED_LRS_THRESHOLDS}" \
  --shared_lrs_layer_strategy "${SHARED_LRS_LAYER_STRATEGY}" \
  --checkpoint_every "${CHECKPOINT_EVERY}" \
  --out_dir "${OUT_DIR}" \
  --run_name "${RUN_NAME}"
