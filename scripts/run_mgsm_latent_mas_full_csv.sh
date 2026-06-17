#!/usr/bin/env bash
set -euo pipefail

# Run LatentMAS sequential analysis on the full MGSM test set for all languages.
# This uses the src/multilingual-latent-reasoning batch analysis script and writes
# CSV/JSON/pickle outputs under src/multilingual-latent-reasoning/results_latent_mas_agents/.
#
# Usage:
#   bash scripts/run_mgsm_latent_mas_full_csv.sh
#
# Optional overrides:
#   MODEL_NAME=Qwen/Qwen3-4B DEVICE=auto RUN_NAME=mgsm_all_sequential_csv bash scripts/run_mgsm_latent_mas_full_csv.sh

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
LANGUAGES="${LANGUAGES:-bn,de,en,es,fr,ja,ru,sw,te,th,zh}"
PROMPT="${PROMPT:-sequential}"
LATENT_STEPS="${LATENT_STEPS:-3}"
MAX_EXAMPLES="${MAX_EXAMPLES:--1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
DEVICE="${DEVICE:-auto}"
DEVICE2="${DEVICE2:-cuda:1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
SEED="${SEED:-42}"
EMERGENCE_RANK_THRESHOLD="${EMERGENCE_RANK_THRESHOLD:-1000}"
EMERGENCE_LAYER_STRATEGY="${EMERGENCE_LAYER_STRATEGY:-final_layer}"
LANGUAGE_REASONING_DISENTANGLE="${LANGUAGE_REASONING_DISENTANGLE:-0}"
LR_VECTOR_PATH="${LR_VECTOR_PATH:-}"
LR_DISENTANGLE_STRENGTH="${LR_DISENTANGLE_STRENGTH:-0.2}"
LR_DISENTANGLE_VECTOR_LAYER="${LR_DISENTANGLE_VECTOR_LAYER:--1}"
LR_DISENTANGLE_ROLES="${LR_DISENTANGLE_ROLES:-planner,critic,refiner}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
RUN_NAME="${RUN_NAME:-mgsm_all_${PROMPT}_csv}"
OUT_DIR="${OUT_DIR:-src/multilingual-latent-reasoning/results_latent_mas_agents}"

EXTRA_ARGS=()
if [[ "${LANGUAGE_REASONING_DISENTANGLE}" == "1" || "${LANGUAGE_REASONING_DISENTANGLE}" == "true" ]]; then
  EXTRA_ARGS+=(--language_reasoning_disentangle)
  EXTRA_ARGS+=(--lr_vector_path "${LR_VECTOR_PATH}")
  EXTRA_ARGS+=(--lr_disentangle_strength "${LR_DISENTANGLE_STRENGTH}")
  EXTRA_ARGS+=(--lr_disentangle_vector_layer "${LR_DISENTANGLE_VECTOR_LAYER}")
  EXTRA_ARGS+=(--lr_disentangle_roles "${LR_DISENTANGLE_ROLES}")
fi

echo "================ Full MGSM LatentMAS CSV run ================"
echo "  model      : ${MODEL_NAME}"
echo "  languages  : ${LANGUAGES}"
echo "  prompt     : ${PROMPT}"
echo "  device     : ${DEVICE}"
echo "  latent     : ${LATENT_STEPS}"
echo "  max examples: ${MAX_EXAMPLES}"
echo "  max tokens : ${MAX_NEW_TOKENS}"
echo "  run_name   : ${RUN_NAME}"
echo "  out_dir    : ${OUT_DIR}"
echo "  checkpoint : every ${CHECKPOINT_EVERY} example(s)"
echo "  LR disent. : ${LANGUAGE_REASONING_DISENTANGLE}"
if [[ "${LANGUAGE_REASONING_DISENTANGLE}" == "1" || "${LANGUAGE_REASONING_DISENTANGLE}" == "true" ]]; then
  echo "  LR vector  : ${LR_VECTOR_PATH}"
  echo "  LR strength: ${LR_DISENTANGLE_STRENGTH}"
  echo "  LR roles   : ${LR_DISENTANGLE_ROLES}"
fi
echo "============================================================="

python src/multilingual-latent-reasoning/run_latent_mas_mgsm_batch_analysis.py \
  --model_name "${MODEL_NAME}" \
  --languages "${LANGUAGES}" \
  --prompt "${PROMPT}" \
  --latent_steps "${LATENT_STEPS}" \
  --max_examples "${MAX_EXAMPLES}" \
  --device "${DEVICE}" \
  --device2 "${DEVICE2}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --seed "${SEED}" \
  --emergence_rank_threshold "${EMERGENCE_RANK_THRESHOLD}" \
  --emergence_layer_strategy "${EMERGENCE_LAYER_STRATEGY}" \
  --checkpoint_every "${CHECKPOINT_EVERY}" \
  --out_dir "${OUT_DIR}" \
  --run_name "${RUN_NAME}" \
  "${EXTRA_ARGS[@]}"
