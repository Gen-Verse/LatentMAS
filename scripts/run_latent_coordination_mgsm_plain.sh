#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
LANGUAGES="${LANGUAGES:-bn,de,en,es,fr,ja,ru,sw,te,th,zh}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
HIDDEN_DIM="${HIDDEN_DIM:-2560}"
UNIVERSAL_DIM="${UNIVERSAL_DIM:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TRANSLATION_MAX_NEW_TOKENS="${TRANSLATION_MAX_NEW_TOKENS:-512}"
MODE="${MODE:-reasoning_only}"
TRANSLATION_TARGET_LANGUAGE="${TRANSLATION_TARGET_LANGUAGE:-same}"
ANCHOR_LANG="${ANCHOR_LANG:-en}"
SFR_THRESHOLD="${SFR_THRESHOLD:-0.3}"
MAX_EXTRA_TRANSLATION_NUMBERS="${MAX_EXTRA_TRANSLATION_NUMBERS:-0}"
MAX_TRANSLATION_LENGTH_RATIO="${MAX_TRANSLATION_LENGTH_RATIO:-2.5}"
INCLUDE_ORIGINAL_QUESTION="${INCLUDE_ORIGINAL_QUESTION:-0}"
PASS_TEXT_CONTEXT="${PASS_TEXT_CONTEXT:-0}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-0}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
OUT_DIR="${OUT_DIR:-results/latent_coordination_mgsm_plain}"
RUN_NAME="${RUN_NAME:-}"

export PYTHONPATH="src:.:${PYTHONPATH:-}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "================ latent_coordination MGSM plain ================"
echo "  model       : ${MODEL_NAME}"
echo "  languages   : ${LANGUAGES}"
echo "  max examples: ${MAX_EXAMPLES}"
echo "  device      : ${DEVICE}"
echo "  dtype       : ${DTYPE}"
echo "  hidden dim  : ${HIDDEN_DIM}"
echo "  universal   : ${UNIVERSAL_DIM}"
echo "  max tokens  : ${MAX_NEW_TOKENS}"
echo "  trans tokens: ${TRANSLATION_MAX_NEW_TOKENS}"
echo "  mode        : ${MODE}"
echo "  trans target: ${TRANSLATION_TARGET_LANGUAGE}"
echo "  anchor lang : ${ANCHOR_LANG}"
echo "  sfr thresh  : ${SFR_THRESHOLD}"
echo "  extra nums  : ${MAX_EXTRA_TRANSLATION_NUMBERS}"
echo "  len ratio   : ${MAX_TRANSLATION_LENGTH_RATIO}"
echo "  orig context: ${INCLUDE_ORIGINAL_QUESTION}"
echo "  pass text   : ${PASS_TEXT_CONTEXT}"
echo "  load 8bit   : ${LOAD_IN_8BIT}"
echo "  load 4bit   : ${LOAD_IN_4BIT}"
echo "  out_dir     : ${OUT_DIR}"
echo "==============================================================="

ARGS=()
if [[ -n "${RUN_NAME}" ]]; then
  ARGS+=(--run_name "${RUN_NAME}")
fi
if [[ "${INCLUDE_ORIGINAL_QUESTION}" == "1" || "${INCLUDE_ORIGINAL_QUESTION}" == "true" ]]; then
  ARGS+=(--include_original_question)
fi
if [[ "${PASS_TEXT_CONTEXT}" == "1" || "${PASS_TEXT_CONTEXT}" == "true" ]]; then
  ARGS+=(--pass_text_context)
fi
if [[ "${LOAD_IN_8BIT}" == "1" || "${LOAD_IN_8BIT}" == "true" ]]; then
  ARGS+=(--load_in_8bit)
fi
if [[ "${LOAD_IN_4BIT}" == "1" || "${LOAD_IN_4BIT}" == "true" ]]; then
  ARGS+=(--load_in_4bit)
fi

python scripts/run_latent_coordination_mgsm_plain.py \
  --model_name "${MODEL_NAME}" \
  --languages "${LANGUAGES}" \
  --max_examples "${MAX_EXAMPLES}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --universal_dim "${UNIVERSAL_DIM}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --translation_max_new_tokens "${TRANSLATION_MAX_NEW_TOKENS}" \
  --mode "${MODE}" \
  --translation_target_language "${TRANSLATION_TARGET_LANGUAGE}" \
  --anchor_lang "${ANCHOR_LANG}" \
  --sfr_threshold "${SFR_THRESHOLD}" \
  --max_extra_translation_numbers "${MAX_EXTRA_TRANSLATION_NUMBERS}" \
  --max_translation_length_ratio "${MAX_TRANSLATION_LENGTH_RATIO}" \
  --out_dir "${OUT_DIR}" \
  "${ARGS[@]}"
