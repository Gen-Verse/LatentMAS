#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
LANGUAGES="${LANGUAGES:-bn,de,en,es,fr,ja,ru,sw,te,th,zh}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
CALIBRATION_EXAMPLES="${CALIBRATION_EXAMPLES:-10}"
CALIBRATION_START_IDX="${CALIBRATION_START_IDX:-0}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-10}"
EVAL_START_IDX="${EVAL_START_IDX:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
LAYERS="${LAYERS:-12-26}"
N_COMPONENTS="${N_COMPONENTS:-8}"
ALPHAS="${ALPHAS:-0,0.025,0.05,0.1,0.2}"
OUT_DIR="${OUT_DIR:-results/mgsm_reasoning_lr_disentangle}"
RUN_NAME="${RUN_NAME:-mgsm_calib${CALIBRATION_EXAMPLES}_eval${EVAL_EXAMPLES}_lr_disentangle}"

echo "================ MGSM Reasoning L-R Disentangle ================"
echo "  model       : ${MODEL_NAME}"
echo "  languages   : ${LANGUAGES}"
echo "  device      : ${DEVICE}"
echo "  dtype       : ${DTYPE}"
echo "  calibration : ${CALIBRATION_EXAMPLES} from idx ${CALIBRATION_START_IDX}"
echo "  eval        : ${EVAL_EXAMPLES} from idx ${EVAL_START_IDX}"
echo "  max tokens  : ${MAX_NEW_TOKENS}"
echo "  layers      : ${LAYERS}"
echo "  components  : ${N_COMPONENTS}"
echo "  alphas      : ${ALPHAS}"
echo "  run_name    : ${RUN_NAME}"
echo "  out_dir     : ${OUT_DIR}"
echo "================================================================"

python scripts/run_mgsm_reasoning_lr_disentangle.py \
  --model_name "${MODEL_NAME}" \
  --languages "${LANGUAGES}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --calibration_examples "${CALIBRATION_EXAMPLES}" \
  --calibration_start_idx "${CALIBRATION_START_IDX}" \
  --eval_examples "${EVAL_EXAMPLES}" \
  --eval_start_idx "${EVAL_START_IDX}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --layers "${LAYERS}" \
  --n_components "${N_COMPONENTS}" \
  --alphas "${ALPHAS}" \
  --out_dir "${OUT_DIR}" \
  --run_name "${RUN_NAME}"
