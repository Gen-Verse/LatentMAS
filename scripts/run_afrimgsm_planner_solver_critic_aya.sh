#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-CohereLabs/aya-expanse-8b}"
LANGUAGES="${LANGUAGES:-am,ee,ha,ig,rw,ln,lg,om,sn,st,sw,tw,wo,xh,yo,zu}"
MAX_EXAMPLES="${MAX_EXAMPLES:-10}"
START_IDX="${START_IDX:-0}"
MODES="${MODES:-all}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
PLAN_TOKENS="${PLAN_TOKENS:-192}"
CRITIC_TOKENS="${CRITIC_TOKENS:-256}"
RUN_NAME="${RUN_NAME:-aya_first10_planner_solver_critic}"
OUT_DIR="${OUT_DIR:-results/afrimgsm_planner_solver_critic}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"

echo "================ AfriMGSM planner/solver/critic Aya ================"
echo "  model      : ${MODEL_NAME}"
echo "  languages  : ${LANGUAGES}"
echo "  examples   : ${MAX_EXAMPLES}"
echo "  start idx  : ${START_IDX}"
echo "  modes      : ${MODES}"
echo "  device     : ${DEVICE}"
echo "  run name   : ${RUN_NAME}"
echo "  out dir    : ${OUT_DIR}"
echo "====================================================================="

python scripts/run_afrimgsm_planner_solver_critic.py \
  --model_name "${MODEL_NAME}" \
  --languages "${LANGUAGES}" \
  --max_examples "${MAX_EXAMPLES}" \
  --start_idx "${START_IDX}" \
  --modes "${MODES}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --plan_tokens "${PLAN_TOKENS}" \
  --critic_tokens "${CRITIC_TOKENS}" \
  --run_name "${RUN_NAME}" \
  --out_dir "${OUT_DIR}" \
  --checkpoint_every "${CHECKPOINT_EVERY}" \
  --load_in_8bit
