#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/bench_suite/hom_mgsm_first10_cvae.yaml}"
STAGES="${STAGES:-A,B,C,D,E,F,G}"
COMM_MODES="${COMM_MODES:-single_agent_baseline,token_based_mas,latent_based_mas_ours}"
BACKEND="${BACKEND:-hf}"
OUTPUT_DIR="${OUTPUT_DIR:-results/bench_suite/hom_mgsm_first10_cvae}"
RESUME="${RESUME:-0}"
RESUME_ARGS=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" ]]; then
  RESUME_ARGS=(--resume)
fi

echo "================ latent_coordination MGSM first10 CVAE ================"
echo "  config    : ${CONFIG}"
echo "  stages    : ${STAGES}"
echo "  comm modes: ${COMM_MODES}"
echo "  backend   : ${BACKEND}"
echo "  out dir   : ${OUTPUT_DIR}"
echo "  resume    : ${RESUME}"
echo "======================================================================="

python -m scripts.run_coordination_pipeline \
  --config "${CONFIG}" \
  --stages "${STAGES}" \
  --comm-modes "${COMM_MODES}" \
  --backend "${BACKEND}" \
  --output-dir "${OUTPUT_DIR}" \
  "${RESUME_ARGS[@]}"
