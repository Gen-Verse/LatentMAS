#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/bench_suite/hom_afrimgsm_first10_cvae_qwen3_4b.yaml}"
STAGES="${STAGES:-A,B,C,D,E,F,G}"
COMM_MODES="${COMM_MODES:-single_agent_baseline,latent_based_mas_ours}"
BACKEND="${BACKEND:-hf}"
OUTPUT_DIR="${OUTPUT_DIR:-results/bench_suite/hom_afrimgsm_first10_cvae_qwen3_4b_math_queries}"
RESUME="${RESUME:-0}"
RESUME_ARGS=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" ]]; then
  RESUME_ARGS=(--resume)
fi

echo "============ latent_coordination AfriMGSM first10 CVAE Qwen3-4B ============"
echo "  config    : ${CONFIG}"
echo "  stages    : ${STAGES}"
echo "  comm modes: ${COMM_MODES}"
echo "  backend   : ${BACKEND}"
echo "  out dir   : ${OUTPUT_DIR}"
echo "  resume    : ${RESUME}"
echo "============================================================================"

python scripts/run_coordination_pipeline_mgsm_queries.py \
  --config "${CONFIG}" \
  --stages "${STAGES}" \
  --comm-modes "${COMM_MODES}" \
  --backend "${BACKEND}" \
  --output-dir "${OUTPUT_DIR}" \
  "${RESUME_ARGS[@]}"
