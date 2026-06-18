#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT="${RESULTS_ROOT:-src/multilingual-latent-reasoning/results_latent_mas_agents/Qwen3-4B}"
PATTERN="${PATTERN:-mgsm_first50_latent_mas}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/lr_disentanglement_steering_figures}"

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"
mkdir -p "${MPLCONFIGDIR}" "${OUT_DIR}"

echo "================ L-R steering plot extraction ================"
echo "  results_root : ${RESULTS_ROOT}"
echo "  pattern      : ${PATTERN}"
echo "  out_dir      : ${OUT_DIR}"
echo "  mpl config   : ${MPLCONFIGDIR}"
echo "=============================================================="

python src/multilingual-latent-reasoning/plot_lr_disentanglement_steering_results.py \
  --results_root "${RESULTS_ROOT}" \
  --pattern "${PATTERN}" \
  --out_dir "${OUT_DIR}"
