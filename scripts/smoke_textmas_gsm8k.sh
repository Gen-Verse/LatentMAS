#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -f "$WORKSPACE/run.py" ]]; then
  REPO="$WORKSPACE"
else
  REPO="$WORKSPACE/LatentMAS"
fi
LOG_DIR="$WORKSPACE/runs/logs"
OUT_DIR="$WORKSPACE/runs/outputs"
STAMP="$(date +%Y%m%d_%H%M%S)"
PYTHON="${PYTHON:-python}"

mkdir -p "$LOG_DIR" "$OUT_DIR"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

cd "$REPO"
"$PYTHON" run.py \
  --method text_mas \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --prompt sequential \
  --max_samples 1 \
  --generate_bs 1 \
  --max_new_tokens 512 \
  --device cpu \
  2>&1 | tee "$LOG_DIR/textmas_gsm8k_$STAMP.log"
