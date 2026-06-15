#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline: optional Unsloth fine-tuning -> evaluation with the chosen backend.
# All knobs live in configs/pipeline.env (or any env file passed as $1). Values can
# also be overridden inline.
#
# Usage:
#   ./scripts/run_pipeline.sh [CONFIG_ENV]
# Examples:
#   ./scripts/run_pipeline.sh
#   BACKEND=vllm METHOD=baseline TASK=gsm8k ./scripts/run_pipeline.sh
#   RUN_TRAINING=1 ./scripts/run_pipeline.sh configs/pipeline.env

# Resolve repo root (this script lives in scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_ENV="${1:-configs/pipeline.env}"
if [[ ! -f "$CONFIG_ENV" ]]; then
  echo "Config file not found: $CONFIG_ENV" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_ENV"

echo "================ Pipeline configuration ================"
echo "  config_env : $CONFIG_ENV"
echo "  method     : $METHOD"
echo "  task       : $TASK${TASK:+ ($([[ "$TASK" == "mgsm" ]] && echo "lang=$MGSM_LANG" || echo "$SPLIT") )}"
echo "  model      : $MODEL_NAME"
echo "  backend    : $BACKEND"
echo "  train/infer: $RUN_TRAINING / $RUN_INFERENCE"
echo "======================================================="

# ---- Stage 1: optional fine-tuning ----
if [[ "$RUN_TRAINING" == "1" ]]; then
  echo "== [train] Unsloth fine-tuning ($UNSLOTH_CONFIG) =="
  python train.py --config "$UNSLOTH_CONFIG"
fi

# ---- Stage 2: optional inference ----
if [[ "$RUN_INFERENCE" == "1" ]]; then
  echo "== [infer] Evaluation =="

  CMD=(python run.py
    --method "$METHOD"
    --model_name "$MODEL_NAME"
    --task "$TASK"
    --prompt "$PROMPT"
    --split "$SPLIT"
    --device "$DEVICE"
    --max_samples "$MAX_SAMPLES"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --generate_bs "$GENERATE_BS"
    --seed "$SEED")

  if [[ "$TASK" == "mgsm" ]]; then
    CMD+=(--mgsm_lang "$MGSM_LANG")
  fi
  if [[ "$METHOD" == "latent_mas" ]]; then
    CMD+=(--latent_steps "$LATENT_STEPS")
  fi

  case "$BACKEND" in
    hf)
      ;;
    vllm)
      CMD+=(--use_vllm
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION")
      ;;
    llamacpp)
      if [[ -z "$LLAMACPP_MODEL_PATH" ]]; then
        echo "BACKEND=llamacpp requires LLAMACPP_MODEL_PATH (path to a .gguf file)." >&2
        exit 1
      fi
      CMD+=(--use_llamacpp
        --llamacpp_model_path "$LLAMACPP_MODEL_PATH"
        --llamacpp_n_ctx "$LLAMACPP_N_CTX"
        --llamacpp_n_gpu_layers "$LLAMACPP_N_GPU_LAYERS")
      ;;
    *)
      echo "Unknown BACKEND: $BACKEND (use hf | vllm | llamacpp)." >&2
      exit 1
      ;;
  esac

  mkdir -p "$RESULTS_DIR"
  RESULT_FILE="$RESULTS_DIR/${METHOD}_${TASK}_${BACKEND}_$(date +%Y%m%d_%H%M%S).log"
  echo "Running: ${CMD[*]}"
  "${CMD[@]}" | tee "$RESULT_FILE"
  echo "Saved output to $RESULT_FILE"
fi

echo "== Pipeline complete =="
