#!/bin/bash
# Minimal asset fetch for the Qwen3-4B / GSM8K reproduction slice.
# Full 3-model, 8-dataset prefetch lives in 01_fetch_assets.sh.
set -euo pipefail
export PYTHONNOUSERSITE=1
export HF_HOME=/pub/$USER/hf
mkdir -p "$HF_HOME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /pub/$USER/envs/latentmas

hf download Qwen/Qwen3-4B || huggingface-cli download Qwen/Qwen3-4B

python - <<'PY'
from datasets import load_dataset
d = load_dataset("openai/gsm8k", "main", split="test")
print("gsm8k test:", len(d))
PY
du -sh "$HF_HOME"
