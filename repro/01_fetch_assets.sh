#!/bin/bash
# Pre-download models + datasets on the LOGIN node so GPU jobs never wait on network.
set -euo pipefail
export PYTHONNOUSERSITE=1
export HF_HOME=/pub/$USER/hf
mkdir -p "$HF_HOME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /pub/$USER/envs/latentmas

for M in Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B; do
  echo "=== $M ==="
  hf download "$M" || huggingface-cli download "$M"
done

python - <<'PY'
from datasets import load_dataset
for spec in [("openai/gsm8k","main","test"), ("evalplus/mbppplus",None,"test"),
             ("evalplus/humanevalplus",None,"test"), ("allenai/ai2_arc","ARC-Easy","test"),
             ("allenai/ai2_arc","ARC-Challenge","test"), ("fingertap/GPQA-Diamond",None,"test"),
             ("HuggingFaceH4/aime_2024",None,"train"), ("yentinglin/aime_2025",None,"train")]:
    name, cfg, split = spec
    try:
        d = load_dataset(name, cfg, split=split); print(f"OK  {name} {cfg} {split}: {len(d)}")
    except Exception as e:
        print(f"ERR {name} {cfg}: {type(e).__name__}: {e}")
PY
echo "Disk used:"; du -sh "$HF_HOME"
