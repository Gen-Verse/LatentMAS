#!/bin/bash
# Build a clean env for LatentMAS. Run on the LOGIN node (needs network).
#
# ROOT CAUSE of the broken `latentmas` env (and of any python3.10 env on this
# account): ~/.local/lib/python3.10/site-packages is 931 MB of user-site packages,
# and site.ENABLE_USER_SITE is True, so that directory lands on sys.path of EVERY
# python3.10 env. Its nvidia-nccl-cu12 2.19.3 shadows whatever nccl the env's own
# torch shipped -> "undefined symbol: ncclCommResume". Rebuilding the env does not
# fix that; PYTHONNOUSERSITE does.
#
# It also makes a bare `pip install` DESTRUCTIVE: pip sees ~/.local packages as
# "existing installations" and uninstalls them out of your home directory.
# PYTHONNOUSERSITE=1 below hides ~/.local from both the interpreter and pip.
set -euo pipefail

export PYTHONNOUSERSITE=1
export PIP_USER=0

PUB=/pub/$USER
ENVDIR=$PUB/envs/latentmas
export HF_HOME=$PUB/hf

mkdir -p "$PUB/envs" "$HF_HOME"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -p "$ENVDIR" python=3.10
conda activate "$ENVDIR"

# CUDA 12.8 wheels: covers sm_80 (A30/A100) and sm_89 (L40S / RTX6000 Ada).
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
# transformers is PINNED: 5.x removed Cache.to_legacy_cache/from_legacy_cache
# (methods/latent_mas.py:68,73) and deprecated torch_dtype=. 4.57.1 matches the
# paper-era API surface.
pip install --no-cache-dir "transformers==4.57.1" accelerate datasets numpy tqdm matplotlib

python - <<'PY'
import site, sys, torch, transformers
leaked = [p for p in sys.path if ".local" in p]
assert not leaked, f"~/.local leaked into sys.path: {leaked}"
print("user-site visible:", site.ENABLE_USER_SITE, "| leaked paths:", leaked)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("nccl", torch.cuda.nccl.version() if hasattr(torch.cuda, "nccl") else "n/a")
print("transformers", transformers.__version__)
PY

echo
echo "Env ready at $ENVDIR"
echo "Activate with: conda activate $ENVDIR"
