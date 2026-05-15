#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/Users/panli/GaTech Dropbox/Pan Li/multiagent_latent_space"
REPO="$WORKSPACE/LatentMAS"
VENV="$WORKSPACE/.venv-latentmas"

if ! command -v python3.10 >/dev/null 2>&1; then
  echo "python3.10 is not on PATH."
  echo "Install it first, for example:"
  echo "  brew install python@3.10"
  echo
  echo "Then rerun this script."
  exit 1
fi

python3.10 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$REPO/requirements.txt"

cat <<EOF
LatentMAS virtual environment is ready.

Activate it with:
  source "$VENV/bin/activate"

Verify with:
  python -m pip show torch transformers datasets accelerate
  cd "$REPO"
  python run.py --help
EOF
