#!/bin/bash
# Two minimal, reversible patches. Both are hard blockers for an HF-backend run.
# Revert with: git -C .. checkout methods/latent_mas.py run.py
set -euo pipefail
cd "$(dirname "$0")/.."

# (1) methods/latent_mas.py imports vllm unconditionally, so --method latent_mas
#     crashes on ImportError even though the paper says to use the HF backend for
#     official numbers, and requirements.txt does not list vllm.
if grep -q '^from vllm import SamplingParams$' methods/latent_mas.py; then
  python - <<'PY'
import re, pathlib
p = pathlib.Path("methods/latent_mas.py")
s = p.read_text()
s = s.replace(
    "from vllm import SamplingParams\n",
    "try:\n"
    "    from vllm import SamplingParams\n"
    "except ImportError:  # HF backend does not need vLLM\n"
    "    SamplingParams = None\n",
    1,
)
# self.sampling_params is only consumed by run_batch_vllm
s = s.replace(
    "        self.sampling_params = SamplingParams(\n",
    "        self.sampling_params = None if SamplingParams is None else SamplingParams(\n",
    1,
)
p.write_text(s)
print("patched methods/latent_mas.py")
PY
else
  echo "methods/latent_mas.py already patched"
fi

# (2) run.py --model_name choices list Qwen3-4B twice and omit Qwen3-8B, which the
#     README documents and the paper reports.
if grep -q '"Qwen/Qwen3-4B", "Qwen/Qwen3-4B", "Qwen/Qwen3-14B"' run.py; then
  sed -i 's|"Qwen/Qwen3-4B", "Qwen/Qwen3-4B", "Qwen/Qwen3-14B"|"Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"|' run.py
  echo "patched run.py model choices"
else
  echo "run.py already patched"
fi

git --no-pager diff --stat
