# LatentMAS Setup Notes

Practical setup tips collected from running LatentMAS locally. For paper and upstream code see https://arxiv.org/abs/2511.20639 and https://github.com/Gen-Verse/LatentMAS.

## Python and package versions

LatentMAS is developed against **Python 3.10**. The following pinned versions are known to work end-to-end:

```text
torch==2.2.2
transformers==4.57.1
datasets==4.8.5
accelerate==1.13.0
matplotlib==3.10.9
numpy==2.2.6
```

PyTorch 2.2.2 emits a NumPy 2.x ABI warning against `numpy==2.2.6`. The warning is benign in this combination; do not downgrade NumPy unless you hit a real failure.

## Hugging Face cache location

Model weights and datasets are large (multi-GB). If your repo lives inside a cloud-synced folder (Dropbox, OneDrive, iCloud, etc.), redirect the HF cache **outside** that folder before any download:

```bash
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
```

The smoke-test scripts under `scripts/` set these automatically.

## Choosing a model for local runs

| Model | Use case |
|---|---|
| `Qwen/Qwen3-0.6B` | Fast CPU pipeline checks. |
| `Qwen/Qwen3-4B`   | Slower but closer to the original smoke plan. |
| `Qwen/Qwen3-14B`  | Kept in the CLI allowlist for GPU reproduction; not practical on a CPU-only laptop. |
| `Qwen/Qwen3.5-0.8B` | Requires the VLM (image-text-to-text) backend — see caveat below. |

## Qwen3.5-0.8B caveat on Intel Mac

Qwen3.5 needs Transformers 5.x (`qwen3_5` support) and the `AutoModelForImageTextToText` loader path, not `AutoModelForCausalLM`. Transformers 5.x in turn requires a newer PyTorch than the x86_64 macOS wheels currently provide (capped at `torch==2.2.2`). On Intel macOS this combination is not installable; use a Linux/CUDA box or Apple Silicon for Qwen3.5 runs.
