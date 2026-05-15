# LatentMAS Local Setup Notes

Workspace root: `/Users/panli/GaTech Dropbox/Pan Li/multiagent_latent_space`
Upstream clone: `/Users/panli/GaTech Dropbox/Pan Li/multiagent_latent_space/LatentMAS`
Paper: https://arxiv.org/abs/2511.20639
Repo: https://github.com/Gen-Verse/LatentMAS

## Local Environment

A user-space Miniforge install is available at:

```bash
/Users/panli/.local/miniforge3-latentmas
```

Use the `latentmas` environment Python directly:

```bash
/Users/panli/.local/miniforge3-latentmas/envs/latentmas/bin/python
```

To activate manually:

```bash
source /Users/panli/.local/miniforge3-latentmas/bin/activate latentmas
```

## Package State

Current package versions:

```text
Python 3.10
torch==2.2.2
transformers==4.57.1
datasets==4.8.5
accelerate==1.13.0
matplotlib==3.10.9
numpy==2.2.6
```

Note: PyTorch emits a NumPy 2.x ABI warning in this environment. NumPy was intentionally left unchanged.

## Hugging Face Cache

Downloaded models and datasets are kept outside Dropbox:

```bash
export HF_HOME=/Users/panli/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
```

The smoke-test scripts set these variables automatically.

## Supported Local Models

Use these for local runs:

- `Qwen/Qwen3-0.6B`: fast CPU pipeline checks.
- `Qwen/Qwen3-4B`: slower but closer to the original smoke plan.
- `Qwen/Qwen3-14B`: kept in the CLI allowlist for future GPU reproduction, not practical on this CPU laptop.

Do not use `Qwen/Qwen3.5-0.8B` in this local Intel Mac setup. Qwen3.5 needs Transformers 5.x / `qwen3_5` support, but this x86_64 macOS environment can only install prebuilt `torch` up to `2.2.2`; Transformers 5.x expects newer PyTorch. Qwen3.5 also uses an image-text-to-text loader path rather than the current LatentMAS `AutoModelForCausalLM` path.

## Smoke Scripts

Fast local baseline:

```bash
"/Users/panli/GaTech Dropbox/Pan Li/multiagent_latent_space/scripts/smoke_qwen3_0p6b_gsm8k.sh"
```

Qwen3-4B baseline:

```bash
"/Users/panli/GaTech Dropbox/Pan Li/multiagent_latent_space/scripts/smoke_gsm8k.sh"
```

Qwen3-4B TextMAS:

```bash
"/Users/panli/GaTech Dropbox/Pan Li/multiagent_latent_space/scripts/smoke_textmas_gsm8k.sh"
```

Qwen3-4B LatentMAS:

```bash
"/Users/panli/GaTech Dropbox/Pan Li/multiagent_latent_space/scripts/smoke_latentmas_gsm8k.sh"
```

## Source Patches

Two intentional source patches are kept in the upstream clone:

- `run.py`: adds `Qwen/Qwen3-0.6B` to the model allowlist.
- `methods/latent_mas.py`: makes vLLM optional so local Hugging Face backend runs do not require vLLM.
