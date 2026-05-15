# LatentMAS Reproduction Log

Workspace created: 2026-05-15
Workspace root: `/Users/panli/GaTech Dropbox/Pan Li/multiagent_latent_space`
Upstream clone: `LatentMAS/`
Upstream commit at clone time: `bf8174b`

## Clean Local Setup

This workspace is configured for local Hugging Face CPU smoke tests with:

- `Qwen/Qwen3-0.6B` for fast laptop checks.
- `Qwen/Qwen3-4B` for closer but much slower checks.
- `Qwen/Qwen3-14B` left in the CLI allowlist for future GPU reproduction.

Qwen3.5 is intentionally not part of the local setup. `Qwen/Qwen3.5-0.8B` requires Transformers 5.x / `qwen3_5` support, but this Intel x86_64 Mac can only install prebuilt `torch` up to `2.2.2`, while Transformers 5.x expects newer PyTorch. It also uses a different image-text-to-text model family than the current LatentMAS `AutoModelForCausalLM` path.

## Environment

Local Miniforge:

```bash
/Users/panli/.local/miniforge3-latentmas
```

Python:

```bash
/Users/panli/.local/miniforge3-latentmas/envs/latentmas/bin/python
```

Hugging Face cache is outside Dropbox:

```bash
HF_HOME=/Users/panli/.cache/huggingface
```

Known package state:

```text
torch==2.2.2
transformers==4.57.1
datasets==4.8.5
accelerate==1.13.0
matplotlib==3.10.9
numpy==2.2.6
```

Note: PyTorch emits a NumPy 2.x ABI warning in this environment. NumPy was intentionally left unchanged.

## Source Patches Kept

- `run.py`: model allowlist includes `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-4B`, and `Qwen/Qwen3-14B`.
- `methods/latent_mas.py`: vLLM import and `SamplingParams` are optional, so local Hugging Face runs do not require vLLM.

## Smoke Results

### Qwen3-0.6B Baseline GSM8K

Script:

```bash
/Users/panli/GaTech\ Dropbox/Pan\ Li/multiagent_latent_space/scripts/smoke_qwen3_0p6b_gsm8k.sh
```

Result: completed in 33.2518 seconds on CPU. The single answer was incorrect because the fast script used `--max_new_tokens 128`, which truncated reasoning.

Final JSON:

```json
{"method": "baseline", "model": "Qwen/Qwen3-0.6B", "split": "test", "seed": 42, "max_samples": 1, "accuracy": 0.0, "correct": 0, "total_time_sec": 33.2518, "time_per_sample_sec": 33.2518}
```

### Qwen3-4B Baseline GSM8K

Script:

```bash
/Users/panli/GaTech\ Dropbox/Pan\ Li/multiagent_latent_space/scripts/smoke_gsm8k.sh
```

Result: completed in 620.7929 seconds on CPU and answered correctly.

Final JSON:

```json
{"method": "baseline", "model": "Qwen/Qwen3-4B", "split": "test", "seed": 42, "max_samples": 1, "accuracy": 1.0, "correct": 1, "total_time_sec": 620.7929, "time_per_sample_sec": 620.7929}
```

## Pending Tests

- `scripts/smoke_textmas_gsm8k.sh`: Qwen3-4B TextMAS, expected to be slow on CPU.
- `scripts/smoke_latentmas_gsm8k.sh`: Qwen3-4B LatentMAS, expected to be slow on CPU.
- Optional: create 0.6B TextMAS/LatentMAS scripts if fast multi-agent pipeline tests are needed.
