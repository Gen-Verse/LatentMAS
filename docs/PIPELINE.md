# Developer Guide — Running the Pipeline & Components

This is the internal dev doc for the **MultilingualLatentMAS** fork. It explains how
to run the full pipeline and each component (inference, backends, fine-tuning,
analysis). The top-level `README.md` is the upstream cloned-repo README and is left
as-is — use *this* doc for day-to-day development.

- [1. Environment setup](#1-environment-setup)
- [2. Repository layout](#2-repository-layout)
- [3. One-shot pipeline](#3-one-shot-pipeline-recommended)
- [4. Inference (`run.py`)](#4-inference-runpy)
- [5. Inference backends](#5-inference-backends)
- [6. Fine-tuning with Unsloth (`train.py`)](#6-fine-tuning-with-unsloth-trainpy)
- [7. Latent-reasoning analysis (`src/`)](#7-latent-reasoning-analysis-src)
- [8. Helper scripts](#8-helper-scripts)
- [9. Troubleshooting](#9-troubleshooting)

---

## 1. Environment setup

```bash
# conda env (defined in environment.yml; built around torch + vLLM)
conda activate vllm_env          # or: conda env create -f environment.yml

# core install (editable) + whichever optional integrations you need
pip install -e .                 # core: transformers, torch, datasets, ...
pip install -e .[vllm]           # vLLM fast inference
pip install -e .[llamacpp]       # llama.cpp (GGUF) fast inference
pip install -e .[unsloth]        # Unsloth LoRA fine-tuning
pip install -e .[all]            # everything

# avoid anonymous HF rate limits / gated models
export HF_TOKEN="hf_..."
```

The optional extras are **modular and import-guarded** — the core inference path runs
without any of them installed. Extras are declared in `pyproject.toml` under
`[project.optional-dependencies]`.

If you cloned without submodules, also run:

```bash
git submodule update --init      # pulls multilingual-latent-reasoner/
```

---

## 2. Repository layout

| Path | Purpose |
|------|---------|
| `run.py` | Main evaluation entrypoint (baseline / text_mas / latent_mas) |
| `train.py` | Unsloth LoRA fine-tuning entrypoint (optional) |
| `models.py` | `ModelWrapper` — HF + vLLM + llama.cpp + latent realignment |
| `methods/` | `baseline.py`, `text_mas.py`, `latent_mas.py` |
| `training/` | Modular Unsloth trainer (`UnslothTrainer`) |
| `data.py` / `prompts.py` / `utils.py` | Loaders, prompt builders, helpers (incl. `load_yaml_config`) |
| `configs/` | `pipeline.env`, `unsloth_train.yaml`, `llamacpp.yaml`, `accelerate_config.yaml` |
| `scripts/` | `run_pipeline.sh` (end-to-end) + MGSM sweep helpers |
| `src/multilingual-latent-reasoning/` | Latent-reasoning analysis scripts (run from repo root) |
| `multilingual-latent-reasoner/` | Git submodule (external, `cisnlp/...`) |
| `results/`, `outputs/` | Run logs and training artifacts (git-ignored) |

---

## 3. One-shot pipeline (recommended)

Everything configurable lives in **`configs/pipeline.env`**; the runner
**`scripts/run_pipeline.sh`** sources it and orchestrates *optional fine-tuning →
evaluation* with the selected backend, teeing output to a timestamped log under
`results/`.

```bash
# 1) edit configs/pipeline.env, then:
./scripts/run_pipeline.sh

# 2) or override any variable inline (no file edits needed):
BACKEND=vllm METHOD=baseline TASK=gsm8k ./scripts/run_pipeline.sh
RUN_TRAINING=1 BACKEND=llamacpp LLAMACPP_MODEL_PATH=/path/model.gguf ./scripts/run_pipeline.sh

# 3) or point at a different env file:
./scripts/run_pipeline.sh path/to/my.env
```

### `configs/pipeline.env` variables

Every value uses `${VAR:-default}`, so inline overrides win over the file.

| Variable | Default | Notes |
|----------|---------|-------|
| `RUN_TRAINING` / `RUN_INFERENCE` | `0` / `1` | Stage toggles (1 = run) |
| `MODEL_NAME` | `Qwen/Qwen3-4B` | HF id or local path |
| `METHOD` | `latent_mas` | `baseline` \| `text_mas` \| `latent_mas` |
| `TASK` | `gsm8k` | any supported task (see §4) |
| `MGSM_LANG` | `en` | only used when `TASK=mgsm` |
| `PROMPT` | `sequential` | `sequential` \| `hierarchical` |
| `SPLIT` | `test` | dataset split |
| `DEVICE` | `cuda` | `cuda` \| `cuda:N` \| `cpu` \| `mps` \| `auto` |
| `MAX_SAMPLES` | `-1` | `-1` = all |
| `MAX_NEW_TOKENS` | `2048` | |
| `LATENT_STEPS` | `3` | latent_mas only |
| `TEMPERATURE` / `TOP_P` | `0.6` / `0.95` | sampling |
| `GENERATE_BS` | `20` | batch size |
| `SEED` | `42` | |
| `BACKEND` | `hf` | `hf` \| `vllm` \| `llamacpp` |
| `TENSOR_PARALLEL_SIZE` / `GPU_MEMORY_UTILIZATION` | `1` / `0.9` | vLLM |
| `LLAMACPP_MODEL_PATH` / `LLAMACPP_N_CTX` / `LLAMACPP_N_GPU_LAYERS` | `""` / `4096` / `-1` | llama.cpp |
| `UNSLOTH_CONFIG` | `configs/unsloth_train.yaml` | training config |
| `RESULTS_DIR` | `results` | log destination |

The runner validates choices (e.g. `llamacpp` requires `LLAMACPP_MODEL_PATH`; only one
backend at a time) and assembles the right `run.py` flags automatically.

---

## 4. Inference (`run.py`)

The pipeline ultimately calls `run.py`. To run it directly:

```bash
python run.py \
  --method latent_mas \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --prompt sequential \
  --latent_steps 3 \
  --max_samples -1 \
  --max_new_tokens 2048 \
  --generate_bs 20 \
  --device cuda
```

**Methods** (`--method`): `baseline` (single agent), `text_mas` (token-space
multi-agent), `latent_mas` (latent-space multi-agent — the core method).

**Tasks** (`--task`): `gsm8k`, `aime2024`, `aime2025`, `gpqa`, `arc_easy`,
`arc_challenge`, `mbppplus`, `humanevalplus`, `medqa`, `mgsm`. For `mgsm` add
`--mgsm_lang` (one of `bn de en es fr ja ru sw te th zh`).

**Key flags**: `--prompt {sequential,hierarchical}`, `--latent_steps N` (latent_mas),
`--temperature`, `--top_p`, `--max_new_tokens`, `--generate_bs`, `--seed`,
`--device`, `--split`, `--think`, `--latent_space_realign`.

`run.py` prints a JSON summary at the end (`accuracy`, `correct`, timing). The
multilingual MGSM sweeps from the README map onto the `mgsm` task — see §8 for the
ready-made sweep scripts.

---

## 5. Inference backends

Selected via flags on `run.py` (or `BACKEND=` in `pipeline.env`). They are mutually
exclusive.

### `hf` (default, transformers)
No extra flags. Works for **all** methods, including `latent_mas` (needs hidden
states). Use `--device auto` for HF `device_map="auto"` sharding.

### `vllm`
```bash
python run.py --method baseline --model_name Qwen/Qwen3-14B --task gsm8k \
  --use_vllm --tensor_parallel_size 1 --gpu_memory_utilization 0.9 --max_new_tokens 2048
```
For `latent_mas`, `run.py` auto-enables a second HF model + prefix caching (it needs
hidden states that vLLM alone doesn't expose). Requires `pip install -e .[vllm]`.

### `llamacpp` (GGUF, fast — **text generation only**)
```bash
python run.py --method baseline --model_name Qwen/Qwen3-4B --task gsm8k \
  --use_llamacpp \
  --llamacpp_model_path /path/to/model-Q4_K_M.gguf \
  --llamacpp_n_ctx 4096 --llamacpp_n_gpu_layers -1
```
- Supports `baseline` and `text_mas` only — `latent_mas` is rejected (GGUF exposes no
  hidden states).
- `--model_name` is still used to load the **HF tokenizer** for chat templating.
- Flags: `--llamacpp_model_path` (required), `--llamacpp_n_ctx`,
  `--llamacpp_n_gpu_layers` (`-1` = all on GPU), `--llamacpp_n_threads`,
  `--llamacpp_verbose`. See `configs/llamacpp.yaml` for documented defaults.
- Requires `pip install -e .[llamacpp]`.

---

## 6. Fine-tuning with Unsloth (`train.py`)

Config-driven LoRA SFT. The trainer is modular and only imports Unsloth when actually
run (`pip install -e .[unsloth]`).

```bash
python train.py --config configs/unsloth_train.yaml

# override any nested value inline:
python train.py --config configs/unsloth_train.yaml \
  --set model.model_name=Qwen/Qwen3-4B training.max_steps=120
```

`configs/unsloth_train.yaml` sections:
- **`model`** — `model_name`, `max_seq_length`, `load_in_4bit`, `dtype`.
- **`lora`** — `r`, `lora_alpha`, `lora_dropout`, `target_modules`, etc.
- **`data`** — `dataset_name` (HF id **or** local `.json`/`.jsonl`), `dataset_split`,
  `text_field`, or `messages_field` to render chat templates.
- **`training`** — batch size, grad accumulation, `max_steps` (> 0 overrides
  `num_train_epochs`), `learning_rate`, `optim`, `seed`, ...
- **`save`** — `save_method` (`lora` / `merged_16bit` / `merged_4bit`), `output_dir`,
  and `gguf: true` (+ `gguf_quant`) to export a GGUF.

**Bridge to llama.cpp:** set `save.gguf: true`, then serve the result with
`--use_llamacpp --llamacpp_model_path outputs/.../model.gguf`. The pipeline does this
in one go with `RUN_TRAINING=1 BACKEND=llamacpp ...`.

Programmatic use:

```python
from training import UnslothTrainer
from utils import load_yaml_config
UnslothTrainer(load_yaml_config("configs/unsloth_train.yaml")).run()
```

---

## 7. Latent-reasoning analysis (`src/`)

Research/analysis scripts live under `src/multilingual-latent-reasoning/`. They add the
repo root to `sys.path` themselves, so **launch them from the repository root**:

```bash
# agent-similarity / latent-space emergence
python src/multilingual-latent-reasoning/run_latent_mas_agent_similarity.py \
  --model_name Qwen/Qwen3-4B \
  --languages bn,de,en,es,fr,ja,ru,sw,te,th,zh \
  --ref_lang en --prompt sequential --latent_steps 3 --device cuda
```

Other entrypoints in that folder: `run.py` (trace generation), `run_truncation.py`,
`run_logitlens_dynamics.py`, `run_save_hidden_states.py`,
`run_text_mas_agent_similarity.py`, `run_latent_mas_mgsm_batch_analysis.py`, plus the
`analysis/` utilities and the folder's own `run.sh`/`run_truncations.sh` wrappers.
See `src/multilingual-latent-reasoning/README.md` for the experimental workflow.

> The similarly named `multilingual-latent-reasoner/` (note: *reasoner*) is an
> **external git submodule**, not part of this codebase.

---

## 8. Helper scripts

| Script | What it does |
|--------|--------------|
| `scripts/run_pipeline.sh` | End-to-end pipeline (see §3) |
| `scripts/run_mgsm_all.sh` | Sweep a method over all MGSM languages — `./scripts/run_mgsm_all.sh [MODEL] [METHOD] [PROMPT] [DEVICE]` |
| `scripts/run_mgsm_text_mas.sh` | Same sweep, defaulting to `text_mas` |

Example:
```bash
./scripts/run_mgsm_all.sh Qwen/Qwen3-4B latent_mas sequential cuda
```

---

## 9. Troubleshooting

- **`ImportError: llama-cpp-python is not installed`** → `pip install -e .[llamacpp]`.
- **`latent_mas` rejected with llama.cpp** → expected; GGUF has no hidden states. Use
  `hf`/`vllm`, or switch the method to `baseline`/`text_mas`.
- **`Choose only one inference backend`** → don't set both `--use_vllm` and
  `--use_llamacpp` (or `BACKEND` to two things).
- **Analysis script can't import `data`/`models`** → run it from the **repo root**;
  these scripts resolve the root via `Path(__file__).resolve().parents[2]`.
- **vLLM OOM** → lower `--gpu_memory_utilization` or raise `--tensor_parallel_size`.
- **Gated/slow model downloads** → `export HF_TOKEN=...`.
- **Editable install can't find packages** → only `methods` and `training` are
  packaged; the `src/` analysis code runs as standalone scripts by design.
