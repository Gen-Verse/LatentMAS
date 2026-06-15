# Setup & Run
- acivate conda env `conda activate vllm_env`
- `export HF_TOKEN="hf_..."` and use a hugging face token so you aren't subject to anonymous rate limits
- **run baseline** `python run.py --method baseline --model_name ./models/qwen3-4b --task gsm8k --max_samples 1 --generate_bs 1 --device mps`
- **run latentMAS** `python run.py --method latent_mas --model_name ./models/qwen3-4b --task gsm8k  --max_samples 1 --generate_bs 1 --device mps --prompt sequential`

- **test on mgsm** `langs=(bn de en es fr ja ru sw te th zh) #you can choose your language you want to run in, this just generates a sample q **Baseline:**
```
for L in "${langs[@]}"; do
  echo "=== $L ==="
  python run.py --method baseline --model_name Qwen/Qwen3-4B \
    --task mgsm --mgsm_lang $L --max_samples 1 --generate_bs 1 --device mps ##this is for my mac, with another device probably look into torch cuda or cpu`

```
**Explicit Multiagent Communication**

```
langs=(bn de en es fr ja ru sw te th zh)
for L in "${langs[@]}"; do
  echo "=== $L ==="
  python run.py --method text_mas --model_name Qwen/Qwen3-4B \
    --task mgsm --mgsm_lang $L --max_samples 1 --generate_bs 1 --device mps
```


**Latent Multiagent Communication**
```
langs=(bn de en es fr ja ru sw te th zh)
for L in "${langs[@]}"; do
  echo "=== $L ==="
  python run.py --method latent_mas --model_name Qwen/Qwen3-4B \
    --task mgsm --mgsm_lang $L --max_samples 1 --generate_bs 1 --device mps
```


- run the **whole pipeline** (optional fine-tuning → evaluation) from one config:
  `./scripts/run_pipeline.sh` (see `configs/pipeline.env`).

- how to observe **latent space** (analysis code lives under `src/`; run from the repo root):
```
python src/multilingual-latent-reasoning/run_latent_mas_agent_similarity.py \
  --model_name Qwen/Qwen3-4B \
  --languages bn,de,en,es,fr,ja,ru,sw,te,th,zh \
  --ref_lang en \
  --prompt sequential \
  --latent_steps 3 \
  --device mps \
  --emergence_rank_threshold 1000 \
  --emergence_layer_strategy final_layer
```

# Prior Work - Monolingual LatentMAS



<a name="readme-top"></a>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo.png">
    <img alt="LatentMAS" src="assets/logo.png" width=500>
  </picture>
</p>

<h3 align="center">
Latent Collaboration in Multi-Agent Systems
</h3>



<p align="center">
    <a href="https://arxiv.org/abs/2511.20639"><img src="https://img.shields.io/badge/arXiv-2511.20639-B31B1B.svg?logo=arxiv" alt="Arxiv"></a>
    <a href="https://huggingface.co/papers/2511.20639"><img src="https://img.shields.io/badge/Huggingface-DailyPaper-FFD21E.svg?logo=huggingface" alt="Huggingface Paper"></a>
    <a href="https://x.com/Jiaru_Zou/status/1994724438135169196"><img src="https://img.shields.io/badge/Coverage-LatentMAS-2176BC.svg?logo=x" alt="X"></a>
    <a href="https://github.com/Gen-Verse/LatentMAS/tree/Science-LatentMAS"><img src="https://img.shields.io/badge/Science--LatentMAS-Branch-2D8CFF.svg?logo=github" alt="Science-LatentMAS Branch"></a>
  </p>

---

<p align="center">
  <img src="assets/main_res.png" width="1000">
</p>

## 💡 Introduction


**LatentMAS** is a multi-agent reasoning framework that **moves agent collaboration from token space into the model’s latent space**.  
Instead of producing long textual reasoning traces, agents communicate by **passing latent thoughts** through their own **working memory**. LatentMAS has the following key features:

- **Efficient** multi-step reasoning with drastically fewer tokens  
- **Training-free** latent-space alignment for stable generation  
- **A general technique** compatible with **any HF model** and optionally **vLLM** backends.

Overall, LatentMAS achieves **superior performance**, **lower token usage**, and **major wall-clock speedups** of the multi-agent system.

<p align="center">
  <img src="assets/main.png" width="1000">
</p>


## 🔔 News
- **[2026-05-01]** LatentMAS has been accepted into ICML 2026 as a **spotlight** ! 
- **[2026-02-26]** 🦞 Check out [**OpenClaw-RL**](https://github.com/Gen-Verse/OpenClaw-RL) from our Gen-Verse group! OpenClaw-RL is a fully asynchronous RL framework that trains personalized AI agents directly from natural conversation feedback — no manual labels, no API keys. It introduces two learning paradigms (Binary RL via GRPO and On-Policy Distillation) and runs the entire stack on your own infrastructure. A great complement to LatentMAS's efficient multi-agent reasoning! 
- **[2025-12-20]** Check [**Science-LatentMAS**](https://github.com/Gen-Verse/LatentMAS/tree/Science-LatentMAS), an excellent extension of LatentMAS developed by Prof. Markus J. Buehler and the [LAMM Lab](https://github.com/lamm-mit) at MIT. Science-LatentMAS is specifically designed for the scientific discovery downstream applications! For more details and instructions, please check our README section "Science-LatentMAS" below and the new `Science-LatentMAS` branch.
- **[2025-12-15]** Check out these amazing community-driven extensions of LatentMAS!
  - **[KNN-LatentMAS](https://github.com/Bookmaster9/kNN-latentMAS)** — Enables more efficient KV utilization for latent memory.
  - **[Hybrid-LatentMAS](https://github.com/nhminle/LatentMAS-Hybrid)** — Extends LatentMAS to support hybrid, heterogeneous multi-agent systems.

- **[2025-11-25]** We have released our paper and code implementations for LatentMAS! Stay tuned for more model-backbone supports and advanced features!
- **[2025-11-25]** We are featured as 🤗 [**HuggingFace 1st Paper of the Day**](https://huggingface.co/papers/2511.20639)!


## 🌐 Awesome Works Built on Top of LatentMAS

Explore community-driven extensions that expand LatentMAS into new domains, architectures, and collaboration patterns:


### 🔬 1. **Science-LatentMAS**
**By Prof. Markus J. Buehler & MIT LAMM Group**  
- **New Branch:** https://github.com/Gen-Verse/LatentMAS/tree/Science-LatentMAS  
- **Original Code:** https://github.com/lamm-mit/LatentMAS/tree/flexible_agents  
**New Features:** Extends LatentMAS for scientific modeling and material-system collaboration, enabling flexible agent types and specialized latent communication for science domains.


### 🧠 2. **KNN-LatentMAS**
**By Bookmaster9**
- **Blog (Overview):** https://bookmaster9.github.io/kNN-latentMAS/  
- **Code:** https://github.com/Bookmaster9/kNN-latentMAS  
- **New Features:** Introduce kNN-based latent retrieval to improve KV-cache usage, boosting memory efficiency and multi-step reasoning stability across agents.

### 🤖 3. **Hybrid-LatentMAS**
**By nhminle**
- **Code:** https://github.com/nhminle/LatentMAS-Hybrid  
- **New Features:** Support heterogeneous/hybrid agent collaboration (LLM + non-LLM agents), enabling modular multi-agent pipelines that mix models, tools, and reasoning strategies.


### 🌍 4. **Awareness Network**
**By Everest-AN**
- **Website:** https://awareness.market/
- **Code:** https://github.com/everest-an/Awareness-Market
- **New Features:** A decentralized AI awareness market product built on LatentMAS research, enabling autonomous agent collaboration and memory sharing.

### 🧩 5. LatentMAS-SLoRA
**By Arifuzzaman Joy**
- **Demo:** https://www.youtube.com/watch?v=g7sxYjwgRRk
- **Code:** https://github.com/Arifuzzamanjoy/latent_mas_slora
- **New Features:** Augment LatentMAS with role-specialized, dynamically switchable LoRA adapters for better specialization and adaptability.

### 🛰️ 6. AVP (Agent Vector Protocol)
**By VectorArc**
- **Blog:** https://blog.avprotocol.ai/avp-binary-protocol-latent-agent-communication/
- **Code:** https://github.com/VectorArc/avp-python
- **New Features:** Enables agents to share KV-cache and hidden states instead of text, supporting zero-training latent handoff, cross-model transfer, and faster multi-agent collaboration.

**If your work extends LatentMAS, feel free to open a PR and we’ll feature it here! 🚀**


## 📊 Experiments Overview

### ⭐ Main Results  
Three main tables from our paper spanning 9 tasks across math & science reasoning, commensonse reasoning, and code generation:

- **Table 1 — LatentMAS under the Sequantial MAS setting**  
  <p align="center"><img src="assets/main_table1.png" width="1000"></p>

- **Table 2 — LatentMAS under the Hierarchical MAS setting**  
  <p align="center"><img src="assets/main_table2.png" width="1000"></p>

- **Table 3 — Main Results on Reasoning Intensive Tasks**
  <p align="center"><img src="assets/main_table3.png" width="1000"></p>


### ⚡ Superior Efficiency on **Time and Tokens**

Overall, LatentMAS reduces:
- **~50–80% tokens**
- **~3×–7× wall-clock time**
compared to standard Text-MAS or chain-of-thought baselines.


## 🛠️ Getting Started

This repository provides all code for reproducing LatentMAS, TextMAS, and baseline single-agent experiments across GSM8K, AIME24/25, GPQA, ARC-Easy/Challenge, MBPP+, HumanEval+, and MedQA.

### ⚙️ Setup Environment Variables

We recommend setting your HF cache directory to avoid repeated downloads:

```bash
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
````

Models and datasets will automatically be downloaded into `$HF_HOME`.


### 📦 Install Packages

```bash
conda create -n latentmas python=3.10 -y
conda activate latentmas

pip install -r requirements.txt
```

If you want **vLLM support**, also install:

```bash
pip install vllm
```

### 🧩 Optional integrations (modular)

The repo ships optional, import-guarded extras. Install only what you need — the
core inference path never requires them.

```bash
pip install -e .[vllm]       # vLLM fast inference
pip install -e .[llamacpp]   # llama.cpp (GGUF) fast inference
pip install -e .[unsloth]    # Unsloth fast LoRA training/fine-tuning
pip install -e .[all]        # everything
```

**Llama.cpp inference** — serve a local GGUF model for fast text generation
(`baseline` / `text_mas` only; `latent_mas` needs hidden states and is unsupported):

```bash
python run.py --method baseline --model_name Qwen/Qwen3-4B --task gsm8k \
  --use_llamacpp --llamacpp_model_path /path/to/model-Q4_K_M.gguf \
  --llamacpp_n_ctx 4096 --llamacpp_n_gpu_layers -1
```

See `configs/llamacpp.yaml` for the documented knobs.

**Unsloth training/fine-tuning** — config-driven LoRA SFT:

```bash
python train.py --config configs/unsloth_train.yaml
# override any value inline:
python train.py --config configs/unsloth_train.yaml --set training.max_steps=120
```

Set `save.gguf: true` in the config to export the fine-tuned model to GGUF and
serve it back through the `--use_llamacpp` backend above.

### 🧰 Global config & full pipeline

Instead of passing flags by hand, keep all variable settings in one place —
`configs/pipeline.env` — and run the whole pipeline (optional fine-tuning →
evaluation with the chosen backend) with a single script:

```bash
# edit configs/pipeline.env, then:
./scripts/run_pipeline.sh

# or override any variable inline (no file edits needed):
BACKEND=vllm METHOD=baseline TASK=gsm8k ./scripts/run_pipeline.sh
RUN_TRAINING=1 BACKEND=llamacpp LLAMACPP_MODEL_PATH=/path/model.gguf ./scripts/run_pipeline.sh
```

`configs/pipeline.env` controls the model, task, method, generation params,
inference `BACKEND` (`hf` | `vllm` | `llamacpp`), the training stage
(`RUN_TRAINING`), and where logs land (`RESULTS_DIR`). The script picks the right
`run.py` flags for the selected backend and tees output to a timestamped log under
`results/`. Pass an alternate env file as the first argument:
`./scripts/run_pipeline.sh path/to/my.env`.

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Gen-Verse/LatentMAS.git
cd LatentMAS
```

### 2. Repository Structure

```
LatentMAS/
│── run.py                 # Main entry for experiments
│── train.py               # Optional Unsloth fine-tuning entry
│── models.py              # Wrapper for HF + vLLM + llama.cpp + latent realignment
│── methods/
│   ├── baseline.py        # Single-agent baseline
│   ├── text_mas.py        # Token-space multi-agent method
│   └── latent_mas.py      # Latent-space multi-agent (our method)
│── training/              # Optional, modular Unsloth LoRA trainer
│── configs/
│   ├── pipeline.env       # Global pipeline config (sourced by run_pipeline.sh)
│   ├── unsloth_train.yaml # Unsloth fine-tuning config
│   └── llamacpp.yaml      # llama.cpp backend reference config
│── scripts/
│   ├── run_pipeline.sh    # End-to-end: optional training -> evaluation
│   ├── run_mgsm_all.sh    # Sweep LatentMAS over MGSM languages
│   └── run_mgsm_text_mas.sh
│── prompts.py             # Prompt constructors
│── data.py                # Dataset loaders
│── data/                  # Provided data + figures (We give medqa.json as an example here)
│── utils.py               # Answer parsing / timeout / config loader / helpers
│── example_logs/          # Example logs from LatentMAS
│── src/
│   └── multilingual-latent-reasoning/   # Latent-reasoning analysis scripts (truncation,
│                                        # logit lens, similarity); run from repo root
│── multilingual-latent-reasoner/        # Git submodule (cisnlp/multilingual-latent-reasoner)
│── pyproject.toml         # Packaging + optional extras (vllm / llamacpp / unsloth / all)
│── requirements.txt
```

> The analysis code under `src/multilingual-latent-reasoning/` adds the repo root to
> `sys.path` itself, so launch those scripts from the repository root. The
> `multilingual-latent-reasoner/` directory is a git submodule — clone with
> `git clone --recurse-submodules`, or run `git submodule update --init` after cloning.


## 🧪 Running Experiments (standard HF backend)

### 🔹 **Baseline (single model)**

```bash
python run.py --method baseline --model_name Qwen/Qwen3-14B --task gsm8k --max_samples -1 --max_new_tokens 2048
```


### 🔹 **TextMAS (text based multi-agent system)**

```bash
python run.py --method text_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048
```


### 🔹 **LatentMAS (our latent mas method)**

```bash
# 4B example command
python run.py --method latent_mas --model_name Qwen/Qwen3-4B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048

# 8B example command
python run.py --method latent_mas --model_name Qwen/Qwen3-8B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048

# 14B example command
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048
```

#### Notes:

* **`--latent_steps`** ∈ [0, 80]
  Tune for best performance.
* **`--latent_space_realign`**
  Enables latent→embedding alignment
  We treat this as a **hyperparameter** — enable/disable depending on task/model:

```bash
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --latent_space_realign --max_new_tokens 2048
```


## 📘 Example Logs

Two example LatentMAS logs are provided for reference purposes:

* `example_logs/qwen3_14b_mbppplus_sequential.txt`
* `example_logs/qwen3_14b_humanevalplus_hierarchical.txt`


Please refer to additional experiment logs [here](https://drive.google.com/drive/folders/1evGv5YAmLb4YM_D9Yu0ABa1nfqHC5N-l?usp=drive_link).
You can open them to view the full agent interaction traces and outputs.


## ⚡ vLLM Integration

LatentMAS supports vLLM for faster inference.

### 🔹 Baseline with vLLM

```bash
python run.py --method baseline --model_name Qwen/Qwen3-14B --task gsm8k --max_samples -1 --use_vllm --max_new_tokens 2048
```

### 🔹 TextMAS with vLLM

```bash
python run.py --method text_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --use_vllm --max_new_tokens 2048
```

### 🔹 LatentMAS with vLLM

LatentMAS supports a **hybrid HF + vLLM pipeline** for fast inference:
- vLLM handles **final text generation** (with prefix caching, tensor parallelism, etc.)
- A HuggingFace model handles **latent-space rollout** and hidden-state alignment

For this setup, we recommend using two GPUs:
- One GPU for vLLM (`--device`, e.g., `cuda:0`)
- One GPU for the auxiliary HF model (`--device2`, e.g., `cuda:1`)

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048 \
  --use_vllm \
  --use_second_HF_model \
  --enable_prefix_caching \
  --device2 cuda:1
```

**📍Important Note:**

> vLLM does **not** officially support modifying KV-cache or prompting via latent embeddings.
> We modify the partial inner package inside vLLM backend for our method implementation.
> Note minor numeric differences may arise compared to offical HF backend due to different decoding (generation) strategies. Please Use the HF backend to reproduce the official published results.

## 📚 Citation

💫 If you find **LatentMAS** helpful, please kindly give us a star ⭐️ and cite below. Thanks!

```
@inproceedings{
zou2025latentmas,
  title={Latent Collaboration in Multi-Agent Systems},
  author={Jiaru Zou and Ruizhong Qiu and Gaotang Li and Xiyuan Yang and Katherine Tieu and Pan Lu and Ke Shen and Hanghang Tong and Yejin Choi and Jingrui He and James Zou and Mengdi Wang and Ling Yang},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026}
}
```

## 🤝 Ackowledgement 

This code is partially based on the amazing work of [vLLM](https://github.com/vllm-project/vllm).
