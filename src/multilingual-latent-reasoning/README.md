# Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners

This repository contains the codebase for the paper **_Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners_**.

We study **latent reasoning** in large reasoning models (LRMs) across **11 languages**, using **truncation-based probing**, **logit lens analysis**, and **representation-level similarity analysis** to understand how predictions emerge internally before answers are explicitly produced in the reasoning traces.

---

## Overview

The codebase supports four core components of the paper:

1. **Reasoning trace generation** across languages  
2. **Truncation-based latent reasoning evaluation**  
3. **Internal dynamics analysis**, including logit lens and hidden-state similarity  
4. **Memorization vs. latent reasoning**

The overall workflow mirrors the experimental pipeline in the paper.

---
## Repository Structure

```
.
├── README.md
├── analysis
│ ├── calculate_trunc_acc.py
│ ├── compute_cosine_sim.py
│ ├── compute_all_pairs_cosine_similarity.py
│ ├── create_counterfactuals.py
│ ├── run_memorization_counterfactual_vllm.py
│ ├── run_counterfactual_solvability_gemini.py
│ ├── memorization_test_helper.py
│ └── helper.py
├── instructions.json
├── hack_prefix.json
├── helper.py
├── run.py
├── run.sh
├── run_truncation.py
├── run_truncations.sh
├── run_logitlens_dynamics.py
├── run_logitlens_dynamics.sh
└── run_save_hidden_states.py
```

---

## Core Components

### 🔹 Reasoning Trace Generation

- **`run.sh`**  
  Generates full chain-of-thought reasoning traces for each language.

- **`run.py`**  
  Main driver script for multilingual reasoning generation and evaluation.

---

### 🔹 Truncation-Based Latent Reasoning

These scripts probe *when* correct predictions emerge as reasoning traces are progressively truncated.

- **`run_truncation.py`**  
  Evaluates model accuracy given partial reasoning traces.

- **`run_truncations.sh`**  
  Runs truncation experiments across languages, models, and truncation ratios.

This corresponds to the **latent prediction formation** analysis in the paper.

---

### 🔹 Internal Prediction Dynamics

#### Logit Lens Analysis
Tracks how the model’s predicted answer evolves across layers and reasoning steps.

- **`run_logitlens_dynamics.py`**  
- **`run_logitlens_dynamics.sh`**

#### Hidden-State Representation Analysis
Used for cross-lingual similarity and alignment analysis.

- **`run_save_hidden_states.py`**  
  Saves the hidden state of the last token at each reasoning step for a fixed set of truncation ratios.

---

## Analysis Utilities (`analysis/`)

This directory contains scripts for post-hoc analysis and robustness checks:

- **`calculate_trunc_acc.py`**  
  Computes truncation-based accuracy metrics.

- **`compute_cosine_sim.py`**, **`compute_all_pairs_cosine_similarity.py`**  
  Representation similarity analysis across languages and layers.

- **`create_counterfactuals.py`**  
  Generates controlled counterfactual edits for memorization vs. reasoning tests.

- **`run_memorization_counterfactual_vllm.py`**  
  Generates the responses from LRMs on the editted counterfactuals.

- **`run_counterfactual_solvability_gemini.py`**  
  Counterfactual solvability and memorization diagnostics.

- **`memorization_test_helper.py`**, **`helper.py`**  
  Shared analysis utilities.

---

## Typical Workflow

> This code lives at `src/multilingual-latent-reasoning/` within the
> [MultilingualLatentMAS](../../README.md) repo. The commands below assume you are
> **inside this directory** (`cd src/multilingual-latent-reasoning`). The
> `run_*_agent_similarity.py` / `run_latent_mas_mgsm_batch_analysis.py` scripts add the
> repo root to `sys.path` automatically, so they may also be launched from the repo root.

```bash
# 1. Generate multilingual reasoning traces
bash run.sh

# 2. Run truncation-based latent reasoning evaluation
bash run_truncations.sh

# 3. Analyze internal prediction dynamics (logit lens)
bash run_logitlens_dynamics.sh

# 4. Save hidden states for representation similarity analysis
python run_save_hidden_states.py
```

---

## Citation

If you use this code, please cite:

```bibtex
@misc{liu2026largereasoning,
      title={Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners}, 
      author={Yihong Liu and Raoyuan Zhao and Hinrich Schütze and Michael A. Hedderich},
      year={2026},
      eprint={2601.02996},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.02996}, 
}
```

We use the prompt templates and the prompt-hacking prefixes from the previous papers, please also consider citing:

```bibtex
@misc{zhao2025comprehensive,
      title={A Comprehensive Evaluation of Multilingual Chain-of-Thought Reasoning: Performance, Consistency, and Faithfulness Across Languages}, 
      author={Raoyuan Zhao and Yihong Liu and Hinrich Schütze and Michael A. Hedderich},
      year={2025},
      eprint={2510.09555},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.09555}, 
}
```

```bibtex
@inproceedings{qi-etal-2025-models,
    title = "When Models Reason in Your Language: Controlling Thinking Language Comes at the Cost of Accuracy",
    author = "Qi, Jirui  and
      Chen, Shan  and
      Xiong, Zidi  and
      Fern{\'a}ndez, Raquel  and
      Bitterman, Danielle  and
      Bisazza, Arianna",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.1103/",
    doi = "10.18653/v1/2025.findings-emnlp.1103",
    pages = "20279--20296"
}
```
