<a name="readme-top"></a>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo.png">
    <img alt="ScienceLatentMAS" src="assets/logo.png" width=500>
  </picture>
</p>

<h3 align="center">
Latent Collaboration in Multi-Agent Systems, based on the LatentMAS repository.
</h3>



<p align="center">
    <a href="https://arxiv.org/abs/2511.20639"><img src="https://img.shields.io/badge/arXiv-2511.20639-B31B1B.svg?logo=arxiv" alt="Arxiv"></a>
    <a href="https://huggingface.co/papers/2511.20639"><img src="https://img.shields.io/badge/Huggingface-DailyPaper-FFD21E.svg?logo=huggingface" alt="Huggingface Paper"></a>
    <a href="https://x.com/LingYang_PU/status/1993510834245714001"><img src="https://img.shields.io/badge/Coverage-LatentMAS-2176BC.svg?logo=x" alt="X"></a>
  
  </p>

---

<p align="center">
  <img src="assets/main_res.png" width="1000">
</p>

## 💡 Introduction


**LatentMAS** is a flexible multi-agent reasoning framework that **moves agent collaboration from token space into the model’s latent space**.  This repo extends the original code to have more flexibility. 

Instead of producing long textual reasoning traces, agents communicate by **passing latent thoughts** through their own **working memory**. LatentMAS has the following key features:

- **Efficient** multi-step reasoning with drastically fewer tokens  
- **Training-free** latent-space alignment for stable generation  
- **A general technique** compatible with **any HF model** and optionally **vLLM** backends.

<p align="center">
  <img src="assets/main.png" width="1000">
</p>


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
│── models.py              # Wrapper for HF + vLLM + latent realignment
│── methods/
│   ├── baseline.py        # Single-agent baseline
│   ├── text_mas.py        # Token-space multi-agent method
│   └── latent_mas.py      # Latent-space multi-agent (our method)
│── prompts.py             # Prompt constructors
│── data.py                # Dataset loaders
│── data/                  # Provided data + figures (We give medqa.json as an example here)
│── utils.py               # Answer parsing / timeout / helpers
│── example_logs/          # Example logs from LatentMAS
│── requirements.txt
```


## 🧪 Running Experiments (standard HF backend)

### 🔹 **Baseline (single model)**

```bash
python run.py --method baseline --model_name Qwen/Qwen3-14B --task gsm8k --max_samples -1
```


### 🔹 **TextMAS (text based multi-agent system)**

```bash
python run.py --method text_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1
```


### 🔹 **LatentMAS (our latent mas method)**

```bash
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1
```

#### Notes:

* **`--latent_steps`** ∈ [0, 80]
  Tune for best performance.
* **`--latent_space_realign`**
  Enables latent→embedding alignment
  We treat this as a **hyperparameter** — enable/disable depending on task/model:

```bash
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --latent_space_realign
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
python run.py --method baseline --model_name Qwen/Qwen3-14B --task gsm8k --max_samples -1 --use_vllm
```

### 🔹 TextMAS with vLLM

```bash
python run.py --method text_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --use_vllm
```

### 🔹 LatentMAS with vLLM

LatentMAS supports a **hybrid HF + vLLM pipeline** for fast inference:
- vLLM handles **final text generation** (with prefix caching, tensor parallelism, etc.)
- A HuggingFace model handles **latent-space rollout** and hidden-state alignment

For this setup, we recommend using two GPUs:
- One GPU for vLLM (`--device`, e.g., `cuda:0`)
- One GPU for the auxiliary HF model (`--device2`, e.g., `cuda:1`)

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 \
  --use_vllm \
  --use_second_HF_model \
  --enable_prefix_caching \
  --device2 cuda:1
```

**📍Important Note:**

> vLLM does **not** officially support modifying KV-cache or prompting via latent embeddings.
> We modify the partial inner package inside vLLM backend for our method implementation.
> Note minor numeric differences may arise compared to offical HF backend due to different decoding (generation) strategies. Please Use the HF backend to reproduce the official published results.

## 🎯 Advanced Features

### 🔀 Understanding Prompting Modes: Sequential vs Hierarchical

LatentMAS supports two distinct multi-agent collaboration patterns controlled by the `--prompt` flag:

#### **Sequential Mode** (Workflow Pattern)

Agents work in a **sequential workflow**, where each agent builds directly on the previous agent's work:

```bash
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential
```

**Example workflow for a research question:**

1. **Planner Agent**: Creates a step-by-step plan
   - Prompt: "Design a clear, step-by-step plan for how to solve the question"
   - Output: A detailed plan (in latent space)

2. **Critic Agent**: Reviews and critiques the plan
   - Prompt: "Evaluate the correctness of the input plan and provide helpful feedback"
   - Uses: The planner's output (via KV cache)
   - Output: Feedback on the plan

3. **Refiner Agent**: Improves the plan based on feedback
   - Prompt: "Provide a refined step-by-step plan based on previous plan with feedback"
   - Uses: Both planner + critic outputs
   - Output: Refined plan

4. **Judger Agent**: Produces final answer
   - Prompt: "Solve the target question using latent information for reference"
   - Uses: All previous agent outputs
   - Output: Final detailed answer

**Sequential Pattern:** `Build → Critique → Refine → Solve`

---

#### **Hierarchical Mode** (Parallel Perspectives)

Agents approach the **same question from different perspectives** in parallel, then combine their insights:

```bash
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt hierarchical
```

**Example workflow for the same research question:**

1. **Planner (Math Agent)**: Solves from mathematical perspective
   - Prompt: "You are a math agent. Reason step-by-step and put the final answer inside \boxed{}"
   - Output: Mathematical analysis + answer

2. **Critic (Science Agent)**: Solves from scientific perspective
   - Prompt: "You are a science agent. Reason step-by-step and put the final answer inside \boxed{}"
   - Output: Scientific analysis + answer

3. **Refiner (Code Agent)**: Solves from computational perspective
   - Prompt: "You are a code agent. Reason step-by-step and put the final answer inside \boxed{}"
   - Output: Computational analysis + answer

4. **Judger (Synthesizer)**: Combines all perspectives
   - Prompt: "You are a task summarizer. Given responses from previous agents as reference..."
   - Uses: All three parallel analyses
   - Output: Synthesized final answer

**Hierarchical Pattern:** `Multiple Parallel Perspectives → Synthesis`

---

#### **Key Differences Summary**

| Aspect | Sequential | Hierarchical |
|--------|-----------|--------------|
| **Agent roles** | Specialized workflow stages | Different domain experts |
| **Information flow** | Linear chain (A→B→C→D) | Parallel then merge (A,B,C→D) |
| **Each agent** | Performs different tasks | Solves same problem differently |
| **Best for** | Complex planning/refinement | Multi-perspective analysis |
| **Example use case** | Multi-step problem solving with feedback loops | Combining math, science, and code perspectives |

#### **When to Use Each Mode**

- **Use Sequential** when:
  - You need iterative refinement (plan → critique → improve)
  - The problem benefits from explicit feedback loops
  - You want a clear workflow of distinct tasks

- **Use Hierarchical** when:
  - You want diverse perspectives on the same problem
  - Different domain expertise adds value (math + science + code)
  - You need to combine multiple reasoning approaches

Both modes can be combined with custom prompts (via `--custom_prompt_file`) to define exactly how each agent behaves.

### Custom Prompts

LatentMAS supports fully customizable prompts via JSON configuration files. This allows you to tailor agent behavior for specific domains or tasks.

#### Creating a Custom Prompt File

Create a JSON file (e.g., `prompts.json`) with role-specific prompts:

```json
{
  "system": "You are a helpful research assistant.",
  "planner": "Draft a response to the question.\n\nQuestion:\n{question}",
  "critic": "Review the prior answer critically. Give specific, actionable feedback.\n\nQuestion:\n{question}",
  "refiner": "Produce an improved answer that incorporates the Critic's feedback.\n\nQuestion:\n{question}",
  "judger": "Answer the question using prior context as hints. Provide a detailed response.\n\nQuestion:\n{question}",
  "baseline": "You are a problem solver. Reason step by step.\n\nQuestion:\n{question}"
}
```

**Supported placeholders:**
- `{question}` - The input question
- `{context}` - Previous agent outputs (for text_mas)

**System message behavior:**
- If `"system"` is provided and non-empty, it will be included in the message list
- If `"system"` is empty (`""`) or omitted, no system message is added
- Works across all methods: baseline, text_mas, and latent_mas

#### Using Custom Prompts

```bash
# For LatentMAS with custom prompts
python run.py --method latent_mas \
  --model_name Qwen/Qwen3-14B \
  --task custom \
  --custom_question "What is the structure of spider silk?" \
  --prompt hierarchical \
  --custom_prompt_file prompts_advanced.json

# For TextMAS with custom prompts
python run.py --method text_mas \
  --model_name Qwen/Qwen3-14B \
  --task gsm8k \
  --prompt sequential \
  --custom_prompt_file prompts_advanced.json

# For Baseline with custom prompts
python run.py --method baseline \
  --model_name Qwen/Qwen3-14B \
  --task custom \
  --custom_question "Explain photosynthesis." \
  --custom_prompt_file prompts_advanced.json
```

### 💬 Custom Questions and Tasks

Run LatentMAS on your own questions using the `--task custom` option:

```bash
# Direct question via command line
python run.py --method latent_mas \
  --model_name Qwen/Qwen3-14B \
  --task custom \
  --custom_question "Give me a research idea to make a new composite using silk." \
  --latent_steps 64 \
  --prompt hierarchical

# Question from a text file
python run.py --method latent_mas \
  --model_name Qwen/Qwen3-14B \
  --task custom \
  --custom_question_file my_question.txt \
  --latent_steps 50

# With optional gold answer for evaluation
python run.py --method latent_mas \
  --model_name Qwen/Qwen3-14B \
  --task custom \
  --custom_question "What is 2+2?" \
  --custom_gold "4"
```

### 🔧 Model-Specific Settings: Qwen Enforcement

LatentMAS may work with any HuggingFace model without model-specific requirements (further testing is needed, especiallly for benchmark performance). To do this you need to turn off enforcing Qwen-specific behavior (system message and model name validation), use the `--do_not_enforce_qwen` flag:

```bash
# Works with any model
python run.py --method latent_mas \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --task gsm8k \
  --prompt sequential --do_not_enforce_qwen

```
### Customizable Thinking Tokens

Control the thinking prompt tokens that trigger reasoning mode in your model:

```bash
# Use default thinking tokens: <think>\n
python run.py --method latent_mas \
  --model_name Qwen/Qwen3-14B \
  --task gsm8k \
  --think

# Use custom thinking tokens
python run.py --method latent_mas \
  --model_name Qwen/Qwen3-14B \
  --task gsm8k \
  --think "<reasoning>\n"

# Use custom thinking tokens
python run.py --method latent_mas \
  --model_name Qwen/Qwen3-14B \
  --task gsm8k \
  --think "<think>\n<brainstorm>\n"

# Use chain-of-thought style prompt
python run.py --method latent_mas \
  --model_name Qwen/Qwen3-14B \
  --task gsm8k \
  --think "Let's think step by step:\n"

# No thinking tokens (omit flag)
python run.py --method latent_mas \
  --model_name Qwen/Qwen3-14B \
  --task gsm8k
```

### Custom Agents and Agent Ordering

You can define custom agents and their execution order in your `prompts.json` file. The **last agent in your list** always generates the final text output, regardless of its role name.

**Define custom agents:**
```json
{
  "agents": [
    {"name": "Researcher", "role": "researcher"},
    {"name": "Analyst", "role": "analyst"},
    {"name": "Writer", "role": "writer"}
  ],
  "researcher": "Research the topic: {question}",
  "analyst": "Analyze the research findings.",
  "writer": "Write the final answer based on analysis."
}
```

**Key features:**
- **Flexible agent count**: Use 1, 2, 3, 4, 5+ agents
- **Custom roles**: Name agents based on their actual function
- **Last agent behavior**: The last agent in the list always produces decoded text output
- **All other agents**: Generate latent representations (compressed reasoning)

**Example with single agent:**
```json
{
  "agents": [
    {"name": "Expert", "role": "expert"}
  ],
  "expert": "Answer the question: {question}"
}
```

**Example with 5 agents:**
```json
{
  "agents": [
    {"name": "Planner", "role": "planner"},
    {"name": "Researcher", "role": "researcher"},
    {"name": "Critic", "role": "critic"},
    {"name": "Refiner", "role": "refiner"},
    {"name": "Synthesizer", "role": "synthesizer"}
  ]
}
```

### Hybrid Text-Latent Generation

The `--first_agent_text` flag enables the first agent to generate text instead of latent representations. This is particularly useful for:

**1. Graph reasoning models**: Models trained to express graphs in natural language can output their structure as text, then subsequent agents reason over it efficiently in latent space.

**2. Structured reasoning**: Models that benefit from explicitly generating intermediate structures (plans, outlines, decompositions) in text form before latent reasoning.

**3. GRPO-trained models**: Models trained via Group Relative Policy Optimization to produce specific text patterns can maintain their learned behavior while benefiting from latent efficiency.

```bash
python run.py --method latent_mas \
  --model_name Qwen/Qwen3-14B \
  --task custom \
  --custom_question "Design a molecular structure for a biodegradable plastic." \
  --latent_steps 64 \
  --first_agent_text \
  --max_new_tokens 3000
```

**How it works:**
- First agent generates explicit text output (e.g., graph structure, detailed plan)
- Middle agents use latent-space reasoning (efficient, compressed thinking)
- Last agent produces the final answer using all accumulated context

**Technical notes:**
- The first agent's generated text becomes part of the KV cache that subsequent agents attend to
- Set `--max_new_tokens` appropriately for the first agent's output complexity
- The `--latent_steps` parameter only affects middle agents (not first or last)
- EOS tokens are preserved to maintain chat template structure

### Complete Example: Research Question with Custom Setup

Combining all advanced features for a research task:

```bash
python run.py --method latent_mas \
  --model_name Qwen/Qwen3-14B \
  --task custom \
  --custom_question "Give me a research idea to make a composite inspired by spider silk." \
  --custom_prompt_file prompts_advanced.json \
  --prompt hierarchical \
  --latent_steps 64 \
  --think "<scientific_reasoning>\n" \
  --first_agent_text \
  --max_new_tokens 3000 \
  --temperature 0.4
```

This setup:
- Uses custom research-focused prompts from `prompts.json`
- Runs on a custom scientific question
- Uses hierarchical agent organization
- Applies custom thinking tokens
- First agent generates text for structured output
- Remaining agents use latent reasoning with 64 steps

### Complete Example: Sequential Workflow with Custom Agents

For a sequential workflow (plan → critique → refine) using custom agents:

```bash
python run.py --method latent_mas \
  --model_name Qwen/Qwen3-14B \
  --task custom \
  --custom_question "Give me a research idea to make a composite inspired by spider silk." \
  --latent_steps 64 \
  --prompt sequential \
  --custom_prompt_file prompts_advanced.json \
  --max_new_tokens 3000
```

**Example `prompts.json` file:**

```json
{
  "agents": [
    {"name": "Researcher", "role": "researcher"},
    {"name": "Critic", "role": "critic"},
    {"name": "Writer", "role": "writer"}
  ],
  "researcher": "Question:\n{question}\n\nBrainstorm and research the topic.",
  "critic": "Question:\n{question}\n\nCritically review the research.",
  "writer": "Question:\n{question}\n\nYou are provided with latent information for reference and a target question to solve.\n\nThe latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.\n\nWrite a final answer, very detailed, without outputting other irrelevant information."
}
```

**Workflow:**
1. **Researcher** → Brainstorms and researches (latent, 64 steps)
2. **Critic** → Reviews the research (latent, 64 steps)
3. **Writer** → Synthesizes into final answer (text output)

**Key differences from hierarchical mode:**
- Sequential builds chain of reasoning: Research → Critique → Write
- Each agent has a distinct role in the workflow
- Information flows linearly through agents
- Best for iterative refinement tasks

**Associated `prompts_advanced.json` file:**

```json
{
  "agents": [
    {"name": "Strategist", "role": "strategist"},
    {"name": "Investigator", "role": "investigator"},
    {"name": "Evaluator", "role": "evaluator"},
    {"name": "Synthesizer", "role": "synthesizer"},
    {"name": "Communicator", "role": "communicator"}
  ],
  "strategist": "Question:\n{question}\n\nYou are a strategic planning agent. Your task is to:\n1. Break down the question into key components\n2. Identify the core challenges and requirements\n3. Outline a high-level approach to address the question\n4. Highlight critical assumptions and constraints\n\nProvide a clear strategic framework for solving this problem.",
  "investigator": "Question:\n{question}\n\nYou are a deep investigation agent. Building on the strategic framework, your task is to:\n1. Conduct thorough analysis of each component\n2. Explore multiple perspectives and approaches\n3. Identify relevant principles, methods, and precedents\n4. Uncover potential challenges and opportunities\n5. Generate detailed insights and findings\n\nProvide comprehensive investigative findings.",
  "evaluator": "Question:\n{question}\n\nYou are a critical evaluation agent. Your task is to:\n1. Rigorously assess the strategic approach and investigative findings\n2. Identify logical flaws, gaps, or weaknesses\n3. Challenge assumptions and test robustness\n4. Propose improvements and alternative considerations\n5. Verify consistency and completeness\n\nProvide a thorough critical evaluation with specific recommendations for improvement.",
  "synthesizer": "Question:\n{question}\n\nYou are a synthesis agent. Your task is to:\n1. Integrate the strategy, investigation, and evaluation into a coherent whole\n2. Resolve conflicts and reconcile different perspectives\n3. Strengthen weak points identified in the evaluation\n4. Construct a comprehensive solution or answer\n5. Ensure logical flow and completeness\n\nProvide an integrated, well-reasoned synthesis.",
  "communicator": "Question:\n{question}\n\nYou are provided with latent information containing strategic planning, deep investigation, critical evaluation, and synthesis from previous agents.\n\nYour task is to:\n1. Extract the most valuable insights from the latent context\n2. Organize the information in a clear, logical structure\n3. Present a comprehensive, well-articulated final answer\n4. Ensure the response is precise, actionable, and complete\n5. Address the original question directly and thoroughly\n\nProvide a polished, professional final answer that leverages all prior analysis."
}
```

## 🌐 Awesome Works based on LatentMAS

1. KNN-LatentMAS: [Blog](https://bookmaster9.github.io/kNN-latentMAS/) and [Code](https://github.com/Bookmaster9/kNN-latentMAS).

## 📚 Citation

💫 If you find **LatentMAS** helpful, please kindly give us a star ⭐️ and cite below. Thanks!

```
@article{zou2025latentmas,
  title={Latent Collaboration in Multi-Agent Systems},
  author={Zou, Jiaru and Yang, Xiyuan and Qiu, Ruizhong and Li, Gaotang and Tieu, Katherine and Lu, Pan and Shen, Ke and Tong, Hanghang and Choi, Yejin and He, Jingrui and Zou, James and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2511.20639},
  year={2025}
}
```

## 🤝 Ackowledgement 

This code is partially based on the amazing work of [vLLM](https://github.com/vllm-project/vllm).

This code is based on the amazing [LatentMAS](https://github.com/Gen-Verse/LatentMAS) repo, adapted here for flexible use cases in scientific and technical applications.