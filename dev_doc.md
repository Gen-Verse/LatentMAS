# Multilingual Scaling Engine - Developer Documentation

Welcome to the development guide for the **Multilingual Latent MAS (Multi-Agent System) Engine**. This document outlines the architectural boundaries, the continuous latent reasoning mechanics, the comprehensive evaluation matrix, and our zero-tolerance policy against mock data.

## 1. Architectural Topography & System Firewall

The repository is divided into isolated zones to ensure strict mathematical and functional integrity:
*   **`src/latent_coordination/`**: The decentralized multi-agent hub. Handles text-free interaction graphs, recursive latent iterations, topology routing, and baseline validations.
*   **`src/mechanistic_disentangle/`**: The mechanistic representation engineering utilities. **All SVD math, contrastive geometric analysis, and Gaussian depth scheduling live exclusively here.**
*   **`src/shared/`**: Unified cross-pipeline infrastructure (caching, deterministic seeding, and metric implementations).

*Rule of thumb:* `latent_coordination` must never natively compute SVDs or projection matrices; it strictly consumes the vectors (like the 9-dimensional $Geo_L$ risk vector) precomputed by `mechanistic_disentangle`.

---

## 2. Core Operational Modules

The coordination system evaluates heterogeneous language agents via four primary modules:

### Module B: Interlingua-Regularized Latent Hub (`universal_space.py`)
*   **Objective:** Map heterogeneous model hidden states into a language-agnostic shared space $\mathbb{R}^{512}$.
*   **Mechanism:** Implements a multivariant loss $\mathcal{L}_{hub}$ merging standard Autoencoder (AE) loss, Denoising Autoencoder (DAE) loss, and Cross-Lingual Alignment (CKA). 
*   **Adapter Scaling Protocol:** Enables $O(1)$ scaling. When a new language is introduced, `fit_isolated_adapter()` freezes the central hub and only trains the specialized $E_i/D_i$ boundary layers for the incoming agent.

### Module C: Recursive Latent Space Reasoning (`recursive_core.py`)
*   **Objective:** Refine latent representations iteratively without degrading or collapsing into text.
*   **Mechanism:** Runs a $T$-step two-layer bottleneck residual network (`z_t = z_{t-1} + W_2(GeLU(W_1(z_{t-1})))`).
*   **Control Flow:** Utilizes a sigmoid early-exit classifier. When network confidence exceeds $\tau_{exit}$, the iterative loop halts and forwards the vector.

### Module D: Geometry-Conditioned CVAE Graph Prior (`cvae_prior.py`)
*   **Objective:** Generate collaboration topologies adapted to the complexity and volatility of the target language.
*   **Mechanism:** The topology is parameterized by concatenating the query $q$ with the precomputed mechanistic risk vector $Geo_L$ to form $x = [q \| Geo_L]$, driving the Variational Autoencoder formulation.

### Module E: Closed-Loop Test-Time Reconstruction Probe (`verification_probe.py`)
*   **Objective:** Detect semantic drift inside the continuous hub in real-time.
*   **Mechanism:** Decodes the final latent state $z_T$ and calculates a cosine-similarity drift score. If $\mathcal{D}_{drift} > \tau_{drift}$, the system throws a `LatentDriftException` to initiate a graph repair pass.

---

## 3. The Evaluation Matrix

The evaluation suite allows for a composable (Model $\times$ Baseline $\times$ Benchmark $\times$
Language $\times$ Metric) testing harness -- but it is **not a free cartesian product**:
compatibility constraints (a baseline's CLI only wires up specific benchmarks, a benchmark only
has data for specific languages, a metric only applies to specific task types) rule out most
combinations. `src/shared/combinations.py` is the single source of truth for these constraints,
every fact in it verified against real HF dataset/model metadata rather than assumed from docs.
Enumerate or validate combinations with `python scripts/list_combinations.py` (see
`--help`; supports `--check MODEL BASELINE BENCHMARK LANGUAGE METRIC` for a single lookup).

### Available Backbones & Execution Models
*   `Qwen/Qwen2.5-7B-Instruct` (Default generalized multilingual solver)
*   `aisingapore/Llama-SEA-LION-v3-8B-IT` (Default agent model in configs/*.yaml)
*   `aisingapore/sea-lion-7b-instruct`, `SeaLLMs/SeaLLMs-v3-7B-Chat` (SEA-specific solvers)
*   `llava-hf/llava-1.5-7b-hf` (Multimodal latent alignment solver)
*   **Integrated 2026-07-02** (downloaded, wired into `configs/latent_coordination_heterogeneous.yaml`, promoted to `"in_use"` in `combinations.py`):
    *   `sail/Sailor2-8B-Chat` -- broadest single-model SEA coverage (12 languages incl. lo/my/km/jv/su/fil), apache-2.0, same footprint as current models. Translation agent in the heterogeneous config.
    *   `CohereLabs/aya-expanse-8b` -- the only genuinely different architecture family (`cohere`, vs qwen2/llama/gemma everywhere else); makes the heterogeneous cross-architecture ablation (safety agent on one arch, reasoning agent on another) actually true rather than same-arch-different-checkpoint. cc-by-nc-4.0 (research use), gated (auto-approved). Safety agent in the heterogeneous config.
    *   `meta-llama/Llama-3.1-8B-Instruct` -- standard llama-arch comparison point. Gated, access approved 2026-07-02. Reasoning agent in the heterogeneous config.
*   **Excluded:** `deepset/xlm-roberta-large-squad2` -- encoder-only extractive-QA model, incompatible with every baseline/pipeline here (all require `AutoModelForCausalLM`). Previously listed here as an "available backbone" in error.

### Benchmarks & Evaluation Suites (verified language coverage, not aspirational)
*   **Math & Logic:** MGSM (bn/de/en/es/fr/ja/ru/sw/te/th/zh -- **no** lo/km/my/am upstream), MGSM-Pro (am/en/fr/ig/ja/sw/tw/yo/zh -- notably has Amharic, which base MGSM lacks; languages are HF *splits*, not a config param), GSM8K/AIME (English-only).
*   **Regional & Cultural Capabilities:** Belebele (13 SEA+ languages incl. th/my/km/lo/am/sw), **LaoBench** (`BAAI/LaoBench`, added to close the Lao math/reasoning gap -- MGSM/MRG have none; apache-2.0, ungated, MCQA subset only), SEA-HELM (gated; real component repos are `aisingapore/<ComponentName>`, e.g. `aisingapore/NLU-Belebele-MCQA` -- the old `aisingapore/sea-helm-{subset}` ID 404s and was fixed; access granted + `load_sea_helm()` schema fixed 2026-07-02, see Section 5), SeaBench, MultiChallenge.
*   **Verification:** Multilingual Reasoning Gym / MRG (de/en/es/fr/hi/it/ja/ko/pl/pt/ru/uk/zh -- **no SEA languages at all**), GPQA-Diamond (English-only).
*   **Cross-Lingual QA & Translation:** XQuAD (12 languages incl. th), MLQA (7 languages, no SEA), FLORES+ (~200 languages, primary corpus for `coordination_pipeline.py`).
*   **Robustness appendix only:** MathMist (`mahbubhimel/MathMist`, gated, access granted 2026-07-02; covers am/sw of our target set, not th/lo/km/my despite the paper's 13-language claim; `load_mathmist()` schema fixed to match the real per-language-split layout), BanglaMATH (Bengali only).

### Tracked Languages
*   **Anchor:** English (`en`).
*   **High-Risk Target Scripts:** Thai (`th`), Lao (`lo`), Khmer (`km`), Burmese (`my`), Amharic (`am`), Swahili (`sw`), Bengali (`bn`), Telugu (`te`).
*   Coverage is per-benchmark, not universal -- see `src/shared/combinations.py::BENCHMARKS[...].languages` for the authoritative per-benchmark set before assuming a language is testable somewhere.

### Available Baselines & Topologies
*   **Runnable today:** `LatentMASBaseline`, `ThoughtCommBaseline` (CLI: `run_latentmas.py`/`run_thoughtcomm.py`, benchmarks: mgsm, mgsm_pro, belebele), plus the coordination pipeline's built-in `single_agent_baseline` / `token_based_mas` / `latent_based_mas_ours` modes (benchmarks: flores_plus, sea_vision, sea_safeguardbench).
*   **Implemented but not benchmark-wired yet:** `CacheToCacheBaseline`, `GDesignerBaseline`, `MasRouterBaseline`, `VisionWormholeBaseline`, `BlackboardMASBaseline` -- classes exist, no `run_*.py` CLI integration.
*   **Recommended additions from the 2024-2026 literature** (not implemented; verify code availability before committing engineering time): KVComm (online cross-context KV-cache communication, arXiv:2510.12872), DyTopo (dynamic semantic-similarity topology routing, arXiv:2602.06039), Optimal-Agent-Selection (arXiv:2511.02200). MAPS (arXiv:2505.15935) is the closest existing multilingual-MAS benchmark paper -- position this project's novelty claim against it in related work.

---
## 4. Rigorous Metrics & Zero-Tolerance Mocks

To guarantee mathematical integrity, the repository strictly enforces a **fail-fast, zero-fallback policy**. 

### Hardened Dependencies
If a required evaluation package (e.g., `sacrebleu` for chrF, `unbabel-comet` for COMET, `transformers` for pipelines) is missing, the code **will unconditionally crash with an `ImportError`**. It will never silently return $0.0$.

### No Dummy Ablations
Ablation arrays (like those in `multi_agent_runner.py`) are strictly generated from dynamic, runtime computations via `get_ablation_metrics()`. Hardcoded mocks have been eradicated; if a system lacks the logic to compute real ablations, it will throw a `NotImplementedError`.

### CKA and Geometry Alignment Constraints
When running the `UniversalLatentHub`, if the English anchor state (`anchor_hidden_states`) is missing from the tensor batch, the pipeline throws a `ValueError` rather than defaulting to a $0.0$ CKA loss.

### Metric Definitions
*   **CLAP (Cross-Lingual Alignment Probe):** Computes the SVD projection gap ($\delta$) using the top singular concept direction ($u_1$).
*   **SFR (Script Fidelity Rate):** Validates script integrity against exact Unicode block boundaries for target languages. Blind spot: same-script language pairs (e.g. Swahili vs English, both Latin-script) always score a high SFR regardless of actual language.
*   **LC (Language Consistency):** `eval.script_fidelity::LanguageConsistencyEvaluator`, added to close SFR's blind spot -- whole-response language ID (`langid`) rather than a per-character script check, so it can actually distinguish Swahili/Indonesian/Malay/Cebuano/Filipino generations from English drift. Unsupported for Burmese (`langid` has no `my` class); those samples report `is_consistent=None`, not a silently-wrong `False`.
*   **IFL (Involuntary Fidelity Loss):** The direct "English-drift" metric, calculated as $IFL = 1.0 - SFR$.
*   **COMET, chrF, Exact Match, CKA, Drift (Activation divergence):** Natively implemented via their respective strict algorithms.
*   **xCOMET / CometKiwi (wired 2026-07-02, DISABLED BY DEFAULT as of the same day -- see caution below):** `shared.metrics::compute_xcomet` / `compute_cometkiwi` -- as of 2025/2026, plain COMET (`Unbabel/wmt22-comet-da`) is no longer the frontier. xCOMET (`Unbabel/XCOMET-XL`, reference-based, best correlation with human judgment + fine-grained error spans) and CometKiwi (`Unbabel/wmt23-cometkiwi-da-xl`, reference-free QE) are called from `MultiAgentBenchmarkRunner._compute_translation_quality` against the real FLORES+ gold reference/source carried on each `AgentTask` (`task.context`/`task.query`). Opt-in per benchmark via `configs/*.yaml`'s `benchmarks.flores_plus.translation_metrics`.
    **Caution (found live, 2026-07-02):** `unbabel-comet` was listed in `pyproject.toml` as a dependency but was never actually installed in this environment. `pip install unbabel-comet` resolves to a dependency set that force-upgrades `transformers`/`accelerate` (observed 4.46.3->4.57.6, 1.1.1->1.13.0), breaking the versions this pipeline's agent generation is pinned to (exactly the risk `pyproject.toml`'s own comment already warned about for a different install path). Even after installing it and reverting the transformers/accelerate upgrade, a real run then crashed with `ValueError: Backend should be defined in the BACKENDS_MAPPING. Offending backend: tensorflow_text` inside COMET's own tokenizer init. Both `configs/latent_coordination_heterogeneous_timeboxed.yaml` and any run using xcomet/cometkiwi crashed on this before a single mode's results were saved. **Do not enable `xcomet`/`cometkiwi` in the same process as agent generation until this is resolved in an isolated env/subprocess.** chrF has no such risk (sacrebleu only) and stays on by default.

---

## 5. Comprehensive Test Combination Matrix & Time Estimates

All items in Section 3's "Recommended additions" have been integrated (2026-07-02):
Sailor2-8B-Chat, aya-expanse-8b, and Llama-3.1-8B-Instruct are downloaded and wired into
`configs/latent_coordination_heterogeneous.yaml` (a genuinely cross-architecture agent pool
-- llama / qwen2 / cohere -- as translation/safety/reasoning agents alongside the SEA-LION
orchestrator); xCOMET and CometKiwi are downloaded and wired per above; `laobench`,
`sea_helm`, `mgsm_pro`, and `mathmist` loaders in `data.py` were run against the real
(now-cached) HF datasets and had schema bugs fixed (wrong split names, wrong column names,
missing per-language dataset configs -- see `src/shared/combinations.py` notes on each for
specifics). `python -m pytest tests/` passes (196/196) after fixing a stale test that
asserted the now-wired models were still `"recommended"`-status.

`enumerate_valid_combinations()` (no filters) currently returns **3,608** valid
`(model, baseline, benchmark, language, metric)` tuples. Listing all 3,608 individually
would not be useful -- metrics are computed for free from the same generations within one
run (except xcomet/cometkiwi, which load their own checkpoint once per run), so the actual
unit of work is one **(model, baseline, benchmark, language)** run: **936** of those.
Grouping further by `(baseline, benchmark)` -- the grain that determines which CLI/pipeline
entry point runs and what a single invocation costs -- gives the following comprehensive,
readable breakdown.

### Methodology (analytical, not measured -- see dev_doc.md's own instruction for this
### section; re-derive empirically with `scripts/run_mechanistic_pipeline.py`-style timing
### if measured numbers are needed)

*   **Hardware:** 1x Tesla V100-PCIE-16GB (this box has 8; naive per-GPU parallelization divides total time by up to 8).
*   **Decode throughput:** 10 tokens/s per agent call. All 6 in-use backbones are 7-8.5B params, loaded 8-bit (`load_in_8bit: true` in every config) -- Volta (cc7.0) lacks int8 tensor cores, so bitsandbytes 8-bit inference is markedly slower than on Turing+; this is a deliberately conservative single-stream (batch=1) figure, not a measured benchmark.
*   **MEASURED CALIBRATION (2026-07-03) -- supersedes the 10 tok/s assumption; scale
    every estimate in this section by ~1.5x.** Two independent measured sources agree
    on **~4.7 effective tok/s** (V100, 8-bit, batch=1, incl. prefill):
    (1) the 6 real LatentMAS/ThoughtComm MGSM runs (Qwen2.5-7B-Instruct, n=200,
    `max_new_tokens=256`, copied into `results/baselines/{latentmas,thoughtcomm}/`
    from LRL-MRRE-MAS): 67.8 (en) / 88.6 (sw) / 96.8 (th) s per 2-call sample, i.e.
    ~34-48 s/agent-call at 154-213 actually-generated tokens/call -- language-dependent
    via tokenizer inflation (th ~1.4x en);
    (2) the heterogeneous FLORES+ timeboxed runs (`results/coordination_heterogeneous_
    timeboxed*/`): 18-21 s/call at a 96-token cap ~= 1.5 + 96/4.7, so the mixed
    sailor2/llama/cohere pool decodes at the same rate as the Qwen homogeneous case.
    Calibrated per-call planning figure for MGSM-length outputs: **~40 s average
    across MGSM's 11 languages (34 s en-like Latin/CJK, 48 s th/bn/te-like)**.
*   **Prefill + orchestration overhead:** 1.5s per agent call (tokenization, prompt prefill at our context lengths, router/orchestrator bookkeeping).
*   **Model load time:** 90s per model per run, amortized once (not per-sample).
*   **Agent calls per sample**, by baseline/comm-mode: `LatentMASBaseline`/`ThoughtCommBaseline` = 2 (documented two-step homogeneous chain, see `run_latentmas.py` docstring); `single_agent_baseline` = 1; `token_based_mas`/`latent_based_mas_ours` = 3 (translation + reasoning + safety agents; the orchestrator only routes).
*   **Output tokens per sample**, by `TaskType`: MATH_REASONING/CODE/OPEN_GENERATION = 256 (chain-of-thought or free-form), READING_COMPREHENSION/KNOWLEDGE_MCQA = 64, COMMONSENSE = 32, TRANSLATION = 64, SAFETY = 32 -- except `mathmist`, overridden to 512 (verified: its "Proof"-type solutions are long-form, not short answers).
*   **Samples per run:** `min(200, benchmark_size)`, matching this repo's own `run_latentmas.py --n` default of 200. Real verified dataset sizes (2026-07-02): mgsm=250/lang, mgsm_pro=2250/split, belebele=900/lang, laobench (MCQA subset)=5000, sea_helm (NLU-Belebele-MCQA)=895/lang, mathmist=1445/lang, flores_plus devtest=1012/lang, gpqa_diamond=198, humanevalplus=164, mbppplus=399. Everything else defaults to 200.
*   **flores_plus language count:** `combinations.py` lists `languages=None` (any of FLORES+'s ~200), but `MultiAgentBenchmarkRunner._load_real_tasks` (the only real loader path exercised by the 3 comm-mode baselines) hardcodes exactly 6 pairs (th, my, km, lo, am, sw) -- that's the number used below, not "any language."
*   **Formula:** `total_s = 90 + n_units x n_samples x agent_calls x (1.5 + output_tokens / 10)`, where `n_units = n_models x n_languages` for that (baseline, benchmark) pair.

### Per-(baseline, benchmark) breakdown

| Baseline | Benchmark | Models | Languages | Units (model x lang) | Samples/run | Est. time / unit | Est. total (all units, 1 GPU) | Est. total (8 GPUs) |
|---|---|---|---|---|---|---|---|---|
| `LatentMASBaseline` | mgsm | 8 | 11 | 88 | 200 | 182.2 min (3.0h) | 267.2h (11.1d) | 33.4h (1.4d) |
| `LatentMASBaseline` | mgsm_pro | 8 | 9 | 72 | 200 | 182.2 min (3.0h) | 218.6h (9.1d) | 27.3h (1.1d) |
| `LatentMASBaseline` | belebele | 8 | 13 | 104 | 200 | 54.2 min (0.9h) | 93.9h (3.9d) | 11.7h (0.5d) |
| `ThoughtCommBaseline` | mgsm | 8 | 11 | 88 | 200 | 182.2 min (3.0h) | 267.2h (11.1d) | 33.4h (1.4d) |
| `ThoughtCommBaseline` | mgsm_pro | 8 | 9 | 72 | 200 | 182.2 min (3.0h) | 218.6h (9.1d) | 27.3h (1.1d) |
| `ThoughtCommBaseline` | belebele | 8 | 13 | 104 | 200 | 54.2 min (0.9h) | 93.9h (3.9d) | 11.7h (0.5d) |
| `single_agent_baseline` | flores_plus | 8 | 6 (hardcoded) | 48 | 200 | 27.8 min (0.5h) | 22.3h (0.9d) | 2.8h (0.1d) |
| `single_agent_baseline` | sea_vision | 8 | 10 | 80 | 200 | 91.8 min (1.5h) | 122.4h (5.1d) | 15.3h (0.6d) |
| `single_agent_baseline` | sea_safeguardbench | 8 | 1 (repo-dependent) | 8 | 200 | 17.2 min (0.3h) | 2.3h | 0.3h |
| `token_based_mas` | flores_plus | 8 | 6 (hardcoded) | 48 | 200 | 80.5 min (1.3h) | 64.4h (2.7d) | 8.1h (0.3d) |
| `token_based_mas` | sea_vision | 8 | 10 | 80 | 200 | 272.5 min (4.5h) | 363.3h (15.1d) | 45.4h (1.9d) |
| `token_based_mas` | sea_safeguardbench | 8 | 1 (repo-dependent) | 8 | 200 | 48.5 min (0.8h) | 6.5h | 0.8h |
| `latent_based_mas_ours`\* | flores_plus | 8 | 6 (hardcoded) | 48 | 200 | 80.5 min (1.3h) | 64.4h (2.7d) | 8.1h (0.3d) |
| `latent_based_mas_ours`\* | sea_vision | 8 | 10 | 80 | 200 | 272.5 min (4.5h) | 363.3h (15.1d) | 45.4h (1.9d) |
| `latent_based_mas_ours`\* | sea_safeguardbench | 8 | 1 (repo-dependent) | 8 | 200 | 48.5 min (0.8h) | 6.5h | 0.8h |

\* `latent_based_mas_ours` requires the HF backend. **Correction (2026-07-02):** an earlier
version of this note claimed this mode fails fast on `configs/latent_coordination_
heterogeneous.yaml`'s mixed llama/qwen2/cohere pool, by analogy with the standalone
`LatentMASBaseline.share_hidden_state` (used by `run_latentmas.py`/`run_thoughtcomm.py`),
which does hard-require matching hidden dims and raises `ValueError` across architectures.
That analogy was wrong: the coordination pipeline's `latent_based_mas_ours` mode goes
through `AdaptiveOrchestrator.execute` -> `UniversalLatentHub.transfer`, which registers
each sender/receiver agent by its *actual* `hidden_dim` and routes through a per-agent
adapter pair into the shared universal space (`src/latent_coordination/latent_space/
universal_space.py::register_agent`/`encode`/`decode`) -- dimension- and architecture-
agnostic by design (that's the whole point of Module B, Section 2). It runs on
heterogeneous pools same as homogeneous ones; there is no special-cased failure mode here.

**Totals:** 15 `(baseline, benchmark)` groups, 936 `(model, baseline, benchmark, language)`
execution units, 3,608 valid `(..., metric)` tuples overall.

*   **Grand total, sequential on 1x V100:** ~2,175 hours (~90.6 days).
*   **Naive parallel across this box's 8x V100s** (no shared-resource contention accounted for): ~271.8 hours (~11.3 days).
*   **Added one-time overhead** (not in the table, paid once regardless of how many combinations run): xCOMET/CometKiwi checkpoint load, ~2-3 min each the first time each is invoked in a process.

**Reality check:** running the full comprehensive matrix is not a practical goal for a
single research pass -- treat the table above as a costing tool for scoping a specific
experiment (e.g. "just the heterogeneous-config ablation on flores_plus" = 1 model-pool x
6 languages x 3 comm-modes context ≈ single-digit hours), not a to-do list to exhaustively
execute. Use `python scripts/list_combinations.py --baseline X --benchmark Y` to enumerate
the exact model/language/metric subset for whatever slice you actually intend to run, and
re-derive the time estimate for that slice from the formula above.

---

## 9. 2026-07-03 Evaluation configuration & pipeline audit (post-port from LRL-MRRE-MAS)

A full reconciliation of the ported eval config/pipeline against the LRL-MRRE-MAS
strategy documents (`strategy.md`, `implementation strategy.pdf`). Regression tests:
`tests/shared_infra/test_eval_pipeline_fixes.py`. Fixed this session:

**Measurement correctness**
- Accuracy denominator now `len(tasks)`, not `len(answers)` (unanswered tasks were
  silently dropped → inflated accuracy in all three comm-modes).
- Token cost now counted with the producing agent's real tokenizer; whitespace
  `split()` counted an entire unsegmented Thai/Burmese/Khmer sentence as ~1 token.
- `shared/metrics._detect_script_ratio` (SFR/IFL) is now character-level; the
  token-level version was near-binary 0/1 for unsegmented scripts.
- `eval/metrics.compute_perplexity` masks PAD labels (-100); previously scored the
  model on predicting padding. `compute_bleu`'s "bleu_4" now really is 4-gram
  precision (overall score moved to "bleu").
- Latent mode's 0-token claim is now true: `AdaptiveOrchestrator.execute()` no longer
  passes each agent's decoded text to the next agent as `context` (hidden token
  side-channel), and the fabricated `words*2.0` token estimate is gone.
- `EfficiencyAnalyzer.run_ablation` no longer fabricates a fresh untrained
  128-dim hub + constant 256.0 latent cost; it requires the pipeline's real hub and
  reports measured latent bytes.
- Benchmark report metadata no longer claims a vLLM backend that never generated a
  token; explicit `--backend vllm` now fails loudly (vLLM is not wired into the
  agent path).

**Pipeline logic**
- Stage B honors `cvae.training.{n_epochs,lr,batch_size}` (were hardcoded 20/1e-3/8)
  and trains on per-query TaskDecomposer-derived topology targets instead of a
  constant all-ones matrix (which made the CVAE prior ignore its conditioning).
- Stage C actually trains adapters now: `UniversalLatentHub.fit_adapters` implements
  Module A+B (`L_recon + γ·L_DAE + μ·L_CKA`, unbiased-HSIC CKA per the audit), gated
  by `universal_latent_space.adapter_training.enabled` (default false → loud warning
  that latent-mode numbers are not reportable on random adapters).
- Stage C/D checkpoints carry real state (hub object / centroids) instead of `True`
  sentinels, so `--stages E` runs no longer lose registrations/centroids.
- Stage D fits centroids on float `encode_query_bow` embeddings (was: Stage B's long
  CVAE token-ids when B ran first) at `orchestration.n_intent_centroids` (was 3).
- Stage E cache key includes every agent's model_id (was: orchestrator only — a
  heterogeneous config swap silently reused stale cached results); checkpoint key is
  `stage_e` (legacy `stage_f` still readable).
- Stage F CVAE plot: `encode(G, Q)` argument order fixed (was swapped + flattened,
  so the plot silently failed every run).
- Router: canonical deterministic role order translation→reasoning→safety (was
  PYTHONHASHSEED-random set iteration); route-time query embeddings share
  `QUERY_EMBED_DIM=64` with centroid fitting (was 32 vs 64 → k-means path crashed);
  `compare_communication_modes` registers receivers at their own hidden_dim.
- `BaseAgent.inject_latent_and_generate` injects at prefill only (the hook used to
  re-overwrite every decode step's hidden state with the same vector); greedy
  decoding everywhere in eval for reproducibility.
- TranslationAgent's SFR gate uses the target language's actual Unicode ranges (raw
  non-ASCII density scored every correct Swahili output as "low quality").
- `verification_probe` (Module E) refuses to gate on an untrained decoder and has a
  real `fit_decoder`; per-sample drift reporting.
- Configs: heterogeneous config's xcomet/cometkiwi disabled (documented crash);
  `--agents` CLI overrides type-coerce values (`load_in_8bit=false` was a truthy
  string) and reject unknown agent ids.

**Known gaps at the time of the audit — ALL SIX SINCE CLOSED (status as of
2026-07-03, later sessions; kept for the historical record):**

1. ~~Stage E completeness proxy~~ **Fixed:** MGSM (EM), Belebele (log-likelihood or
   EM via `benchmarks.belebele.scoring`), SEA-Vision and SEA-SafeguardBench are
   Stage-E workloads with real correctness (`benchmark_runner.py::_load_real_tasks`
   and `_compute_correctness`; per-benchmark `accuracy_<bench>` metrics,
   `accuracy_kind='correctness'` when gold-carrying tasks exist).
2. ~~No Geo_L conditioning; route() ignores topology~~ **Fixed:** `CVAETopologyPrior`
   conditions on `x = [q ‖ Geo_L]` when `cvae.condition_on_geometry=true`
   (`geo_profile_path` artifact from `scripts/export_geo_profiles.py`, loader in
   `topology/geo_profile.py`); `routing_strategy: cvae_topology` makes the trained
   prior drive agent selection/order at route time (zero-fallback: missing Geo_L
   raises, never zero-substitutes).
3. ~~Modules C/E dead code~~ **Fixed:** `latent_reasoning.enabled` /
   `verification.enabled` put `RecursiveLatentCore` and the drift probe in the hub
   transfer path (`router._hub_transfer`: refine → drift-check → one repair hop →
   flag-and-continue). Probe refuses to gate untrained; fitted on the real Stage-C
   corpus.
4. ~~Firewall~~ **Fixed:** SVD/steering/geometry/mechanistic pipeline moved to
   `src/mechanistic_disentangle/`; `scripts/firewall_check.sh` (AST-based, Rules
   1-4 of strategy.md §6) passes and appends to `ARTIFACTS/firewall_audit_log.md`.
5. ~~Output-text re-encode as "latent state"~~ **Fixed:**
   `BaseAgent.generate_and_capture` hands off generation-time hidden states from
   `communication.latent_transfer_layer` (regression:
   `tests/shared_infra/test_generation_time_latent_capture.py`).
6. ~~Ignored config knobs~~ **Fixed** (last three closed 2026-07-03, see §10):
   `latent_transfer_layer`, `routing_strategy`, `sea_vision`/`sea_safeguardbench`
   loaders in earlier sessions; `checkpointing.checkpoint_dir`,
   `timeout_per_agent_s`, and `ablation.*` in §10. `parallel_agents` was removed
   rather than wired — see §10 for why.

---

## 10. 2026-07-03 Strategy-gap closure (gap-6 remainder + strategy.md Phase-4 items)

Regression tests: `tests/shared_infra/test_strategy_gap_fixes.py` (16 tests;
full suite 257/257, firewall PASS). Fixed this session:

* **`checkpointing.checkpoint_dir` honored** (was hardcoded to
  `{output_dir}/checkpoints`). Note: runs resuming from the old default location
  should either set the knob to that path or move the checkpoint tree once.
* **`orchestration.parallel_agents` removed from configs; `true` now fails
  loudly.** It was never implemented and cannot be: the latent chain is
  sequential by design — each agent consumes the previous agent's transferred
  hidden state, so there is nothing to parallelize within one task. Parallelize
  across languages/instances instead (the 2-instance 8-GPU split in
  `scripts/build_experimental_report.py`). `max_parallel_workers` removed with it.
* **`orchestration.timeout_per_agent_s` wired** to transformers'
  `generate(max_time=…)` stopping criterion via `AgentConfig.max_time_s` — a
  runaway decode now ends cleanly at the budget instead of hanging the chain.
* **Staircase ablation runner (strategy.md §7.3):**
  `latent_coordination/eval/ablation_staircase.py` + `scripts/run_ablation_staircase.py`.
  Rows 0-6 plus the 7a loss-term split map to the REAL module toggles
  (`adapter_training.enabled`, `routing_strategy=cvae_topology` with
  `condition_on_geometry`, `latent_reasoning.enabled`, `verification.enabled`,
  `mu_cka`/`gamma_dae` zeroing); 7b (OneFlow single-agent row) rides in every
  row's `eval_modes`; 7d/7e are expressible as `ablation.extra_rows` overrides.
  Each row runs in an isolated output+checkpoint dir — REQUIRED, because the
  Stage-E cache key does not encode module toggles. The old `ablation:` YAML
  block (`communication_modes`/`n_agents_sweep`/`topology_types`) was config no
  code ever read, promising sweeps the 3-role pipeline cannot execute — replaced.
  **Always `--dry-run` first**: a full staircase is ~10 full pipeline runs.
* **OneFlow narrative-gating conditional (strategy.md §7.2)** implemented in the
  report generator, not left editorial: Stage G's `final_report["headline_framing"]`
  (`coordination_pipeline.py::derive_headline_framing`) pivots to
  `efficiency_fallback` (token-overhead reduction vs `token_based_mas`,
  bandwidth savings, heterogeneous cross-architecture caveat) whenever
  `single_agent_baseline` accuracy ≥ `latent_based_mas_ours` accuracy. The
  Phase-4-gate synthetic-trigger test is in the regression file.
* **Drift-probe shallow-MLP variant (strategy.md §4.4 / ablation 7e):**
  `verification.probe_arch: linear|mlp` (+ `mlp_hidden_dim`);
  `QueryReconstructionProbe.query_dim` is now the canonical dim attribute
  (the MLP's `Sequential` decoder has no `out_features`; router falls back for
  pre-MLP checkpoints).

**Still open (research execution, not wiring):** actually *running* the staircase
with adapter training + trained CVAE + exported Geo_L artifacts at reportable
sample sizes (see Section 5's cost model), and the Phase-3/Paper-2 gate items
(strategy.md Phase 0 differentiation write-up) which are writing tasks, not code.

---

## 11. Current Status, Future Plans, and Ongoing Issues

### Project Status: Latent Coordination (Paper 3)
This repository represents the **Latent Coordination** project (targeted for AAAI 2027), which focuses on text-free multi-agent reasoning via a shared continuous latent space. The primary goal is to establish the defensible novelty of the query-conditioned CVAE graph prior with zero-shot topology transfer.

### Ongoing Issues and Critical Blockers
- **Correctness Scorer (CRITICAL):** The current completeness proxy is disqualifying for a reasoning-coordination claim and must be replaced with a real reference scorer. Tasks like MGSM (exact-match) and MMLU-ProX (multiple-choice log-likelihood) must be integrated to produce real accuracy metrics.
- **Baseline Integration:** The pipeline lacks head-to-head MAS baselines. It is critical to integrate `LatentMAS` (training-free latent SOTA) and either `ThoughtComm` or `G-Designer` to report an accuracy-vs-token-cost frontier.
- **Cost Accounting:** Need to implement honest cost accounting (prompt/total tokens + wall-clock) versus accuracy at different agent scales (N=4, 8, 16).
- **Router Ablation:** The attention router needs to be ablated against k-means, and the Transformer query encoder against BiLSTM, reporting correctness deltas instead of completeness.

### Future Plans and Architecture
- **Repo Firewall & Reframing:** Ensure all CLAP/SVD/U_R content is removed from this repository to maintain the firewall between the MAS-domain work and the geometry/steering work. The narrative will be reframed to clearly distinguish the CVAE prior from existing works like Vision Wormhole and LatentMAS.
- **Hardware Constraints:** All operations must adhere to V100 limits (fp16 only, no bf16). One agent per device will be mapped for heterogeneous backbones (e.g., Qwen2.5-7B, Llama-3.1-8B), coupled through the 512-dimensional universal hub.
