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

The evaluation suite allows for a fully composable (Model $\times$ Benchmark $\times$ Language $\times$ Baseline) cross-section testing harness.

### Available Backbones & Execution Models
*   `Qwen/Qwen2.5-7B-Instruct` (Default generalized multilingual solver)
*   `aisingapore/sea-lion-7b-instruct` (SEA-specific solver)
*   `deepset/xlm-roberta-large-squad2` (Zero-shot extractive cross-lingual solver)
*   `llava-hf/llava-1.5-7b-hf` (Multimodal latent alignment solver)

### Benchmarks & Evaluation Suites
*   **Math & Logic:** MGSM, MGSM-Pro, MathMist, BanglaMATH, AIME, GSM8K.
*   **Regional & Cultural Capabilities:** Belebele, SEA-HELM, SeaBench, MultiChallenge.
*   **Verification:** Multilingual Reasoning Gym (MRG), GPQA-Diamond.
*   **Cross-Lingual QA & Translation:** XQuAD, MLQA, FLORES+.

### Tracked Languages
*   **Anchor:** English (`en`).
*   **High-Risk Target Scripts:** Thai (`th`), Lao (`lo`), Khmer (`km`), Burmese (`my`), Amharic (`am`), Bengali (`bn`), Telugu (`te`). 

### Available Baselines & Topologies
*   **Multi-Agent / Latent Frameworks:** LatentMAS, ThoughtComm, Cache-to-Cache, G-Designer, MASRouter.
*   **Adversarial / Regional Checks:** `SingleAgentOneFlow`, `SeaHelmBaseline`, `SeaLionBaseline`, `XQuadBaseline`, `VisionWormholeBaseline`, etc.

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
*   **SFR (Script Fidelity Rate):** Validates script integrity against exact Unicode block boundaries for target languages.
*   **IFL (Involuntary Fidelity Loss):** The direct "English-drift" metric, calculated as $IFL = 1.0 - SFR$.
*   **COMET, chrF, Exact Match, CKA, Drift (Activation divergence):** Natively implemented via their respective strict algorithms.
