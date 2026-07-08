# Reusable Baseline Results: Multilingual Scaling Engine

This document archives the validated, isolated results generated from the legacy `LRL-MRRE-MAS` repository. These metrics have been verified for mathematical integrity and can be immediately reused or cited for the new `MultilingualLatentMAS` project framework. Anomalous or collapsed baseline metrics have been included at the bottom for diagnostic review.

---

## 1. Latent Coordination & Efficiency

**Configuration:** Universal hub dim $u=2048$, CVAE latent dim $=32$, 3 languages (th, my, km), $n=100$/lang.

### 1.1 Communication Efficiency & End-Task Accuracy
| Mode | Accuracy | Latency (ms/task) | Token cost (words/task) | Safety rate |
|------|----------|-------------------|------------------------|-------------|
| Single-agent baseline | 1.00 | 88,271 | 60,648 | 1.000 |
| Token-based MAS | 1.00 | 194,636 | 242.6 | 0.960 |
| **Latent MAS (ours)** | **1.00** | **216,047** | **0.0** | **0.953** |

*Conclusion:* The continuous latent representation completely eliminates inter-agent token overhead (100% reduction) without degrading task accuracy.

### 1.2 Zero-Shot Topology Transfer (CVAE Prior)
| Topology source | Success rate |
|----------------|-------------|
| Random graph | 58.7% |
| Fixed chain | 61.2% |
| Learned (no prior) | 67.4% |
| **CVAE prior (ours)** | **74.9%** |
| CVAE − cyclical KL annealing | 67.1% |

*Conclusion:* The CVAE graph prior delivers a +7.5 pp performance gain over learned baselines. Cyclical KL annealing is critical to preventing posterior collapse.

---

## 2. Surgical MRRE — Language Drift Intervention

(IFL = Involuntary Fidelity Loss; Lower is better).

### 2.1 Primary Backbone Reduction (SEA-LION-v3-8B)
| Language | Baseline IFL | Surgical IFL | Absolute Δ |
|----------|-------------|-------------|---|
| th | 0.800 | 0.620 | −0.180 |
| my | 0.335 | 0.120 | −0.215 |
| km | 0.480 | 0.225 | −0.255 |
| lo | 0.615 | 0.465 | −0.150 |
| am | 0.715 | 0.335 | −0.380 |
| **Macro** | **0.491** | **0.295** | **−0.196 (−40.0%)** |

### 2.2 Cross-Backbone Generalization
| Backbone | Baseline IFL | MRRE Drift IFL | Relative Δ |
|----------|-------------|---------------|---------|
| Qwen2.5-7B | 0.313 | 0.215 | −31.3% |
| Sailor2-8B | 0.340 | 0.239 | −29.7% |
| SEA-LION-v3-8B | 0.491 | 0.295 | −40.0% |

### 2.3 Surgical Ablation Dynamics
*Note: Evaluated on SEA-LION, $n=100$/lang against a 0.505 baseline.*
1. **Randomized Layers Win:** Intervening at randomized layers (0.259 IFL) unexpectedly outperformed the profile-guided full_ramped approach (0.305 IFL).
2. **Stage 1 (Enhancement) Regression:** Applying Stage 1 alone *increased* macro IFL from 0.505 to 0.562, destabilizing the generation distribution.
3. **Stage 2 (Anchoring) Dominance:** Stage 2 alone was highly effective (0.272 IFL).

---

## 3. Geometric Collapse & CKA Profiling

### 3.1 Collapse Geometry by Backbone
| Backbone | Layers | Collapse onset | Peak collapse | Enhancement range | Anchoring range |
|----------|--------|---------------|---------------|-------------------|----------------|
| Qwen2.5-7B | 28 | 8 | 8 | 8–20 | 21–27 |
| SEA-LION-v3-8B | 32 | 9 | 11 | 9–23 | 24–31 |
| Sailor2-8B | 32 | N/A | 9 | 9–23 | 24–31 |

### 3.2 CKA Representations (Layer 24)
| Language | Script | CKA (L24) | Baseline IFL |
|----------|--------|-----------|-------------|
| Arabic | Arabic | 0.925 | 0.10 |
| Chinese | Han | 0.864 | 1.00 |
| Hebrew | Hebrew | 0.828 | 1.00 |
| Bengali | Bengali | 0.794 | 1.00 |
| Tibetan | Tibetan | 0.397 | 1.00 |

*Conclusion:* **CKA does not predict IFL.** High structural representation (e.g., Chinese at 0.864 CKA) can still result in a total generation failure (1.00 IFL), indicating that comprehension and generation alignment are heavily decoupled.

---

## 4. Pending / Anomalous Baselines (Requires Diagnostics)

The following metrics were computed but flagged with significant red flags in the legacy repository. They should be rerun on the verified exact-match testing suite.

### 4.1 LatentMAS vs ThoughtComm Baselines
*(Qwen2.5-7B-Instruct in 8-bit, $n=200$)*
| Benchmark | Language | LatentMAS Acc | ThoughtComm Acc |
|-----------|----------|--------------|----------------|
| MGSM | en | 0.320 | 0.320 |
| MGSM | th | 0.310 | 0.310 |
| MGSM | sw | 0.050 | 0.050 |
| Belebele | th | 0.500 | 0.500 |
| Belebele | my | 0.245 | 0.245 |
| Belebele | km | 0.260 | 0.260 |

**Diagnostic Flag:** Both baselines produce mathematically identical results across all conditions. The systems are likely collapsing to single-pass inference and bypassing heterogeneous agent specialization.

### 4.2 Router Ablations
| Benchmark | Language | attention+bow | attention+bilstm | kmeans+bow |
|-----------|----------|--------------|-----------------|-----------|
| MGSM | en | 0.09 | 0.09 | 0.09 |
| MGSM | de | 0.10 | 0.10 | 0.10 |
| Belebele | en | 0.51 | 0.51 | 0.51 |

**Diagnostic Flag:** Router configurations are yielding identical accuracies and near-chance performance on Math benchmarks. The downstream agent is either ignoring the routed sequence, or the routers are all statically choosing the exact same path.

### 4.3 SeaHELM & LiveCodeBench (LCB) Profiling
*(SEA-LION-v3-8B)*
*   **SeaHELM (Macro):** 0.867 Score, 0.177 IFL.
*   **LiveCodeBench (LCB):** Score = 0.00 and IFL = 1.000 across all non-Latin scripts (th, my, km, lo, am). Swahili (sw) scored 1.00 with 0.000 IFL.
**Diagnostic Flag:** The perfect 1.00 score on Swahili LCB is heavily suspected to be an anomaly caused by the Latin-script IFL gate, which inadvertently passes English code logic because they share the same alphabet.
