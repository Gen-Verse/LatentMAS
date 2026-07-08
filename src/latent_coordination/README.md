# Unified Latent Coordination & Steering Package

This package contains the unified implementation of two core papers/frameworks for low-resource languages (LRL) and multi-agent systems (MAS):
1. **Decentralized Multi-Agent Latent Coordination (Latent MAS)**: Text-free multi-agent reasoning via a shared continuous latent space, utilizing CVAE graph priors, universal adapter-based spaces, and intent-based dispatch routing.
2. **Mechanistic Disentanglement & Geometric Isomorphism (Latent Steering)**: Training-free cross-lingual latent steering using SVD-based subspace decomposition, Gaussian depth scheduling, and magnitude normalization.

---

## Directory Structure & Component Mapping

The unified package is structured as follows:

```
src/latent_coordination/
├── agents/             # Multi-agent coordination agents (Base, Translation, Reasoning, Safety)
├── baselines/          # Real MAS and baseline implementations (LatentMAS, ThoughtComm, G-Designer)
├── data/               # Latent steering data utilities (ContrastiveLexicon, DatasetLoader)
├── eval/               # Combined evaluation metrics, scorers, and benchmark runners
│   ├── benchmark_runner.py    # Multi-agent coordination benchmark runner
│   ├── steering_benchmark.py  # Latent steering evaluation benchmark runner
│   └── script_fidelity.py     # Script Fidelity Rate (SFR) evaluator
├── geometry/           # Latent steering geometric analysis (SVD decomposers, isomorphisms)
├── latent_space/       # Universal Latent Space (L-MAS) mapping and adapter architectures
├── orchestration/      # Adaptive orchestrator (TRIAD-TS style) and task decomposition
├── pipeline/           # Unified execution pipelines
│   ├── coordination_pipeline.py  # Latent Coordination (CVAE + MAS) pipeline
│   └── mechanistic_pipeline.py   # Latent Steering (Gaussian-scheduled) pipeline
├── steering/           # Latent steering controllers (Gaussian depth scheduler, magnitude normalizer)
├── topology/           # CVAE Graph Topology Prior (TopoPrior) & topology dataset mapping
├── utils/              # Localized caching, logging, device parallelism wrappers
└── viz/                # Combined visualizations (topology, efficiency, steering, geometry)
```

---

## Core Capabilities

### 1. Decentralized Multi-Agent Latent Coordination
* **TopoPrior (CVAE Graph Prior)**: A $\beta$-CVAE model with cyclical KL annealing that learns $p(G|Q)$, mapping collaboration topologies directly to task queries.
* **Universal Latent Space (L-MAS)**: A shared continuous manifold mapping heterogeneous model hidden states via hub-and-spoke adapters.
* **TRIAD-TS Centroid Routing**: Intent clustering using PyTorch $k$-means to route tasks to specific role sequences (translation $\rightarrow$ reasoning $\rightarrow$ safety).
* **Specialized Agents**:
  - `TranslationAgent`: Translates instructions with quality checks and latent state transfer.
  - `ReasoningAgent`: Amplifies critical reasoning subspaces using SVD-based mapping.
  - `SafetyAgent`: Runs safety checks using fallbacks.

### 2. Mechanistic Disentanglement & Latent Steering
* **Subspace Decomposition**: Contrastive SVD isolates language-specific axes ($U_L$) from the semantic/reasoning manifold ($U_R$) using paired activations.
* **Gaussian Depth Scheduling**: Focuses steering weight in mid-stack reasoning layers ($\alpha_l = \alpha_0 \exp(-(l-\mu_s)^2 / 2\sigma_s^2)$) to protect token formation and projection boundaries.
* **Magnitude Normalization**: Dynamic scaling ($\gamma_l = \eta \|h_{tgt}\| / \|v_{steer}\|$) to avoid over-saturating LRL representation spaces.
* **Fidelity Gating**: Uses Script-Fidelity-Rate (SFR) evaluation and Image-Induced Fidelity Loss (IFL) correction to track and minimize script/language drift.

---

## Unified APIs

The package exports all unified classes and helpers from its root `__init__.py`:

```python
from latent_coordination import (
    # Coordination & Topology
    CVAETopologyPrior,
    UniversalLatentSpace,
    AdaptiveOrchestrator,
    BaseAgent,
    
    # Steering & Disentanglement
    SVDSubspaceDecomposer,
    GeometricIsomorphismAnalyzer,
    GaussianDepthScheduler,
    MagnitudeNormalizer,
    LatentSteerer,
)
```

---

## Executing Pipelines

Run the coordination pipeline:
```bash
PYTHONPATH=src python scripts/run_coordination_pipeline.py --config configs/latent_coordination.yaml
```

Run tests to verify the unified architecture:
```bash
PYTHONPATH=src pytest tests/
```
