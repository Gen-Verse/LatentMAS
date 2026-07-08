"""
Latent Coordination Package (Paper 3): Multi-Agent Planning with Latent Coordination,
Transferable Topology, and Adaptive Orchestration.

Firewall (strategy.md §6): this package is SVD-free — all SVD/projection/steering
math lives exclusively in ``src/mechanistic_disentangle`` and must never be
imported here. Precomputed geometry diagnostics (the Geo_L risk vector) arrive as
plain data artifacts via ``topology.geo_profile``. Run ``scripts/firewall_check.sh``
to verify.
"""


# Latent Coordination & Reasoning exports
from .topology.cvae_prior import CVAETopologyPrior, TrainingConfig
from .latent_space.universal_space import UniversalLatentHub
from .orchestration.router import AdaptiveOrchestrator, AttentionRouter
from .agents.base_agent import BaseAgent
from .eval.adversarial import BoundedLatentAttacker, LatentGateDefense
from .eval.information_theory import InfoTheoreticAnalyzer
from .baselines.latent_mas import LatentMASBaseline
from .baselines.blackboard_mas import BlackboardMASBaseline
from .baselines.thoughtcomm import ThoughtCommBaseline, ThoughtCommConfig
from .baselines.cache_to_cache import CacheToCacheBaseline
from .baselines.gdesigner_mas_router import (
    GDesignerBaseline,
    MasRouterBaseline,
    MasRouterConfig,
)

# Shared metric evaluators (implementation lives in shared/script_fidelity.py)
from .eval.script_fidelity import ScriptFidelityEvaluator, LanguageConsistencyEvaluator

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

__all__ = [
    "CVAETopologyPrior",
    "TrainingConfig",
    "UniversalLatentHub",
    "AdaptiveOrchestrator",
    "AttentionRouter",
    "BaseAgent",
    "BoundedLatentAttacker",
    "LatentGateDefense",
    "InfoTheoreticAnalyzer",
    "LatentMASBaseline",
    "BlackboardMASBaseline",
    "ThoughtCommBaseline",
    "ThoughtCommConfig",
    "CacheToCacheBaseline",
    "GDesignerBaseline",
    "MasRouterBaseline",
    "MasRouterConfig",
    "ScriptFidelityEvaluator",
    "LanguageConsistencyEvaluator",
]
