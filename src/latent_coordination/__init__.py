"""
Latent Coordination Package (Unified): Multi-Agent Planning with Latent Coordination,
Transferable Topology, Adaptive Orchestration, and Cross-Lingual Latent Steering.
"""


# Latent Coordination & Reasoning exports
from .topology.cvae_prior import GeometryConditionedCVAEPrior
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

# Latent Steering / Mechanistic Disentanglement exports
from .geometry.svd_decomposer import SVDSubspaceDecomposer
from .geometry.isomorphism import GeometricIsomorphismAnalyzer
from .steering.gaussian_scheduler import GaussianDepthScheduler
from .steering.magnitude_norm import MagnitudeNormalizer
from .steering.latent_steerer import LatentSteerer
from .eval.script_fidelity import ScriptFidelityEvaluator

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

__all__ = [
    "GeometryConditionedCVAEPrior",
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
    
    "SVDSubspaceDecomposer",
    "GeometricIsomorphismAnalyzer",
    "GaussianDepthScheduler",
    "MagnitudeNormalizer",
    "LatentSteerer",
    "ScriptFidelityEvaluator",
]
