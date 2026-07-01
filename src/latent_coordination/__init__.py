"""
Latent Coordination Package (Unified): Multi-Agent Planning with Latent Coordination,
Transferable Topology, Adaptive Orchestration, and Cross-Lingual Latent Steering.
"""


# Latent Coordination & Reasoning exports
from latent_coordination.topology.cvae_prior import CVAETopologyPrior
from latent_coordination.latent_space.universal_space import UniversalLatentSpace
from latent_coordination.orchestration.router import AdaptiveOrchestrator, AttentionRouter
from latent_coordination.agents.base_agent import BaseAgent
from latent_coordination.eval.adversarial import BoundedLatentAttacker, LatentGateDefense
from latent_coordination.eval.information_theory import InfoTheoreticAnalyzer
from latent_coordination.baselines.latent_mas import LatentMASBaseline
from latent_coordination.baselines.blackboard_mas import BlackboardMASBaseline
from latent_coordination.baselines.thoughtcomm import ThoughtCommBaseline, ThoughtCommConfig
from latent_coordination.baselines.cache_to_cache import CacheToCacheBaseline
from latent_coordination.baselines.gdesigner_mas_router import (
    GDesignerBaseline,
    MasRouterBaseline,
    MasRouterConfig,
)

# Latent Steering / Mechanistic Disentanglement exports
from latent_coordination.geometry.svd_decomposer import SVDSubspaceDecomposer
from latent_coordination.geometry.isomorphism import GeometricIsomorphismAnalyzer
from latent_coordination.steering.gaussian_scheduler import GaussianDepthScheduler
from latent_coordination.steering.magnitude_norm import MagnitudeNormalizer
from latent_coordination.steering.latent_steerer import LatentSteerer
from latent_coordination.eval.script_fidelity import ScriptFidelityEvaluator

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
    "UniversalLatentSpace",
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
