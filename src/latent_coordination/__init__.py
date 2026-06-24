"""
Latent Coordination Paper: Multi-Agent Planning with Latent Coordination,
Transferable Topology, and Adaptive Orchestration.

This package implements:
    - CVAE Graph Topology Prior (TopoPrior)
    - Hub-and-Spoke Universal Latent Space (L-MAS)
    - Text-free latent state transfer between heterogeneous agents
    - Adaptive orchestration with latent intent centroids (TRIAD-TS style)
    - Dynamic sub-task routing to specialized agents
"""

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

from latent_coordination.topology.cvae_prior import CVAETopologyPrior
from latent_coordination.latent_space.universal_space import UniversalLatentSpace
from latent_coordination.orchestration.router import AdaptiveOrchestrator
from latent_coordination.agents.base_agent import BaseAgent

__all__ = [
    "CVAETopologyPrior",
    "UniversalLatentSpace",
    "AdaptiveOrchestrator",
    "BaseAgent",
]
