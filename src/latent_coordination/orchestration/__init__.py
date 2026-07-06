"""Orchestration: adaptive routing, latent intent centroids, task decomposition."""


from latent_coordination.orchestration.router import (
    AdaptiveOrchestrator,
    LatentIntentCentroid,
    RoutingPlan,
    OrchestrationResult,
)
from latent_coordination.orchestration.task_decomposer import TaskDecomposer, SubTask

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

__all__ = [
    "AdaptiveOrchestrator",
    "LatentIntentCentroid",
    "RoutingPlan",
    "OrchestrationResult",
    "TaskDecomposer",
    "SubTask",
]
