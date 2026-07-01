"""Baseline MAS communication strategies for fair comparison with LatentMAS.

Baselines:
    LatentMASBaseline      — training-free last-layer hidden-state sharing
                             (homogeneous-only, models LatentMAS ICML 2026)
    BlackboardMASBaseline  — shared-memory text MAS, O(N) reads/writes
    ThoughtCommBaseline    — shared/private latent decomposition (NeurIPS 2025)
    CacheToCacheBaseline   — pairwise KV-cache projection, O(N²) (arXiv:2510.03215)
    GDesignerBaseline      — VGAE query-conditioned topology (ICML 2025)
    MasRouterBaseline      — cascaded mode+role routing (ACL 2025)
"""

from latent_coordination.baselines.latent_mas import LatentMASBaseline
from latent_coordination.baselines.blackboard_mas import BlackboardMASBaseline
from latent_coordination.baselines.thoughtcomm import ThoughtCommBaseline, ThoughtCommConfig
from latent_coordination.baselines.cache_to_cache import CacheToCacheBaseline
from latent_coordination.baselines.gdesigner_mas_router import (
    GDesignerBaseline,
    MasRouterBaseline,
    MasRouterConfig,
)
from latent_coordination.baselines.vision_wormhole import VisionWormholeBaseline
from latent_coordination.baselines.regional_eval_baselines import (
    SeaHelmBaseline,
    SeaCrowdBaseline,
    SeaEvalBaseline,
    SeaLionBaseline,
    SeaExamBaseline,
    SeaBenchBaseline,
    MultiChallengeBaseline,
    XQuadBaseline,
    MlqaBaseline,
)

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

__all__ = [
    "LatentMASBaseline",
    "BlackboardMASBaseline",
    "ThoughtCommBaseline",
    "ThoughtCommConfig",
    "CacheToCacheBaseline",
    "GDesignerBaseline",
    "MasRouterBaseline",
    "MasRouterConfig",
    "VisionWormholeBaseline",
    "SeaHelmBaseline",
    "SeaCrowdBaseline",
    "SeaEvalBaseline",
    "SeaLionBaseline",
    "SeaExamBaseline",
    "SeaBenchBaseline",
    "MultiChallengeBaseline",
    "XQuadBaseline",
    "MlqaBaseline",
]
