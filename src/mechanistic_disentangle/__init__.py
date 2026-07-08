"""
Mechanistic Disentanglement Package (Paper 2): SVD subspace decomposition,
contrastive geometric analysis, Gaussian depth-scheduled steering, and
geometric isomorphism probes.

Firewall (strategy.md §6): this package owns ALL SVD math, projection
operators, and steering schedules. It must never import from
``latent_coordination`` (and vice versa); shared, non-technical infra lives in
``src/shared/``. Run ``scripts/firewall_check.sh`` to verify both rules.
"""

from .geometry.svd_decomposer import SVDSubspaceDecomposer
from .geometry.isomorphism import GeometricIsomorphismAnalyzer
from .geometry.activation_extractor import ActivationExtractor
from .steering.gaussian_scheduler import GaussianDepthScheduler
from .steering.magnitude_norm import MagnitudeNormalizer
from .steering.latent_steerer import LatentSteerer
from .steering.vector_builder import SteeringVectorBuilder

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

__all__ = [
    "SVDSubspaceDecomposer",
    "GeometricIsomorphismAnalyzer",
    "ActivationExtractor",
    "GaussianDepthScheduler",
    "MagnitudeNormalizer",
    "LatentSteerer",
    "SteeringVectorBuilder",
]
