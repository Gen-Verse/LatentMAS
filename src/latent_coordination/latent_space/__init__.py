"""Universal Latent Space: Hub-and-spoke latent state transfer between heterogeneous agents."""

from .universal_space import UniversalLatentHub
from .recursive_core import RecursiveLatentCore
from latent_coordination.latent_space.adapter import LatentAdapter, AdapterConfig, AdapterBank

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

__all__ = [
    "UniversalLatentHub",
    "RecursiveLatentCore",
    "LatentAdapter",
    "AdapterConfig",
    "AdapterBank",
]
