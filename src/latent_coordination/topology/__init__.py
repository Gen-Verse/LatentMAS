"""Graph topology priors: CVAE encoder-decoder for transferable collaboration graphs."""

from .cvae_prior import CVAETopologyPrior, TrainingConfig, TopologyDataset, free_bits_kl, active_units
from latent_coordination.topology.graph_utils import GraphUtils, GraphProperties

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
    "TopologyDataset",
    "free_bits_kl",
    "active_units",
    "GraphUtils",
    "GraphProperties",
]
