"""Full Latent Coordination pipeline: CVAE training, system setup, multi-agent evaluation."""

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

from latent_coordination.pipeline.coordination_pipeline import CoordinationPipeline, CoordinationPipelineConfig

PipelineConfig = CoordinationPipelineConfig

__all__ = [
    "CoordinationPipeline",
    "CoordinationPipelineConfig",
    "PipelineConfig",
]
