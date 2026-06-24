"""Latent Coordination evaluation: multi-agent efficiency, convergence, communication cost analysis."""

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

from latent_coordination.eval.efficiency_metrics import EfficiencyAnalyzer, AblationReport
from latent_coordination.eval.benchmark_runner import MultiAgentBenchmarkRunner, MultiAgentBenchmarkReport

__all__ = [
    "EfficiencyAnalyzer",
    "AblationReport",
    "MultiAgentBenchmarkRunner",
    "MultiAgentBenchmarkReport",
]
