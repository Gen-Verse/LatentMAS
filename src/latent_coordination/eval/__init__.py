"""Evaluation: Script Fidelity Rate, benchmark runners, metrics."""


from latent_coordination.eval.efficiency_metrics import EfficiencyAnalyzer, AblationReport
from latent_coordination.eval.benchmark_runner import MultiAgentBenchmarkRunner, MultiAgentBenchmarkReport
from latent_coordination.eval.steering_benchmark import BenchmarkRunner, BenchmarkReport
from latent_coordination.eval.script_fidelity import ScriptFidelityEvaluator
from latent_coordination.eval.metrics import MetricsComputer

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"
