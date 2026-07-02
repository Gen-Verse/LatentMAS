"""Evaluation: Script Fidelity Rate, benchmark runners, metrics."""


from latent_coordination.eval.efficiency_metrics import EfficiencyAnalyzer, AblationReport
from .adversarial import BoundedLatentAttacker, LatentGateDefense
from .information_theory import InfoTheoreticAnalyzer
from .script_fidelity import ScriptFidelityEvaluator, LanguageConsistencyEvaluator
from .multi_agent_runner import MultiAgentRunner
from .verification_probe import QueryReconstructionProbe

__all__ = [
    "BoundedLatentAttacker",
    "LatentGateDefense",
    "InfoTheoreticAnalyzer",
    "ScriptFidelityEvaluator",
    "LanguageConsistencyEvaluator",
    "MultiAgentRunner",
    "QueryReconstructionProbe"
]

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"
