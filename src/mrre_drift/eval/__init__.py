"""Evaluation utilities for Surgical MRRE.

Reuses the mechanistic project's :class:`ScriptFidelityEvaluator` for script-fidelity
measurement (no duplicated scoring logic, no fabricated metrics).
"""

from mrre_drift.eval.ifl import IFLReport, IFLValidator
from mrre_drift.eval.dsl import DSLCorrector, DSLReport

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

__all__ = ["IFLValidator", "IFLReport", "DSLCorrector", "DSLReport"]
