"""Evaluation utilities for Surgical MRRE.

Reuses the mechanistic project's :class:`ScriptFidelityEvaluator` for script-fidelity
measurement (no duplicated scoring logic, no fabricated metrics).
"""

from mrre_drift.eval.ifl import IFLReport, IFLValidator
from mrre_drift.eval.dsl import DSLCorrector, DSLReport

__all__ = ["IFLValidator", "IFLReport", "DSLCorrector", "DSLReport"]
