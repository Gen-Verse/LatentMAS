"""Shared answer-selection helpers for multi-agent benchmark scoring.

A multi-agent chain typically ends with the :class:`SafetyAgent`, whose output is
a structured verdict prefixed ``[SAFE]`` / ``[UNSAFE: ...]`` rather than the task
answer. Scoring that verdict as the task output is wrong: the only bracketed
strings in the system are safety verdicts, so any accuracy heuristic that rejects
``[``-prefixed text will spuriously zero out every chain that ends in the safety
gate (token-based and latent modes), while a single-agent baseline that never runs
the safety gate scores 1.0. The fix is to score the *substantive* answer — the last
non-safety response in execution order — and to measure safety separately.
"""

from typing import Any, List, Optional

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"



def is_safety_response(resp: Any) -> bool:
    """True if ``resp`` is a SafetyAgent verdict rather than a task answer.

    Detected via the ``safety_verdict`` key the SafetyAgent writes into response
    metadata, with an ``agent_id`` fallback for responses that predate it.
    """
    if resp is None:
        return False
    meta = getattr(resp, "metadata", None) or {}
    if "safety_verdict" in meta:
        return True
    aid = getattr(resp, "agent_id", "") or ""
    return aid.endswith("safety")


def select_answer(responses: List[Any]) -> Optional[Any]:
    """Return the substantive answer = last non-safety response in execution order.

    Falls back to the final response if every step was a safety check, and returns
    ``None`` for an empty list.
    """
    if not responses:
        return None
    for resp in reversed(responses):
        if not is_safety_response(resp):
            return resp
    return responses[-1]
