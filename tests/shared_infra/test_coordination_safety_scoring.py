"""Regression: multi-agent accuracy must score the substantive answer, not the
SafetyAgent verdict.

A token/latent chain ends with the SafetyAgent, whose output is always prefixed
``[SAFE]``/``[UNSAFE]``. The accuracy proxy rejects ``[``-prefixed text, so before
the fix every such chain scored 0.000 while the single-agent baseline scored 1.000.
These tests pin the corrected behaviour.
"""

import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List

from latent_coordination.eval.benchmark_runner import MultiAgentBenchmarkRunner
from latent_coordination.eval.scoring import is_safety_response, select_answer

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


@dataclass
class _Resp:
    agent_id: str
    output_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    latent_state: Any = None


@dataclass
class _Task:
    task_id: str
    query: str = "q"
    context: str = "ctx"
    target_language: str = "th"
    latent_state: Any = None
    reference: Any = "ref"


@dataclass
class _Plan:
    selected_agents: List[str]
    execution_order: List[str]


def _reason():
    return _Resp("agent_reason", "หนูทดลองเป็นโรคเบาหวาน")


def _safety():
    return _Resp("agent_safety", "[SAFE] Risk=0.00. Neutral text.",
                 {"safety_verdict": {"is_safe": True}})


# ---- unit: select_answer / is_safety_response -------------------------------

def test_select_answer_skips_trailing_safety():
    chain = [_reason(), _safety()]
    assert select_answer(chain).agent_id == "agent_reason"


def test_select_answer_all_safety_falls_back():
    only = [_safety()]
    assert select_answer(only).agent_id == "agent_safety"


def test_is_safety_response_detects_via_metadata_and_id():
    assert is_safety_response(_safety())
    assert not is_safety_response(_reason())
    # agent_id fallback when metadata is absent (older responses)
    assert is_safety_response(_Resp("agent_safety", "[UNSAFE: violence]"))


# ---- end-to-end through the evaluators --------------------------------------

class _Router:
    """Chain reason -> safety; safety output is bracketed (the old 0.000 trap)."""

    def __init__(self):
        self.agents = {
            "agent_reason": _AgentReason(),
            "agent_safety": _AgentSafety(),
        }

    def route(self, task):
        return _Plan(["agent_reason"], ["agent_reason", "agent_safety"])

    def execute(self, task, plan, space):
        class _R:
            agent_responses = [_reason(), _safety()]
        return _R()


class _AgentReason:
    def process(self, task):
        return _reason()


class _AgentSafety:
    def process(self, task):
        return _safety()


def _tasks(n=4):
    return [_Task(task_id=f"t{i}") for i in range(n)]


def test_token_mode_scores_answer_not_safety_verdict():
    runner = MultiAgentBenchmarkRunner(output_dir=tempfile.mkdtemp())
    metrics, answers = runner._eval_token_based(_Router(), _tasks())
    # Substantive reasoning answer is non-empty/non-bracketed -> 1.0 (was 0.000).
    assert metrics["accuracy"] == 1.0
    assert all(r.agent_id == "agent_reason" for r in answers)
    # Safety still measured separately from the verdict responses.
    assert metrics["safety_rate"] == 1.0


def test_latent_mode_scores_answer_not_safety_verdict():
    runner = MultiAgentBenchmarkRunner(output_dir=tempfile.mkdtemp())
    metrics, answers = runner._eval_latent(_Router(), _tasks(), universal_space=None)
    assert metrics["accuracy"] == 1.0
    assert all(r.agent_id == "agent_reason" for r in answers)
