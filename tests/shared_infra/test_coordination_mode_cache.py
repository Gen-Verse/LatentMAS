"""Per-mode selection + caching in the coordination benchmark runner.

Uses lightweight stubs (no real model) to prove the modular-resume contract:
a cached mode is reused, and only newly-requested modes are computed.
"""

import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List

from latent_coordination.eval.benchmark_runner import MultiAgentBenchmarkRunner
from shared.checkpointing import CheckpointManager

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
    output_text: str = "สวัสดีโลก"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Task:
    task_id: str
    query: str = ""
    context: str = "ctx"
    target_language: str = "th"
    latent_state: Any = None


@dataclass
class _Plan:
    selected_agents: List[str]
    execution_order: List[str]


class _Agent:
    def process(self, task):
        return _Resp()


class _Router:
    def __init__(self):
        self.agents = {"a": _Agent(), "b": _Agent()}

    def route(self, task):
        return _Plan(["a"], ["a", "b"])

    def execute(self, task, plan, space):
        class _R:
            agent_responses = [_Resp()]
        return _R()


def _tasks(n=3):
    return [_Task(task_id=f"t{i}") for i in range(n)]


def test_single_mode_runs_and_caches():
    cm = CheckpointManager(tempfile.mkdtemp(), "coordination")
    runner = MultiAgentBenchmarkRunner(output_dir=tempfile.mkdtemp())
    rep = runner.run_eval(
        _Router(), _tasks(), None, modes=["token_based_mas"],
        checkpoint_manager=cm, cache_prefix="coord::M",
    )
    assert list(rep.results_by_mode) == ["token_based_mas"]
    assert cm.has_result("coord::M::mode::token_based_mas")


def test_added_mode_reuses_cached():
    cm = CheckpointManager(tempfile.mkdtemp(), "coordination")
    runner = MultiAgentBenchmarkRunner(output_dir=tempfile.mkdtemp())
    runner.run_eval(_Router(), _tasks(), None, modes=["token_based_mas"],
                    checkpoint_manager=cm, cache_prefix="coord::M")

    calls = {"token": 0}
    orig = runner._eval_token_based

    def _spy(*a, **k):
        calls["token"] += 1
        return orig(*a, **k)

    runner._eval_token_based = _spy
    rep = runner.run_eval(
        _Router(), _tasks(), None,
        modes=["token_based_mas", "latent_based_mas_ours"],
        checkpoint_manager=cm, cache_prefix="coord::M",
    )
    assert sorted(rep.results_by_mode) == ["latent_based_mas_ours", "token_based_mas"]
    assert calls["token"] == 0  # token mode came from cache, not recomputed


def test_invalid_mode_rejected():
    import pytest
    runner = MultiAgentBenchmarkRunner(output_dir=tempfile.mkdtemp())
    with pytest.raises(ValueError):
        runner.run_eval(_Router(), _tasks(), None, modes=["bogus_mode"])
