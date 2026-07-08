"""Regression tests for the 2026-07-03 evaluation-pipeline audit fixes.

Each test pins one confirmed bug found while reconciling the ported
MultilingualLatentMAS eval configuration/pipeline against the LRL-MRRE-MAS
strategy documents:

1. Accuracy denominator: tasks that produced no answer were silently dropped
   from the denominator, inflating every mode's accuracy.
2. Token cost used whitespace `split()`, which counts an entire unsegmented
   Thai/Burmese/Khmer sentence as ~1 token.
3. route() must emit agents in canonical order (translation → reasoning →
   safety); iterating a set() of roles was PYTHONHASHSEED-random.
4. execute() ("latent" mode) passed each agent's decoded text to the next agent
   as `context` — a hidden token side-channel inside the mode whose headline
   claim is 0 inter-agent tokens.
5. shared.metrics SFR was token-level (near-binary 0/1 for unsegmented
   scripts); it must be character-level.
6. The single-agent baseline must not pick the safety agent as sole executor.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from latent_coordination.agents.base_agent import AgentConfig, AgentTask
from latent_coordination.eval.benchmark_runner import MultiAgentBenchmarkRunner
from latent_coordination.latent_space.universal_space import UniversalLatentHub
from latent_coordination.orchestration.router import (
    AdaptiveOrchestrator,
    RoutingPlan,
    canonical_role_sort,
    encode_query_bow,
    QUERY_EMBED_DIM,
)

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

@dataclass
class _Resp:
    agent_id: str = "agent_reason"
    output_text: str = "คำตอบภาษาไทย"
    metadata: Dict[str, Any] = field(default_factory=dict)
    latent_state: Any = None
    task_id: str = "t"
    confidence: Optional[float] = None
    elapsed_ms: float = 0.0


@dataclass
class _Task:
    task_id: str
    query: str = "q"
    context: str = ""
    target_language: str = "th"
    latent_state: Any = None
    reference: Any = "ref"


@dataclass
class _Plan:
    selected_agents: List[str]
    execution_order: List[str]


class _Agent:
    def __init__(self, agent_id="agent_reason", role="reasoning", text="คำตอบภาษาไทย"):
        self.config = AgentConfig(agent_id=agent_id, model_id="stub", role=role)
        self._text = text

    def process(self, task):
        return _Resp(agent_id=self.config.agent_id, output_text=self._text)


# ---------------------------------------------------------------------------
# 1. Accuracy denominator counts unanswered tasks as failures
# ---------------------------------------------------------------------------

def test_accuracy_denominator_counts_dropped_tasks_as_failures(tmp_path):
    runner = MultiAgentBenchmarkRunner(output_dir=tmp_path)
    answers = [_Resp(output_text="ok")]          # only 1 of 4 tasks answered
    tasks = [_Task(task_id=f"t{i}") for i in range(4)]
    # Before the fix this returned 1/1 = 1.0; the three unanswered tasks vanished.
    assert runner._compute_accuracy(answers, tasks) == 0.25


# ---------------------------------------------------------------------------
# 2. Token counting uses the agent tokenizer (unsegmented scripts)
# ---------------------------------------------------------------------------

def test_count_tokens_uses_agent_tokenizer_not_whitespace(tmp_path):
    class _Tok:
        def encode(self, text, add_special_tokens=False):
            return list(text)  # 1 token per char, like a real th/my/km tokenizer

    agent = _Agent()
    agent._tokenizer = _Tok()
    thai = "ประโยคภาษาไทยไม่มีช่องว่าง"  # no whitespace at all
    n = MultiAgentBenchmarkRunner._count_tokens(thai, agent)
    assert n == len(thai)          # real tokenizer count
    assert len(thai.split()) == 1  # what the old whitespace proxy reported


# ---------------------------------------------------------------------------
# 3. Canonical, deterministic role ordering
# ---------------------------------------------------------------------------

def test_canonical_role_sort_orders_translation_reasoning_safety():
    assert canonical_role_sort({"safety", "translation", "reasoning"}) == [
        "translation", "reasoning", "safety",
    ]


def test_route_emits_agents_in_canonical_order():
    router = AdaptiveOrchestrator(device="cpu")
    # Register in scrambled order on purpose.
    router.register_agent(_Agent("agent_safety", "safety"))
    router.register_agent(_Agent("agent_trans", "translation"))
    router.register_agent(_Agent("agent_reason", "reasoning"))
    plan = router.route(AgentTask(task_id="t0", query="translate this sentence"))
    roles_in_order = [router.agents[a].config.role for a in plan.execution_order]
    assert roles_in_order == canonical_role_sort(roles_in_order)
    if "safety" in roles_in_order:
        assert roles_in_order[-1] == "safety"


def test_route_query_embedding_dim_matches_centroid_space():
    # route() and centroid fitting must share one embedding space (32 vs 64 mismatch
    # crashed the k-means routing path before the fix).
    assert encode_query_bow("hello world").shape == (QUERY_EMBED_DIM,)


# ---------------------------------------------------------------------------
# 4. Latent mode has no inter-agent text side-channel
# ---------------------------------------------------------------------------

def test_execute_does_not_pass_previous_output_as_context():
    seen_contexts: List[str] = []

    class _RecordingAgent(_Agent):
        def process(self, task):
            seen_contexts.append(task.context)
            return _Resp(
                agent_id=self.config.agent_id,
                output_text=f"decoded output of {self.config.agent_id}",
                latent_state=torch.randn(1, 4, 64),
            )

    router = AdaptiveOrchestrator(device="cpu")
    a1, a2 = _RecordingAgent("agent_trans", "translation"), _RecordingAgent("agent_reason", "reasoning")
    a1.config.hidden_dim = 64
    a2.config.hidden_dim = 64
    router.register_agent(a1)
    router.register_agent(a2)
    uls = UniversalLatentHub(universal_dim=32)

    task = AgentTask(task_id="t0", query="q", context="ORIGINAL", target_language="th")
    plan = RoutingPlan(task_id="t0", selected_agents=["agent_trans", "agent_reason"],
                       execution_order=["agent_trans", "agent_reason"], estimated_cost=0.0)
    result = router.execute(task, plan, uls)

    # Every agent sees only the ORIGINAL task context — never the previous agent's
    # decoded text (that was the hidden token channel inside the "0-token" mode).
    assert seen_contexts == ["ORIGINAL", "ORIGINAL"]
    # And the reported inter-agent token cost is genuinely zero.
    assert result.communication_cost_tokens == 0


# ---------------------------------------------------------------------------
# 5. SFR is character-level for unsegmented scripts
# ---------------------------------------------------------------------------

def test_shared_sfr_is_character_level_for_thai():
    from shared.metrics import _detect_script_ratio

    # One Thai word embedded in an English sentence, no Thai-side whitespace.
    mixed = "The answer is สวัสดี and nothing else"
    ratio = _detect_script_ratio(mixed, "th")
    assert 0.0 < ratio < 0.5  # old token-level version returned ~1/7 of *tokens*
    pure_thai = "สวัสดีครับผมชื่อสมชาย"
    assert _detect_script_ratio(pure_thai, "th") == 1.0
    assert _detect_script_ratio("Entirely English text", "th") == 0.0


# ---------------------------------------------------------------------------
# 6. Single-agent baseline never elects the safety agent as sole executor
# ---------------------------------------------------------------------------

def test_single_agent_baseline_prefers_non_safety_agent(tmp_path):
    runner = MultiAgentBenchmarkRunner(output_dir=tmp_path)

    class _Router:
        def __init__(self):
            self.agents = {
                "agent_safety": _Agent("agent_safety", "safety", "[SAFE] Risk=0.00."),
                "agent_reason": _Agent("agent_reason", "reasoning"),
            }

    plan = _Plan(["agent_safety", "agent_reason"], ["agent_safety", "agent_reason"])
    picked = runner._pick_single_agent(_Router(), plan)
    assert picked == "agent_reason"


# ---------------------------------------------------------------------------
# 7. Module A+B adapter training exists and actually improves the hub
# ---------------------------------------------------------------------------

def test_fit_adapters_improves_roundtrip_fidelity():
    torch.manual_seed(0)
    hub = UniversalLatentHub(universal_dim=16, adapter_hidden_dim=32)
    hub.register_agent("a", hidden_dim=24)
    hub.register_agent("b", hidden_dim=40)
    states = {"a": torch.randn(32, 24), "b": torch.randn(32, 40)}
    before = hub.compute_transfer_quality("a", states["a"])["cosine_similarity"]
    losses = hub.fit_adapters(states, n_epochs=30, lr=1e-2, batch_size=16)
    after = hub.compute_transfer_quality("a", states["a"])["cosine_similarity"]
    assert after > before
    assert set(losses) == {"recon", "dae", "cka", "total"}


def test_fit_adapters_refuses_unaligned_or_empty_states():
    import pytest

    hub = UniversalLatentHub(universal_dim=16)
    hub.register_agent("a", hidden_dim=8)
    hub.register_agent("b", hidden_dim=8)
    with pytest.raises(ValueError):
        hub.fit_adapters({})
    with pytest.raises(ValueError):
        hub.fit_adapters({"a": torch.randn(10, 8), "b": torch.randn(12, 8)})


def test_unbiased_cka_loss_zero_for_identical_inputs():
    from latent_coordination.latent_space.universal_space import cka_loss_unbiased

    x = torch.randn(16, 8)
    assert float(cka_loss_unbiased(x, x)) < 1e-4
