"""Regression test: AdaptiveOrchestrator.execute() must not corrupt a receiver
agent's adapter registration during a heterogeneous latent hand-off.

Found live on 2026-07-02: execute() registered the *receiving* agent using the
*sender's* outgoing hidden_state.shape[-1] (not the receiver's own hidden_dim), and
UniversalLatentHub.register_agent() unconditionally overwrites with a fresh,
untrained adapter on every call. Together these silently swapped a correctly-sized,
Stage-C-trained adapter for a wrong-shaped one on every hand-off, which crashed
`latent_based_mas_ours` on a real heterogeneous run (Sailor2 hidden_dim=3584 ->
Llama-3.1 hidden_dim=4096) with
`RuntimeError: Sizes of tensors must match ... Expected size 3584 but got size 4096`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from latent_coordination.agents.base_agent import AgentConfig, AgentTask
from latent_coordination.latent_space.universal_space import UniversalLatentHub
from latent_coordination.orchestration.router import AdaptiveOrchestrator, RoutingPlan

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"


@dataclass
class _FakeResponse:
    task_id: str
    agent_id: str
    output_text: str
    latent_state: Optional[torch.Tensor] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


class _FakeAgent:
    """Emits a fixed-shape hidden state matching its own configured hidden_dim,
    regardless of what latent_state it received -- like a real HF model would."""

    def __init__(self, config: AgentConfig):
        self.config = config

    def process(self, task: AgentTask) -> _FakeResponse:
        return _FakeResponse(
            task_id=task.task_id,
            agent_id=self.config.agent_id,
            output_text=f"output from {self.config.agent_id}",
            latent_state=torch.randn(1, 4, self.config.hidden_dim),
        )


def _make_router_with_two_heterogeneous_agents():
    router = AdaptiveOrchestrator(device="cpu")
    sender = _FakeAgent(AgentConfig(agent_id="agent_trans", model_id="fake/sailor2", hidden_dim=3584))
    receiver = _FakeAgent(AgentConfig(agent_id="agent_reason", model_id="fake/llama31", hidden_dim=4096))
    router.register_agent(sender)
    router.register_agent(receiver)
    return router, sender, receiver


def test_execute_preserves_receiver_own_hidden_dim_across_handoff():
    router, sender, receiver = _make_router_with_two_heterogeneous_agents()
    uls = UniversalLatentHub(universal_dim=64)
    uls.register_agent("agent_trans", hidden_dim=3584)
    uls.register_agent("agent_reason", hidden_dim=4096)

    task = AgentTask(task_id="t0", query="q", target_language="th")
    plan = RoutingPlan(task_id="t0", selected_agents=["agent_trans", "agent_reason"],
                       execution_order=["agent_trans", "agent_reason"], estimated_cost=0.0)

    result = router.execute(task, plan, uls)

    # The receiver's adapter must still be its own true dimension, not silently
    # swapped to the sender's -- this is what the crash's root cause corrupted.
    assert uls._agents["agent_reason"].hidden_dim == 4096
    assert uls._agents["agent_trans"].hidden_dim == 3584
    assert len(result.agent_responses) == 2


def test_execute_does_not_recreate_adapters_already_registered():
    """register_agent() overwrites unconditionally on every call; execute() must
    not re-invoke it for agents already registered, or Stage C's trained adapters
    get silently replaced by fresh random ones on every hand-off."""
    router, sender, receiver = _make_router_with_two_heterogeneous_agents()
    uls = UniversalLatentHub(universal_dim=64)
    uls.register_agent("agent_trans", hidden_dim=3584)
    uls.register_agent("agent_reason", hidden_dim=4096)

    # Mark the encoder/decoder objects so we can detect if they get replaced.
    original_reason_encoder = uls._agents["agent_reason"].encoder
    original_reason_decoder = uls._agents["agent_reason"].decoder

    task = AgentTask(task_id="t0", query="q", target_language="th")
    plan = RoutingPlan(task_id="t0", selected_agents=["agent_trans", "agent_reason"],
                       execution_order=["agent_trans", "agent_reason"], estimated_cost=0.0)
    router.execute(task, plan, uls)

    assert uls._agents["agent_reason"].encoder is original_reason_encoder
    assert uls._agents["agent_reason"].decoder is original_reason_decoder
