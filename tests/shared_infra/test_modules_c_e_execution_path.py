"""Regression tests for dev_doc.md §9 gap 3: Module C (RecursiveLatentCore) and
Module E (drift probe) must be IN the multi-agent execution path, not dead code.

Uses stub agents (no models) — the point is the orchestrator's hub-space
transfer plumbing: encode → refine (C) → verify/repair (E) → decode.
"""

import torch

from latent_coordination.agents.base_agent import (
    AgentConfig,
    AgentResponse,
    AgentTask,
    BaseAgent,
)
from latent_coordination.eval.verification_probe import QueryReconstructionProbe
from latent_coordination.latent_space.recursive_core import RecursiveLatentCore
from latent_coordination.latent_space.universal_space import UniversalLatentHub
from latent_coordination.orchestration.router import (
    QUERY_EMBED_DIM,
    AdaptiveOrchestrator,
    RoutingPlan,
    encode_query_bow,
)

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"

HIDDEN = 32
HUB = 16


class _StubAgent(BaseAgent):
    """Echo agent that returns a fixed latent state without loading a model."""

    def process(self, task: AgentTask) -> AgentResponse:
        return AgentResponse(
            task_id=task.task_id,
            agent_id=self.agent_id,
            output_text=f"answer from {self.agent_id}",
            latent_state=torch.randn(1, 3, HIDDEN),
        )


def _setup(with_core=False, with_probe=False, tau_drift=0.5, fit_probe=True):
    router = AdaptiveOrchestrator(device="cpu")
    for aid, role in [("agent_trans", "translation"), ("agent_reason", "reasoning")]:
        router.register_agent(_StubAgent(AgentConfig(
            agent_id=aid, model_id="stub", role=role, device="cpu", hidden_dim=HIDDEN,
        )))
    hub = UniversalLatentHub(universal_dim=HUB)
    hub.register_agent("source", hidden_dim=HIDDEN)
    hub.register_agent("agent_trans", hidden_dim=HIDDEN)
    hub.register_agent("agent_reason", hidden_dim=HIDDEN)

    if with_core:
        router.recursive_core = RecursiveLatentCore(hub_dim=HUB, max_steps=4, tau_exit=0.99)
    if with_probe:
        probe = QueryReconstructionProbe(
            hub_dim=HUB, query_dim=QUERY_EMBED_DIM, tau_drift=tau_drift
        )
        if fit_probe:
            z = torch.randn(64, HUB)
            q = torch.stack([encode_query_bow(f"query number {i}") for i in range(64)])
            probe.fit_decoder(z, q, n_epochs=5)
        router.drift_probe = probe
    return router, hub


def _plan(task_id="t0"):
    return RoutingPlan(
        task_id=task_id,
        selected_agents=["agent_trans", "agent_reason"],
        execution_order=["agent_trans", "agent_reason"],
        estimated_cost=3.0,
    )


def _task():
    return AgentTask(
        task_id="t0", query="translate this sentence", target_language="th",
        latent_state=torch.randn(1, 3, HIDDEN),
    )


def test_recursive_core_runs_in_execution_path_and_logs_steps():
    router, hub = _setup(with_core=True)
    result = router.execute(_task(), _plan(), hub)

    assert router.recursive_core.total_calls >= 1
    assert router.recursive_core.mean_steps > 0
    # Per-transfer step count lands in response metadata.
    metas = [r.metadata for r in result.agent_responses]
    assert any("n_recursive_steps" in m for m in metas)


def test_zero_init_core_is_identity():
    core = RecursiveLatentCore(hub_dim=HUB, max_steps=4)
    z = torch.randn(2, HUB)
    out = core(z)
    assert torch.allclose(out, z)
    assert core.last_n_steps >= 1


def test_drift_probe_gates_and_repairs_in_execution_path():
    # tau_drift=-1 forces EVERY transfer to be flagged as drifted, so the
    # repair hop must run exactly once and the flag must reach metadata.
    router, hub = _setup(with_probe=True, tau_drift=-1.0)
    result = router.execute(_task(), _plan(), hub)

    metas = [r.metadata for r in result.agent_responses]
    drift_metas = [m for m in metas if "drift_score" in m]
    assert drift_metas, "drift scores must be recorded per transfer"
    for m in drift_metas:
        # Repair happened (score re-measured) and, with an impossible
        # threshold, is flagged as NOT repaired — but execution continued.
        assert "drift_score_after_repair" in m
        assert m["drift_repaired"] is False
    assert result.final_output  # the chain still completed


def test_untrained_probe_refuses_to_gate():
    import pytest
    router, hub = _setup(with_probe=True, fit_probe=False)
    with pytest.raises(RuntimeError, match="untrained"):
        router.execute(_task(), _plan(), hub)


def test_clean_transfer_without_modules_is_unchanged():
    router, hub = _setup()
    result = router.execute(_task(), _plan(), hub)
    metas = [r.metadata for r in result.agent_responses]
    assert all("drift_score" not in m and "n_recursive_steps" not in m for m in metas)
    assert result.final_output == "answer from agent_reason"
