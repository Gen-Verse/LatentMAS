"""Regression tests for dev_doc.md §9 gap 2: Module D's Geo_L conditioning and
topology-honoring routing.

Before this fix, CVAETopologyPrior conditioned on Q only and
AdaptiveOrchestrator.route(topology=...) accepted-and-ignored its topology
argument, so sampled topologies never influenced execution.
"""

import json

import pytest
import torch

from latent_coordination.agents.base_agent import (
    AgentConfig,
    AgentResponse,
    AgentTask,
    BaseAgent,
)
from latent_coordination.orchestration.router import (
    TOPOLOGY_ROLE_INDEX,
    AdaptiveOrchestrator,
)
from latent_coordination.topology.cvae_prior import CVAETopologyPrior, TrainingConfig
from latent_coordination.topology.geo_profile import GeoProfile

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"


class _StubAgent(BaseAgent):
    def process(self, task: AgentTask) -> AgentResponse:
        return AgentResponse(
            task_id=task.task_id, agent_id=self.agent_id,
            output_text=f"from {self.agent_id}",
        )


def _router():
    router = AdaptiveOrchestrator(device="cpu")
    for aid, role in [
        ("agent_trans", "translation"),
        ("agent_reason", "reasoning"),
        ("agent_safety", "safety"),
    ]:
        router.register_agent(_StubAgent(AgentConfig(
            agent_id=aid, model_id="stub", role=role, device="cpu", hidden_dim=8,
        )))
    return router


def _task(q="solve this math problem", lang="th"):
    return AgentTask(task_id="t0", query=q, target_language=lang)


# ---------------------------------------------------------------------------
# GeoProfile artifact loader
# ---------------------------------------------------------------------------

def _write_artifact(tmp_path, profiles, feature_names=None):
    payload = {"profiles": profiles}
    if feature_names is not None:
        payload["feature_names"] = feature_names
    p = tmp_path / "geo_profiles.json"
    p.write_text(json.dumps(payload))
    return p


def test_geo_profile_loads_and_looks_up(tmp_path):
    p = _write_artifact(
        tmp_path,
        {"th": [0.4, -0.1, 0.9], "lo": [0.6, 0.0, 1.1]},
        feature_names=["cka", "clap", "norm_ratio"],
    )
    gp = GeoProfile(p)
    assert gp.geo_dim == 3
    assert "th" in gp and "lo" in gp
    assert torch.allclose(gp.vector("th"), torch.tensor([0.4, -0.1, 0.9]))
    assert gp.batch(["th", "lo"]).shape == (2, 3)


def test_geo_profile_missing_artifact_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Geo_L artifact not found"):
        GeoProfile(tmp_path / "nope.json")


def test_geo_profile_unknown_language_raises(tmp_path):
    gp = GeoProfile(_write_artifact(tmp_path, {"th": [0.1, 0.2, 0.3]}))
    with pytest.raises(KeyError, match="No Geo_L profile"):
        gp.vector("xx")


def test_geo_profile_rejects_raw_wide_vectors(tmp_path):
    # 65-dim raw concatenation is exactly what the audit rejected.
    p = _write_artifact(tmp_path, {"th": [0.0] * 65})
    with pytest.raises(ValueError, match="compressed summary"):
        GeoProfile(p)
    assert GeoProfile(p, strict_dim=False).geo_dim == 65  # explicit ablation path


# ---------------------------------------------------------------------------
# CVAE geometry conditioning
# ---------------------------------------------------------------------------

def _tokens(B=4, L=16, vocab=100):
    return torch.randint(1, vocab, (B, L))


def test_cvae_geo_conditioning_roundtrip():
    cfg = TrainingConfig(z_dim=8, query_dim=16, geo_dim=3, max_n_agents=3,
                         query_vocab_size=100)
    prior = CVAETopologyPrior(cfg)
    G = (torch.rand(4, 3, 3) > 0.5).float()
    Q = _tokens()
    geo = torch.randn(4, 3)

    recon, mu, logvar = prior(G, Q, geo=geo)
    assert recon.shape == (4, 3, 3)
    loss, comps = prior.compute_loss(recon, G, mu, logvar)
    assert torch.isfinite(loss)

    adj = prior.sample_topology(Q[:1], n_samples=2, geo=geo[:1])
    assert adj.shape == (2, 3, 3)


def test_cvae_geo_dim_zero_rejects_geo_and_vice_versa():
    plain = CVAETopologyPrior(TrainingConfig(z_dim=8, query_dim=16, max_n_agents=3,
                                             query_vocab_size=100))
    G, Q = (torch.rand(2, 3, 3) > 0.5).float(), _tokens(B=2)
    with pytest.raises(ValueError, match="geo_dim=0"):
        plain(G, Q, geo=torch.randn(2, 3))

    geo_model = CVAETopologyPrior(TrainingConfig(z_dim=8, query_dim=16, geo_dim=3,
                                                 max_n_agents=3, query_vocab_size=100))
    with pytest.raises(ValueError, match="none was passed"):
        geo_model(G, Q)


def test_cvae_geometry_changes_sampled_topology_distribution():
    # A trained-enough prior must produce geometry-dependent decode outputs:
    # check the decoder's output actually depends on geo (gradient flows).
    cfg = TrainingConfig(z_dim=8, query_dim=16, geo_dim=3, max_n_agents=3,
                         query_vocab_size=100)
    prior = CVAETopologyPrior(cfg)
    Q = _tokens(B=1)
    z = torch.randn(1, cfg.z_dim)
    geo = torch.randn(1, 3, requires_grad=True)
    out = prior.decode(z, Q, geo=geo)
    out.sum().backward()
    assert geo.grad is not None and geo.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# route(topology=...) honored
# ---------------------------------------------------------------------------

def test_route_honors_explicit_topology_selection_and_order():
    router = _router()
    n = len(TOPOLOGY_ROLE_INDEX)
    adj = torch.zeros(n, n)
    # reasoning -> translation only; safety not in the graph.
    adj[TOPOLOGY_ROLE_INDEX["reasoning"], TOPOLOGY_ROLE_INDEX["translation"]] = 1.0

    plan = router.route(_task(), topology=adj)
    assert plan.selected_agents == ["agent_reason", "agent_trans"]
    assert plan.execution_order == ["agent_reason", "agent_trans"]


def test_route_topology_cycle_falls_back_to_canonical_order():
    router = _router()
    n = len(TOPOLOGY_ROLE_INDEX)
    adj = torch.zeros(n, n)
    i, j = TOPOLOGY_ROLE_INDEX["translation"], TOPOLOGY_ROLE_INDEX["reasoning"]
    adj[i, j] = 1.0
    adj[j, i] = 1.0  # cycle

    plan = router.route(_task(), topology=adj)
    # Canonical role order: translation before reasoning.
    assert plan.execution_order == ["agent_trans", "agent_reason"]


def test_route_empty_topology_falls_back_to_standard_routing():
    router = _router()
    adj = torch.zeros(3, 3)
    plan = router.route(_task(), topology=adj)
    assert plan.selected_agents  # standard router produced something


def test_cvae_router_type_samples_topology_from_prior():
    cfg = TrainingConfig(z_dim=8, query_dim=16, max_n_agents=3, query_vocab_size=100)
    prior = CVAETopologyPrior(cfg)
    router = _router()
    router.router_type = "cvae"
    router.topology_prior = prior
    router.topology_query_encoder = lambda text: torch.randint(1, 100, (16,))

    plan = router.route(_task())
    assert plan.selected_agents  # either topology-driven or fell back — but routed


def test_cvae_router_without_prior_raises():
    router = _router()
    router.router_type = "cvae"
    with pytest.raises(RuntimeError, match="trained topology prior"):
        router.route(_task())
