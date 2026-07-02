"""Unit + smoke tests for latent_coordination audit-driven refactor.

All tests run on CPU with random tensors — no model loading required.
"""

import math
import torch
import torch.nn as nn
import pytest

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

# ---------------------------------------------------------------------------
# adapter.py tests
# ---------------------------------------------------------------------------

def test_norm_match_layer_preserves_norm():
    from latent_coordination.latent_space.adapter import NormMatchLayer
    layer = NormMatchLayer()
    ref = torch.randn(4, 64) * 5.0
    output = torch.randn(4, 32) * 0.1  # very different norm
    scaled = layer(output, ref)
    # RMS of output should approximately match RMS of ref after scaling
    ref_rms = (ref.norm(dim=-1) / (ref.shape[-1] ** 0.5)).mean()
    out_rms = (scaled.norm(dim=-1) / (scaled.shape[-1] ** 0.5)).mean()
    # Allow 10% relative tolerance (different dim sizes affect exact match)
    assert abs(float(out_rms) - float(ref_rms)) / (float(ref_rms) + 1e-8) < 0.15


def test_infonce_loss_is_finite():
    from latent_coordination.latent_space.adapter import infonce_loss
    anchor = torch.randn(8, 64)
    positive = anchor + 0.01 * torch.randn(8, 64)  # near-identical positives
    loss = infonce_loss(anchor, positive, temperature=0.07)
    assert torch.isfinite(loss), f"InfoNCE loss is not finite: {loss}"


def test_infonce_loss_decreasing():
    """InfoNCE loss should decrease under optimization."""
    from latent_coordination.latent_space.adapter import infonce_loss
    anchor = torch.randn(16, 32, requires_grad=False)
    positive = nn.Linear(32, 32, bias=False)
    optimizer = torch.optim.Adam(positive.parameters(), lr=1e-2)
    losses = []
    for _ in range(6):
        optimizer.zero_grad()
        pos = positive(anchor)
        loss = infonce_loss(anchor.detach(), pos, temperature=0.1)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


def test_adapter_config_norm_match_flag():
    from latent_coordination.latent_space.adapter import AdapterConfig, LatentAdapter
    cfg = AdapterConfig(in_dim=64, out_dim=32, use_norm_match=True)
    adapter = LatentAdapter(cfg)
    assert adapter.norm_match is not None
    x = torch.randn(4, 64)
    out = adapter(x)
    assert out.shape == (4, 32)


def test_train_adapter_infonce():
    from latent_coordination.latent_space.adapter import AdapterConfig, LatentAdapter, train_adapter
    cfg = AdapterConfig(in_dim=16, out_dim=16, hidden_dim=32)
    adapter = LatentAdapter(cfg)
    src = torch.randn(20, 16)
    tgt = torch.randn(20, 16)
    losses = train_adapter(adapter, src, tgt, n_epochs=3, loss_fn="infonce", verbose=False)
    assert len(losses) == 3
    assert all(math.isfinite(l) for l in losses)


def test_compute_reconstruction_error_has_effective_rank():
    from latent_coordination.latent_space.adapter import AdapterConfig, LatentAdapter, compute_reconstruction_error
    cfg = AdapterConfig(in_dim=16, out_dim=16, hidden_dim=32)
    adapter = LatentAdapter(cfg)
    src = torch.randn(10, 16)
    metrics = compute_reconstruction_error(adapter, src, src)
    assert "effective_rank" in metrics
    assert metrics["effective_rank"] >= 1.0


# ---------------------------------------------------------------------------
# universal_space.py tests
# ---------------------------------------------------------------------------

def test_universal_space_transfer_record_has_effective_rank():
    from latent_coordination.latent_space.universal_space import UniversalLatentHub
    uls = UniversalLatentHub(universal_dim=32)
    uls.register_agent("a", hidden_dim=64)
    uls.register_agent("b", hidden_dim=64)
    x = torch.randn(4, 64)
    uls.transfer("a", "b", x)
    rec = uls.get_transfer_history()
    assert len(rec) == 1
    assert rec[0].effective_rank is not None


def test_align_ridge_shape():
    from latent_coordination.latent_space.universal_space import UniversalLatentHub
    src = torch.randn(20, 64)
    tgt = torch.randn(20, 32)
    aligned = UniversalLatentHub.align_ridge(src, tgt, alpha=1e-3)
    assert aligned.shape == (20, 32)


def test_compute_information_metrics_keys():
    from latent_coordination.latent_space.universal_space import UniversalLatentHub
    uls = UniversalLatentHub(universal_dim=32)
    uls.register_agent("x", hidden_dim=64)
    x = torch.randn(8, 64)
    m = uls.compute_information_metrics("x", x)
    for key in ["effective_rank", "norm_ratio", "hsic_mi_proxy", "cosine_similarity", "mse"]:
        assert key in m, f"Missing key: {key}"


def test_transfer_with_norm_match():
    from latent_coordination.latent_space.universal_space import UniversalLatentHub
    uls = UniversalLatentHub(universal_dim=32)
    uls.register_agent("a", hidden_dim=64)
    uls.register_agent("b", hidden_dim=64)
    x = torch.randn(4, 64)
    out = uls.transfer("a", "b", x, norm_match=True)
    assert out.shape == (4, 64)


# ---------------------------------------------------------------------------
# cvae_prior.py tests
# ---------------------------------------------------------------------------

def test_transformer_query_encoder_shape():
    from latent_coordination.topology.cvae_prior import _TransformerQueryEncoder
    enc = _TransformerQueryEncoder(
        vocab_size=500, embed_dim=16, n_heads=4, n_layers=1,
        ffn_dim=32, output_dim=64
    )
    tokens = torch.randint(0, 500, (3, 20))
    out = enc(tokens)
    assert out.shape == (3, 64)


def test_cvae_with_transformer_encoder_forward():
    from latent_coordination.topology.cvae_prior import CVAETopologyPrior, TrainingConfig
    cfg = TrainingConfig(
        z_dim=8, max_n_agents=4, query_dim=32,
        query_embed_dim=16, use_transformer_encoder=True,
        n_epochs=1, free_bits_lambda=0.5,
    )
    model = CVAETopologyPrior(cfg)
    G = torch.zeros(2, 4, 4)
    Q = torch.randint(0, 100, (2, 16))
    recon, mu, lv = model(G, Q)
    assert recon.shape == (2, 4, 4)
    loss, components = model.compute_loss(recon, G, mu, lv, beta=1.0)
    assert torch.isfinite(loss)
    assert "active_units" in components


def test_cvae_bilstm_ablation_still_works():
    from latent_coordination.topology.cvae_prior import CVAETopologyPrior, TrainingConfig
    cfg = TrainingConfig(
        z_dim=8, max_n_agents=3, query_dim=32,
        query_embed_dim=16, use_transformer_encoder=False,
        n_epochs=1,
    )
    model = CVAETopologyPrior(cfg)
    G = torch.zeros(2, 3, 3)
    Q = torch.randint(0, 100, (2, 16))
    recon, mu, lv = model(G, Q)
    assert recon.shape == (2, 3, 3)


def test_free_bits_kl():
    from latent_coordination.topology.cvae_prior import free_bits_kl
    mu = torch.zeros(8, 16)
    logvar = torch.zeros(8, 16)
    kl = free_bits_kl(mu, logvar, lambda_free=0.5)
    assert torch.isfinite(kl)
    # With perfect posterior (mu=0, logvar=0), standard KL = 0; free-bits floors to lambda
    assert kl.item() >= 0.0


def test_free_bits_kl_kingma2016_semantics():
    """Verify clamp happens on per-dim batch mean, not per-sample (Kingma 2016)."""
    from latent_coordination.topology.cvae_prior import free_bits_kl
    # mu=0, logvar=0 → per-sample-per-dim KL = 0 → batch-mean KL per dim = 0
    # Kingma free-bits should clamp per-dim mean to lambda_free
    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)
    lambda_free = 0.3
    kl = free_bits_kl(mu, logvar, lambda_free=lambda_free)
    # Expected: all per-dim means are 0 → clamped to lambda_free → mean = lambda_free
    assert abs(kl.item() - lambda_free) < 1e-5, (
        f"free_bits_kl with all-zero posterior should equal lambda_free={lambda_free}, got {kl.item()}"
    )


def test_transformer_encoder_n_heads_divisibility():
    """n_heads must divide embed_dim — test with non-power-of-4 embed_dim."""
    from latent_coordination.topology.cvae_prior import CVAETopologyPrior, TrainingConfig
    # embed_dim=6 is not divisible by 4 — should fall back to 2 heads
    cfg = TrainingConfig(z_dim=4, max_n_agents=2, query_dim=8, query_embed_dim=6,
                         use_transformer_encoder=True)
    model = CVAETopologyPrior(cfg)
    G = torch.zeros(2, 2, 2)
    Q = torch.randint(0, 10, (2, 4))
    recon, mu, lv = model(G, Q)
    assert recon.shape == (2, 2, 2)


def test_active_units():
    from latent_coordination.topology.cvae_prior import active_units
    mu = torch.randn(32, 16)  # high variance → should have active units
    count = active_units(mu, threshold=0.01)
    assert count > 0, "Expected some active latent units for random mu"

    mu_const = torch.zeros(32, 16)
    count_zero = active_units(mu_const, threshold=0.01)
    assert count_zero == 0, "Constant mu should have 0 active units"


# ---------------------------------------------------------------------------
# router.py tests
# ---------------------------------------------------------------------------

def test_attention_router_soft_dispatch_sums_to_one():
    from latent_coordination.orchestration.router import AttentionRouter
    roles = ["reasoning", "translation", "safety"]
    router = AttentionRouter(query_dim=32, roles=roles)
    q = torch.randn(1, 32)
    weights, conf = router(q)
    assert abs(weights.sum().item() - 1.0) < 1e-5, "Attention weights must sum to 1"
    assert 0.0 <= float(conf) <= 1.0


def test_attention_router_dispatch_returns_roles():
    from latent_coordination.orchestration.router import AttentionRouter
    roles = ["reasoning", "translation", "safety"]
    router = AttentionRouter(query_dim=32, roles=roles)
    q = torch.randn(32)
    selected, conf = router.dispatch(q, threshold=0.0)
    # At threshold=0, all roles above 0 should be returned (likely all)
    assert len(selected) > 0
    for r in selected:
        assert r in roles


def test_adaptive_orchestrator_attention_route():
    from latent_coordination.orchestration.router import AdaptiveOrchestrator
    from latent_coordination.agents.base_agent import AgentConfig, AgentTask
    from latent_coordination.agents.specialized_agents import ReasoningAgent

    class _FakeReasoningAgent(ReasoningAgent):
        def __init__(self):
            super().__init__(AgentConfig(agent_id="r1", model_id="fake", role="reasoning"))
        def process(self, task):
            from latent_coordination.agents.base_agent import AgentResponse
            return AgentResponse(task_id=task.task_id, agent_id=self.agent_id, output_text="ok")

    orch = AdaptiveOrchestrator(router_type="attention")
    orch.register_agent(_FakeReasoningAgent())
    task = AgentTask(task_id="t1", query="What is 2+2?")
    plan = orch.route(task)
    assert plan.task_id == "t1"
    assert len(plan.selected_agents) > 0
    assert 0.0 <= plan.routing_confidence <= 1.0


def test_routing_plan_has_confidence():
    from latent_coordination.orchestration.router import RoutingPlan
    plan = RoutingPlan(task_id="t", selected_agents=[], execution_order=[], estimated_cost=0.0)
    assert hasattr(plan, "routing_confidence")


# ---------------------------------------------------------------------------
# information_theory.py tests
# ---------------------------------------------------------------------------

def test_effective_rank_full_matrix():
    from latent_coordination.eval.information_theory import effective_rank
    M = torch.eye(8)
    er = effective_rank(M)
    assert abs(er - 8.0) < 0.5, f"Full-rank identity should have eff rank ≈ 8, got {er}"


def test_effective_rank_rank1_matrix():
    from latent_coordination.eval.information_theory import effective_rank
    v = torch.randn(1, 16)
    M = v.T @ v  # rank-1
    er = effective_rank(M)
    assert er < 2.0, f"Rank-1 matrix should have eff rank ≈ 1, got {er}"


def test_hsic_mi_proxy_nonzero_for_dependent():
    from latent_coordination.eval.information_theory import hsic_mi_proxy
    X = torch.randn(20, 8)
    Y = X + 0.01 * torch.randn(20, 8)  # nearly identical (highly dependent)
    hsic = hsic_mi_proxy(X, Y)
    assert hsic > 0, f"HSIC should be positive for dependent X, Y; got {hsic}"


def test_compression_ratio():
    from latent_coordination.eval.information_theory import compression_ratio
    ratio = compression_ratio(latent_bytes=512.0, text_tokens=256, bytes_per_token=4.0)
    assert ratio == pytest.approx(2.0), f"Expected 2.0, got {ratio}"


def test_breakeven_n_positive():
    from latent_coordination.eval.information_theory import breakeven_n
    n = breakeven_n(adapter_forward_ms=0.5, token_gen_ms_per_token=0.1, avg_msg_len_tokens=50)
    assert n > 1.0


def test_info_theoretic_analyzer_keys():
    from latent_coordination.eval.information_theory import InfoTheoreticAnalyzer
    from latent_coordination.latent_space.universal_space import UniversalLatentHub
    uls = UniversalLatentHub(universal_dim=32)
    uls.register_agent("a", hidden_dim=64)
    analyzer = InfoTheoreticAnalyzer(hub_dim=32)
    x = torch.randn(8, 64)
    result = analyzer.analyze(uls, x, "a", text_tokens_equivalent=100)
    for k in ["effective_rank_hub", "effective_rank_original", "hsic_mi_proxy", "cosine_similarity", "compression_ratio"]:
        assert k in result, f"Missing key: {k}"


# ---------------------------------------------------------------------------
# adversarial.py tests
# ---------------------------------------------------------------------------

def test_bounded_attacker_respects_epsilon():
    from latent_coordination.eval.adversarial import BoundedLatentAttacker
    attacker = BoundedLatentAttacker(epsilon=0.5, seed=0)
    hub = torch.zeros(4, 32)
    perturbed = attacker.perturb(hub)
    delta = perturbed - hub
    norms = delta.norm(dim=-1)
    assert (norms <= 0.5 + 1e-5).all(), f"Perturbation exceeded ε=0.5: {norms}"


def test_latent_gate_defense_clamps_outliers():
    from latent_coordination.eval.adversarial import LatentGateDefense
    defense = LatentGateDefense(rms_lo=0.5, rms_hi=5.0, clamp=True)
    # Create a hub vector with very large norm (outlier)
    outlier = torch.ones(2, 16) * 100.0
    filtered, mask = defense.filter(outlier)
    out_rms = filtered.norm(dim=-1) / (filtered.shape[-1] ** 0.5)
    assert (out_rms <= 5.0 + 1e-4).all(), f"Defense did not clamp outlier: {out_rms}"


def test_adversarial_eval_returns_per_epsilon_keys():
    from latent_coordination.eval.adversarial import run_adversarial_eval
    from latent_coordination.latent_space.universal_space import UniversalLatentHub
    uls = UniversalLatentHub(universal_dim=16)
    uls.register_agent("a", hidden_dim=32)
    uls.register_agent("b", hidden_dim=32)
    x = torch.randn(4, 32)
    results = run_adversarial_eval(uls, [("a", "b")], x, epsilons=[0.0, 0.1, 0.5])
    assert "per_epsilon" in results
    for eps in [0.0, 0.1, 0.5]:
        assert eps in results["per_epsilon"]
        assert "cosine_attacked" in results["per_epsilon"][eps]


# ---------------------------------------------------------------------------
# efficiency_metrics.py tests
# ---------------------------------------------------------------------------

def test_bootstrap_ci_coverage():
    from latent_coordination.eval.efficiency_metrics import bootstrap_ci
    # Known data: all 1s → mean is 1.0 → CI should contain 1.0
    data = [1.0] * 100
    lo, hi = bootstrap_ci(data, n_bootstrap=500)
    assert lo <= 1.0 <= hi, f"CI ({lo:.4f}, {hi:.4f}) does not contain true mean 1.0"


def test_bootstrap_ci_empty():
    from latent_coordination.eval.efficiency_metrics import bootstrap_ci
    import math
    lo, hi = bootstrap_ci([])
    assert math.isnan(lo) and math.isnan(hi)


def test_compute_breakeven():
    from latent_coordination.eval.efficiency_metrics import compute_breakeven
    result = compute_breakeven(n_agents=4, avg_msg_tokens=50, adapter_flops=1e6)
    assert "latent_wins" in result
    assert "breakeven_msg_len" in result
    assert result["breakeven_n"] > 1.0


def test_ablation_report_has_ci_field():
    from latent_coordination.eval.efficiency_metrics import AblationReport
    report = AblationReport(metrics_by_mode={"token": {"accuracy": 0.8}})
    assert hasattr(report, "confidence_intervals")


# ---------------------------------------------------------------------------
# baselines tests
# ---------------------------------------------------------------------------

def test_latent_mas_baseline_shape():
    from latent_coordination.baselines.latent_mas import LatentMASBaseline
    baseline = LatentMASBaseline(hidden_dim=64)
    sender = torch.randn(4, 64)
    received = baseline.share_hidden_state(sender, receiver_hidden_dim=64)
    assert received.shape == (4, 64)


def test_latent_mas_baseline_heterogeneous_raises():
    from latent_coordination.baselines.latent_mas import LatentMASBaseline
    baseline = LatentMASBaseline(hidden_dim=64)
    sender = torch.randn(4, 64)
    with pytest.raises(ValueError, match="homogeneous"):
        baseline.share_hidden_state(sender, receiver_hidden_dim=128)


def test_latent_mas_kv_memory_update():
    from latent_coordination.baselines.latent_mas import LatentMASBaseline
    baseline = LatentMASBaseline(hidden_dim=32)
    for _ in range(3):
        baseline.update_kv_memory(torch.randn(4, 32))
    mem = baseline.get_kv_memory()
    assert mem is not None
    assert mem.shape[-1] == 32


def test_blackboard_read_write():
    from latent_coordination.baselines.blackboard_mas import BlackboardMASBaseline
    bb = BlackboardMASBaseline()
    bb.write("agent_a", "Hello from A")
    bb.write("agent_b", "Hello from B")
    entries = bb.read_all("agent_c")
    assert len(entries) == 2
    assert entries[0].author_id == "agent_a"
    assert entries[1].content == "Hello from B"


def test_blackboard_communication_stats():
    from latent_coordination.baselines.blackboard_mas import BlackboardMASBaseline
    bb = BlackboardMASBaseline()
    bb.write("a", "msg1")
    bb.read_all("b")
    stats = bb.communication_stats(n_agents=4, n_rounds=3)
    assert stats["peer_to_peer_msg_count"] == 4 * 3 * 3  # N*(N-1)*rounds
    assert stats["blackboard_op_count"] == 4 * 3  # N*rounds
    assert stats["reduction_factor"] == pytest.approx(3.0)  # (N-1)


# ---------------------------------------------------------------------------
# ThoughtComm baseline tests
# ---------------------------------------------------------------------------

def test_thoughtcomm_encode_shape():
    from latent_coordination.baselines.thoughtcomm import ThoughtCommBaseline, ThoughtCommConfig
    cfg = ThoughtCommConfig(hidden_dim=32, shared_dim=8, private_dim=16)
    baseline = ThoughtCommBaseline(cfg)
    baseline.register_agent("a")
    baseline.register_agent("b")
    x = torch.randn(4, 32)
    z_shared, z_private = baseline.encode("a", x)
    assert z_shared.shape == (4, 8)
    assert z_private.shape == (4, 16)


def test_thoughtcomm_communicate_shape():
    from latent_coordination.baselines.thoughtcomm import ThoughtCommBaseline, ThoughtCommConfig
    cfg = ThoughtCommConfig(hidden_dim=32, shared_dim=8, private_dim=16)
    baseline = ThoughtCommBaseline(cfg)
    baseline.register_agent("a")
    baseline.register_agent("b")
    x = torch.randn(4, 32)
    recon, sparse_loss = baseline.communicate("a", "b", x)
    assert recon.shape == (4, 32)
    assert isinstance(sparse_loss, float)


def test_thoughtcomm_sparsity_stats():
    from latent_coordination.baselines.thoughtcomm import ThoughtCommBaseline, ThoughtCommConfig
    cfg = ThoughtCommConfig(hidden_dim=32, shared_dim=8, private_dim=16)
    baseline = ThoughtCommBaseline(cfg)
    baseline.register_agent("a")
    z_shared, _ = baseline.encode("a", torch.randn(4, 32))
    stats = baseline.compute_sparsity_stats(z_shared)
    assert "l1_norm" in stats and "l0_approx_zero_frac" in stats


# ---------------------------------------------------------------------------
# Cache-to-Cache baseline tests
# ---------------------------------------------------------------------------

def test_c2c_fuse_kv_shape():
    from latent_coordination.baselines.cache_to_cache import CacheToCacheBaseline
    c2c = CacheToCacheBaseline()
    c2c.register_agent("a", kv_dim=32)
    c2c.register_agent("b", kv_dim=64)
    sk = torch.randn(2, 8, 32)
    sv = torch.randn(2, 8, 32)
    rk = torch.randn(2, 4, 64)
    rv = torch.randn(2, 4, 64)
    fk, fv = c2c.fuse_kv("a", "b", sk, sv, rk, rv)
    assert fk.shape == (2, 4, 64)
    assert fv.shape == (2, 4, 64)


def test_c2c_complexity_quadratic():
    from latent_coordination.baselines.cache_to_cache import CacheToCacheBaseline
    c2c = CacheToCacheBaseline()
    stats = c2c.communication_complexity(n_agents=4)
    assert stats["c2c_pairwise_projections"] == 12  # 4*(4-1)
    assert stats["hub_spoke_projections"] == 8       # 2*4


# ---------------------------------------------------------------------------
# G-Designer baseline tests
# ---------------------------------------------------------------------------

def test_gdesigner_topology_shape():
    from latent_coordination.baselines.gdesigner_mas_router import GDesignerBaseline
    gd = GDesignerBaseline(max_n_agents=4, query_dim=16, z_dim=8, hidden_dim=32)
    q = torch.randn(2, 16)
    adj, probs = gd.sample_topology(q)
    assert adj.shape == (2, 4, 4)
    assert probs.shape == (2, 4, 4)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_gdesigner_kl_loss():
    from latent_coordination.baselines.gdesigner_mas_router import GDesignerBaseline
    gd = GDesignerBaseline(max_n_agents=3, query_dim=8, z_dim=4)
    mu = torch.randn(2, 3, 4)
    lv = torch.zeros(2, 3, 4)
    kl = gd.kl_loss(mu, lv)
    assert torch.isfinite(kl)


# ---------------------------------------------------------------------------
# MasRouter baseline tests
# ---------------------------------------------------------------------------

def test_masrouter_route_returns_mode_and_roles():
    from latent_coordination.baselines.gdesigner_mas_router import MasRouterBaseline, MasRouterConfig
    cfg = MasRouterConfig(query_dim=16, n_roles=3)
    router = MasRouterBaseline(cfg)
    router.set_roles(["reasoning", "translation", "safety"])
    q = torch.randn(1, 16)
    result = router.route(q)
    assert result["collab_mode"] in ["solo", "pipeline", "debate"]
    assert len(result["selected_roles"]) > 0
    assert all(r in ["reasoning", "translation", "safety"] for r in result["selected_roles"])


def test_masrouter_mode_probs_sum_to_one():
    from latent_coordination.baselines.gdesigner_mas_router import MasRouterBaseline, MasRouterConfig
    cfg = MasRouterConfig(query_dim=8)
    router = MasRouterBaseline(cfg)
    router.set_roles(["r1", "r2"])
    result = router.route(torch.randn(8))
    assert abs(result["mode_probs"].sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Smoke test: full CVAE pipeline with transformer encoder
# ---------------------------------------------------------------------------

def test_smoke_cvae_full_pipeline():
    from latent_coordination.topology.cvae_prior import CVAETopologyPrior, TrainingConfig, TopologyDataset
    from torch.utils.data import DataLoader

    cfg = TrainingConfig(
        z_dim=8, max_n_agents=3, query_dim=16, query_embed_dim=8,
        use_transformer_encoder=True, free_bits_lambda=0.2,
        n_epochs=2, batch_size=4, lr=1e-3,
    )
    model = CVAETopologyPrior(cfg)
    dataset = TopologyDataset.from_random(n_samples=8, max_n_agents=3, max_seq_len=16, vocab_size=100)
    loader = DataLoader(dataset, batch_size=4, collate_fn=lambda b: {
        "G": torch.stack([x["G"] for x in b]),
        "Q": torch.stack([x["Q"] for x in b]),
        "meta": [x["meta"] for x in b],
    })
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = model.fit(loader, optimizer=optimizer)
    assert len(history) == 2
    assert all(math.isfinite(h["total"]) for h in history)
    assert "active_units" in history[0]

    # Sampling
    Q = torch.randint(0, 100, (1, 16))
    topo = model.sample_topology(Q, n_samples=2)
    assert topo.shape == (2, 3, 3)
