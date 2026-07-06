"""Regression tests for the 2026-07-03 strategy-gap closure session.

Pins the fixes that closed the remaining dev_doc.md §9 gap-6 items and the
outstanding strategy.md Phase-4 requirements:

1. ``checkpointing.checkpoint_dir`` is honored (was hardcoded to
   ``{output_dir}/checkpoints``).
2. ``orchestration.parallel_agents=true`` fails loudly — the latent chain is
   sequential by design; the knob promised concurrency that never existed.
3. ``orchestration.timeout_per_agent_s`` reaches ``model.generate(max_time=…)``
   via ``AgentConfig.max_time_s``.
4. The staircase ablation runner (strategy.md §7.3) derives per-row configs
   from real module toggles, in isolated output/checkpoint dirs.
5. The OneFlow narrative-gating conditional (strategy.md §7.2) pivots the
   headline framing automatically when the single-agent baseline wins — the
   Phase-4 gate explicitly requires this synthetic-trigger test.
6. The drift probe's shallow-MLP variant (strategy.md §4.4, ablation 7e).
"""

import json

import pytest
import torch

from latent_coordination.agents.base_agent import AgentConfig
from latent_coordination.eval.ablation_staircase import (
    STAIRCASE_ROWS,
    derive_row_config,
    load_extra_rows,
    run_staircase,
    select_rows,
    set_dotted,
)
from latent_coordination.eval.verification_probe import QueryReconstructionProbe
from latent_coordination.pipeline.coordination_pipeline import (
    CoordinationPipeline,
    derive_headline_framing,
)

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"


def _pipeline_cfg(tmp_path, **extra):
    cfg = {
        "project": {"output_dir": str(tmp_path / "results"), "seed": 7},
        "agents": [{"model_id": "stub-model", "device": "cpu", "role": "orchestrator"}],
    }
    cfg.update(extra)
    return cfg


# ---------------------------------------------------------------------------
# 1. checkpointing.checkpoint_dir
# ---------------------------------------------------------------------------

def test_checkpoint_dir_config_is_honored(tmp_path):
    ckpt_dir = tmp_path / "custom_ckpts"
    pipeline = CoordinationPipeline(
        _pipeline_cfg(tmp_path, checkpointing={"checkpoint_dir": str(ckpt_dir)})
    )
    assert pipeline.checkpoint_manager.root == ckpt_dir / "coordination"


def test_checkpoint_dir_defaults_to_output_dir(tmp_path):
    pipeline = CoordinationPipeline(_pipeline_cfg(tmp_path))
    assert pipeline.checkpoint_manager.root == (
        tmp_path / "results" / "checkpoints" / "coordination"
    )


# ---------------------------------------------------------------------------
# 2. orchestration.parallel_agents fails loudly
# ---------------------------------------------------------------------------

def test_parallel_agents_true_fails_loudly(tmp_path):
    pipeline = CoordinationPipeline(
        _pipeline_cfg(tmp_path, orchestration={"parallel_agents": True})
    )
    with pytest.raises(ValueError, match="parallel_agents"):
        pipeline._run_stage_a()


# ---------------------------------------------------------------------------
# 3. timeout_per_agent_s → generate(max_time=…)
# ---------------------------------------------------------------------------

def test_max_time_reaches_generate_kwargs():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from latent_coordination.agents.base_agent import AgentResponse, AgentTask, BaseAgent

    class _TrivialAgent(BaseAgent):
        def process(self, task: AgentTask) -> AgentResponse:
            raise NotImplementedError

    model_id = "Qwen/Qwen1.5-0.5B-Chat"
    agent = _TrivialAgent(AgentConfig(
        agent_id="a", model_id=model_id, role="reasoning", device="cpu",
        hidden_dim=1024, dtype="float32", max_time_s=2.5,
    ))
    agent._model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    agent._tokenizer = AutoTokenizer.from_pretrained(model_id)
    agent._is_loaded = True

    recorded = {}
    original_generate = agent._model.generate

    def _spy(*args, **kwargs):
        recorded.update(kwargs)
        return original_generate(*args, **kwargs)

    agent._model.generate = _spy
    text, _ = agent.generate_and_capture("Hello", max_new_tokens=2)
    assert isinstance(text, str)
    assert recorded.get("max_time") == 2.5

    recorded.clear()
    hidden = agent._model.config.hidden_size
    agent.inject_latent_and_generate(
        torch.zeros(1, 1, hidden), "Hello", max_new_tokens=2
    )
    assert recorded.get("max_time") == 2.5


def test_max_time_absent_when_unset():
    cfg = AgentConfig(agent_id="a", model_id="stub")
    assert cfg.max_time_s is None


# ---------------------------------------------------------------------------
# 4. Staircase ablation runner
# ---------------------------------------------------------------------------

def test_set_dotted_creates_and_rejects():
    cfg = {"a": {"b": 1}}
    set_dotted(cfg, "a.c.d", True)
    assert cfg["a"]["c"]["d"] is True
    with pytest.raises(TypeError):
        set_dotted(cfg, "a.b.c", 1)  # 'a.b' is an int, not a mapping


def test_staircase_rows_match_strategy_table():
    rows = {r.row_id: r for r in STAIRCASE_ROWS}
    # Row 0: proxy — no correctness benchmarks, no modules.
    assert rows["0"].overrides["benchmarks.mgsm.enabled"] is False
    assert rows["0"].overrides["universal_latent_space.adapter_training.enabled"] is False
    # Row 1 isolates the metric change only.
    assert rows["1"].overrides["benchmarks.mgsm.enabled"] is True
    assert rows["1"].overrides["universal_latent_space.adapter_training.enabled"] is False
    # Row 5 is the full closed-loop system: A+B, D, C, E all on.
    r5 = rows["5"].overrides
    assert r5["universal_latent_space.adapter_training.enabled"] is True
    assert r5["orchestration.routing_strategy"] == "cvae_topology"
    assert r5["cvae.condition_on_geometry"] is True
    assert r5["latent_reasoning.enabled"] is True
    assert r5["verification.enabled"] is True
    # Row 6 = row 5 minus E.
    assert rows["6"].overrides["verification.enabled"] is False
    assert rows["6"].overrides["latent_reasoning.enabled"] is True
    # 7b: the OneFlow single-agent baseline is a column of EVERY row.
    for r in STAIRCASE_ROWS:
        assert "single_agent_baseline" in r.overrides["communication.eval_modes"]
    # 7a splits zero individual loss terms.
    assert rows["7a_recon_only"].overrides[
        "universal_latent_space.adapter_training.mu_cka"] == 0.0
    assert rows["7a_recon_only"].overrides[
        "universal_latent_space.adapter_training.gamma_dae"] == 0.0


def test_derive_row_config_isolates_dirs_and_preserves_base(tmp_path):
    base = {"project": {"output_dir": "results/x"}, "cvae": {"latent_dim": 16}}
    row = STAIRCASE_ROWS[5]
    cfg = derive_row_config(base, row, tmp_path)
    # Base config untouched (deep copy).
    assert base["project"]["output_dir"] == "results/x"
    assert "universal_latent_space" not in base
    # Per-row isolation of results AND checkpoints (the Stage-E cache key does
    # not encode module toggles — sharing a checkpoint dir would silently
    # reuse another row's cached mode results).
    assert str(tmp_path) in cfg["project"]["output_dir"]
    assert str(tmp_path) in cfg["checkpointing"]["checkpoint_dir"]
    assert cfg["project"]["output_dir"] != cfg["checkpointing"]["checkpoint_dir"]
    assert cfg["cvae"]["latent_dim"] == 16
    assert cfg["cvae"]["condition_on_geometry"] is True


def test_select_rows_unknown_raises():
    with pytest.raises(ValueError, match="Unknown staircase row"):
        select_rows(["no_such_row"], [])
    subset = select_rows(["0", "closed_loop_full"], [])
    assert [r.row_id for r in subset] == ["0", "5"]


def test_extra_rows_validation():
    good = {"ablation": {"extra_rows": [
        {"name": "probe_mlp", "isolates": "7e",
         "overrides": {"verification.probe_arch": "mlp"}},
    ]}}
    rows = load_extra_rows(good)
    assert rows[0].name == "probe_mlp"
    with pytest.raises(ValueError, match="extra_rows"):
        load_extra_rows({"ablation": {"extra_rows": [{"name": "missing_overrides"}]}})


def test_staircase_dry_run_writes_consolidated_artifact(tmp_path):
    base = {"project": {"output_dir": "results/x"}}
    out = run_staircase(base, STAIRCASE_ROWS[:2], tmp_path, dry_run=True)
    artifact = json.load(open(out["artifact_path"], encoding="utf-8"))
    assert artifact["dry_run"] is True
    assert set(artifact["rows"]) == {"0", "1"}
    assert "derived_config" in artifact["rows"]["0"]
    assert "results_by_mode" not in artifact["rows"]["0"]


# ---------------------------------------------------------------------------
# 5. Narrative defensive gating (strategy.md §7.2 — Phase-4 gate test)
# ---------------------------------------------------------------------------

def _modes(baseline_acc, ours_acc):
    return {
        "single_agent_baseline": {"accuracy": baseline_acc, "token_cost": 0.0},
        "token_based_mas": {"accuracy": 0.5, "token_cost": 120.0},
        "latent_based_mas_ours": {"accuracy": ours_acc, "token_cost": 0.0},
    }


def test_framing_pivots_when_single_agent_baseline_wins():
    # The Phase-4 gate's synthetic case: baseline set to "win" must flip the
    # report framing automatically, not leave it a manual editorial decision.
    framing = derive_headline_framing(
        _modes(baseline_acc=0.8, ours_acc=0.6),
        ["modelA", "modelB", "modelB"],
    )
    assert framing["framing"] == "efficiency_fallback"
    assert framing["token_overhead_reduction_vs_token_mas"] == 1.0
    assert "heterogeneous_cross_architecture_regime" in framing["headline_claims"]
    assert framing["heterogeneous_pool"] is True
    # Ties count as a baseline win (>=), per the strategy's "matches or exceeds".
    assert derive_headline_framing(
        _modes(0.6, 0.6), ["m"])["framing"] == "efficiency_fallback"


def test_framing_stays_on_accuracy_when_ours_wins():
    framing = derive_headline_framing(_modes(0.5, 0.7), ["m", "m"])
    assert framing["framing"] == "accuracy_headline"
    assert framing["heterogeneous_pool"] is False


def test_framing_undetermined_on_incomplete_run():
    framing = derive_headline_framing(
        {"single_agent_baseline": {"accuracy": 0.5}}, ["m"]
    )
    assert framing["framing"] == "undetermined"
    assert "latent_based_mas_ours" in framing["reason"]


# ---------------------------------------------------------------------------
# 6. Drift-probe MLP variant (strategy.md §4.4, ablation 7e)
# ---------------------------------------------------------------------------

def test_probe_mlp_variant_fits_and_scores():
    torch.manual_seed(0)
    for arch in ("linear", "mlp"):
        probe = QueryReconstructionProbe(hub_dim=8, query_dim=6, arch=arch)
        assert probe.query_dim == 6  # canonical attr, both archs
        z = torch.randn(32, 8)
        q = torch.randn(32, 6)
        loss = probe.fit_decoder(z, q, n_epochs=5)
        assert loss == loss  # not NaN
        scores = probe(z[:4], q[:4], raise_on_drift=False)
        assert scores.shape == (4,)


def test_probe_unknown_arch_raises():
    with pytest.raises(ValueError, match="probe arch"):
        QueryReconstructionProbe(arch="transformer")
