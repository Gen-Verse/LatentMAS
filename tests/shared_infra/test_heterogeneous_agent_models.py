"""CoordinationPipeline._run_stage_a/_run_stage_c actually use each agent's own model_id.

Regression test for a real bug: both stages used to read a single global
self._agent_model_id (== the *first* config entry, i.e. the orchestrator's model) for
every specialized agent's model_id and hidden_dim, silently collapsing
configs/latent_coordination_heterogeneous.yaml's mixed llama/qwen2/cohere pool into 4
copies of the orchestrator's model. No model is ever actually loaded here (BaseAgent
loads lazily on .process(), never called) -- this only checks the AgentConfig objects
and universal_space registrations built during setup.
"""

from unittest.mock import patch

from latent_coordination.pipeline.coordination_pipeline import CoordinationPipeline

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"

_HETEROGENEOUS_CONFIG = {
    "project": {"output_dir": None},  # filled in per-test with a tmp_path
    "agents": [
        {"id": "orchestrator", "model_id": "aisingapore/Llama-SEA-LION-v3-8B-IT", "role": "orchestrator", "device": "cpu"},
        {"id": "translation_agent", "model_id": "sail/Sailor2-8B-Chat", "role": "translation", "device": "cpu"},
        {"id": "reasoning_agent", "model_id": "meta-llama/Llama-3.1-8B-Instruct", "role": "reasoning", "device": "cpu"},
        {"id": "safety_agent", "model_id": "CohereLabs/aya-expanse-8b", "role": "safety", "device": "cpu"},
    ],
}

# Real hidden sizes (2026-07-02, verified via AutoConfig) -- distinct per model, which is
# exactly what the bug collapsed to a single value.
_HIDDEN_DIMS = {
    "aisingapore/Llama-SEA-LION-v3-8B-IT": 4096,
    "sail/Sailor2-8B-Chat": 3584,
    "meta-llama/Llama-3.1-8B-Instruct": 4096,
    "CohereLabs/aya-expanse-8b": 4096,
}


def test_each_specialized_agent_gets_its_own_model_id(tmp_path):
    cfg = {**_HETEROGENEOUS_CONFIG, "project": {"output_dir": str(tmp_path)}}
    pipeline = CoordinationPipeline(cfg)

    with patch.object(
        CoordinationPipeline, "_resolve_agent_hidden_dim",
        staticmethod(lambda model_id: _HIDDEN_DIMS[model_id]),
    ):
        router, universal_space = pipeline._run_stage_a()

    assert router.agents["agent_trans"].config.model_id == "sail/Sailor2-8B-Chat"
    assert router.agents["agent_reason"].config.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert router.agents["agent_safety"].config.model_id == "CohereLabs/aya-expanse-8b"

    # The bug specifically collapsed every agent onto the orchestrator's model.
    orchestrator_model = "aisingapore/Llama-SEA-LION-v3-8B-IT"
    assert router.agents["agent_trans"].config.model_id != orchestrator_model
    assert router.agents["agent_reason"].config.model_id != orchestrator_model
    assert router.agents["agent_safety"].config.model_id != orchestrator_model


def test_each_specialized_agent_gets_its_own_hidden_dim(tmp_path):
    cfg = {**_HETEROGENEOUS_CONFIG, "project": {"output_dir": str(tmp_path)}}
    pipeline = CoordinationPipeline(cfg)

    with patch.object(
        CoordinationPipeline, "_resolve_agent_hidden_dim",
        staticmethod(lambda model_id: _HIDDEN_DIMS[model_id]),
    ):
        router, universal_space = pipeline._run_stage_a()

    # Sailor2 (qwen2, hidden_size=3584) must NOT be resolved via the SEA-LION orchestrator's
    # hidden_dim (4096) -- that mismatch is exactly what silently discarded Stage C's
    # pretrained adapters once Stage E re-registered with the model's real hidden_dim.
    assert router.agents["agent_trans"].config.hidden_dim == 3584
    assert router.agents["agent_reason"].config.hidden_dim == 4096
    assert router.agents["agent_safety"].config.hidden_dim == 4096


def test_stage_c_registers_universal_space_with_per_agent_hidden_dim(tmp_path):
    cfg = {**_HETEROGENEOUS_CONFIG, "project": {"output_dir": str(tmp_path)}}
    pipeline = CoordinationPipeline(cfg)

    with patch.object(
        CoordinationPipeline, "_resolve_agent_hidden_dim",
        staticmethod(lambda model_id: _HIDDEN_DIMS[model_id]),
    ):
        _, universal_space = pipeline._run_stage_a()
        pipeline._run_stage_c(universal_space)

    assert universal_space._agents["agent_trans"].hidden_dim == 3584
    assert universal_space._agents["agent_reason"].hidden_dim == 4096
    assert universal_space._agents["agent_safety"].hidden_dim == 4096
