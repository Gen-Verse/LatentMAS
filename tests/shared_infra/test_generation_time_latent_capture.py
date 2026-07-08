"""Regression tests for dev_doc.md §9 gap 5: the latent state an agent hands off
must be the hidden states computed DURING generation, not a re-encoding of the
agent's own output text.

Uses the same tiny real model as test_latent_injection_dtype.py.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_coordination.agents.base_agent import (
    AgentConfig,
    AgentResponse,
    AgentTask,
    BaseAgent,
)

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"

MODEL_ID = "Qwen/Qwen1.5-0.5B-Chat"


class _TrivialAgent(BaseAgent):
    def process(self, task: AgentTask) -> AgentResponse:
        raise NotImplementedError


def _make_agent(latent_transfer_layer: int = -1) -> _TrivialAgent:
    config = AgentConfig(
        agent_id="test_agent", model_id=MODEL_ID, role="reasoning",
        device="cpu", hidden_dim=1024, dtype="float32",
        latent_transfer_layer=latent_transfer_layer,
    )
    agent = _TrivialAgent(config)
    agent._model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    agent._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    agent._is_loaded = True
    return agent


def test_generate_and_capture_returns_generation_time_states():
    agent = _make_agent()
    n_new = 4
    text, captured = agent.generate_and_capture("Hello", max_new_tokens=n_new)

    assert isinstance(text, str)
    assert captured is not None
    # Prefill contributes the final prompt position; each generated token one
    # more. Generation can stop early on EOS, so 1 < L <= n_new + 1.
    hidden = agent._model.config.hidden_size
    assert captured.dim() == 3
    assert captured.shape[0] == 1
    assert 1 <= captured.shape[1] <= n_new + 1
    assert captured.shape[2] == hidden

    # The captured trajectory must differ from a re-encode of the output text
    # (the old behaviour): compare against extract_hidden_states(text).
    if text.strip():
        reencoded = agent.extract_hidden_states(text, layer_ids=[-1])[-1]
        if reencoded.shape == captured.shape:
            assert not torch.allclose(reencoded.float(), captured.float())


def test_generate_and_capture_honors_transfer_layer():
    agent_last = _make_agent(latent_transfer_layer=-1)
    text_a, cap_last = agent_last.generate_and_capture("Hello", max_new_tokens=3)

    agent_mid = _make_agent(latent_transfer_layer=2)
    text_b, cap_mid = agent_mid.generate_and_capture("Hello", max_new_tokens=3)

    # Same model + greedy decoding → same text, but different capture layers
    # → different state trajectories.
    assert text_a == text_b
    assert cap_last.shape == cap_mid.shape
    assert not torch.allclose(cap_last, cap_mid)


def test_generate_and_capture_with_injection():
    agent = _make_agent()
    hidden = agent._model.config.hidden_size
    injected = torch.randn(1, 4, hidden, dtype=torch.float32)
    text, captured = agent.generate_and_capture(
        "Hello", latent_state=injected, injection_layer=-1, max_new_tokens=3,
    )
    assert isinstance(text, str)
    assert captured is not None and captured.shape[-1] == hidden
