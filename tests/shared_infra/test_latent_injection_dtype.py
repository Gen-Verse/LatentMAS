"""Regression test: BaseAgent.inject_latent_and_generate must not crash when the
injected latent state's dtype differs from the model's runtime dtype.

Found live on 2026-07-02: UniversalLatentHub's decode() returns a plain float32
tensor (LatentAdapter has no dtype awareness), while agents run in float16/8bit.
inject_latent_and_generate's hook injected the float32 tensor as-is, which crashed
the next layer's matmul (e.g. lm_head) with `RuntimeError: expected mat1 and mat2 to
have the same dtype, but got: float != c10::Half` during a real latent_based_mas_ours
run on the heterogeneous pool (crashed inside Sailor2/qwen2's forward pass).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_coordination.agents.base_agent import AgentConfig, AgentTask, BaseAgent, AgentResponse

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"

MODEL_ID = "Qwen/Qwen1.5-0.5B-Chat"


class _TrivialAgent(BaseAgent):
    """Minimal concrete BaseAgent subclass -- process() is unused by this test."""

    def process(self, task: AgentTask) -> AgentResponse:
        raise NotImplementedError


def test_inject_latent_and_generate_handles_float32_injection_into_fp16_model():
    config = AgentConfig(
        agent_id="test_agent", model_id=MODEL_ID, role="reasoning",
        device="cpu", hidden_dim=1024, dtype="float16",
    )
    agent = _TrivialAgent(config)
    # Bypass the real download/quantization path in _ensure_model_loaded -- load
    # directly in float16 to reproduce the dtype mismatch the fix addresses.
    agent._model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    agent._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    agent._is_loaded = True

    hidden_size = agent._model.config.hidden_size
    # Deliberately float32, mimicking UniversalLatentHub.decode()'s plain-fp32 output.
    injected_state = torch.randn(1, 4, hidden_size, dtype=torch.float32)

    # Must not raise -- this is exactly the call path that crashed before the fix.
    result = agent.inject_latent_and_generate(
        injected_state, "Hello", injection_layer=-1, max_new_tokens=4, do_sample=False,
    )
    assert isinstance(result, str)
