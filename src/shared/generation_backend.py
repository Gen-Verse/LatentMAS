"""
Pluggable text-generation backends shared across projects.

Two backends implement the same minimal contract (``generate(prompts) -> list[str]``):

* :class:`HFBackend` — HuggingFace ``transformers`` + accelerate. The **only** backend that
  exposes intermediate hidden states, so it is mandatory for any hook-based path
  (latent transfer in coordination, activation steering in mechanistic/surgical).
* :class:`VLLMBackend` — vLLM engine for fast batched **text-only** generation. Suitable only
  for paths that consume decoded text (coordination's single-agent / token-based modes).

Hardware gate
-------------
Stock vLLM does not support NVIDIA Volta (V100, compute capability 7.0) — official wheels drop
sm_70 and the engine errors or falls back to CPU. :func:`resolve_backend` therefore enables vLLM
**only** on Ampere+ (cc >= 8.0) with a working ``vllm`` import, and otherwise falls back to the HF
backend with a clear log line (never crashes). Set ``SRE_FORCE_VLLM=1`` to override the gate (e.g.
after building a community sm_70 fork).
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Protocol, Sequence

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger(__name__)


__all__ = [
    "GenerationBackend",
    "HFBackend",
    "VLLMBackend",
    "resolve_backend",
    "vllm_supported",
]


class GenerationBackend(Protocol):
    """Minimal text-generation contract shared by all backends."""

    name: str

    def generate(self, prompts: Sequence[str], max_new_tokens: int = 128, **kwargs) -> List[str]:
        ...

    @property
    def tokenizer(self):
        """Real tokenizer (required so token-cost metrics are never estimated)."""
        ...


# ---------------------------------------------------------------------------
# Capability gate
# ---------------------------------------------------------------------------

def vllm_supported(device: Optional[str] = "cuda:0") -> bool:
    """True iff vLLM can realistically run here.

    Requires (a) ``vllm`` importable and (b) GPU compute capability >= 8.0 (Ampere+),
    unless ``SRE_FORCE_VLLM=1`` overrides the capability check.
    """
    try:
        import vllm  # noqa: F401
    except Exception as exc:  # ImportError or partial/broken build
        logger.debug("vLLM not importable: %s", exc)
        return False

    if os.environ.get("SRE_FORCE_VLLM") == "1":
        logger.warning("SRE_FORCE_VLLM=1 set — bypassing GPU capability check for vLLM.")
        return True

    try:
        import torch
        if not torch.cuda.is_available():
            return False
        idx = 0
        if device and device.startswith("cuda") and ":" in device:
            idx = int(device.split(":", 1)[1])
        major, _ = torch.cuda.get_device_capability(idx)
        if major < 8:
            logger.info(
                "vLLM disabled: GPU compute capability %d.x < 8.0 (e.g. V100/sm_70 is unsupported). "
                "Set SRE_FORCE_VLLM=1 to override after building an sm_70 fork.", major,
            )
            return False
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("vLLM capability probe failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# HuggingFace backend (hook-capable)
# ---------------------------------------------------------------------------

class HFBackend:
    """transformers + accelerate backend. Supports hidden-state hooks."""

    name = "hf"

    def __init__(self, model, tokenizer, device: str = "cuda:0") -> None:
        self.model = model
        self._tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_spec(cls, spec) -> "HFBackend":
        """Build from a :class:`shared.model_loader.ModelLoadSpec`."""
        from shared.model_loader import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(spec)
        return cls(model, tokenizer, device=spec.device or "cuda:0")

    @property
    def tokenizer(self):
        return self._tokenizer

    def generate(self, prompts: Sequence[str], max_new_tokens: int = 128, **kwargs) -> List[str]:
        import torch

        do_sample = kwargs.get("do_sample", False)
        temperature = kwargs.get("temperature", 1.0)
        outputs: List[str] = []
        for prompt in prompts:
            inputs = self._tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(self.device)
            with torch.no_grad():
                gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
                if do_sample:
                    gen_kwargs["temperature"] = temperature
                out_ids = self.model.generate(**inputs, **gen_kwargs)
            new = out_ids[0, inputs["input_ids"].shape[1]:]
            outputs.append(self._tokenizer.decode(new, skip_special_tokens=True))
        return outputs


# ---------------------------------------------------------------------------
# vLLM backend (text-only, fast)
# ---------------------------------------------------------------------------

class VLLMBackend:
    """vLLM engine for fast batched text generation. No hidden-state access."""

    name = "vllm"

    def __init__(
        self,
        model_id: str,
        device: str = "cuda:0",
        dtype: str = "float16",
        quantization: Optional[str] = None,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
    ) -> None:
        from shared.model_loader import set_alloc_conf
        set_alloc_conf()
        from vllm import LLM  # lazy import; only when actually selected

        # Pin to a single GPU; coordination distributes engines across cards by index.
        if device and device.startswith("cuda") and ":" in device:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", device.split(":", 1)[1])

        self.model_id = model_id
        self._llm = LLM(
            model=model_id,
            dtype=dtype,
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        self._tok = None

    @property
    def tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        return self._tok

    def generate(self, prompts: Sequence[str], max_new_tokens: int = 128, **kwargs) -> List[str]:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=kwargs.get("temperature", 0.0),  # greedy by default (matches HF do_sample=False)
            top_p=kwargs.get("top_p", 1.0),
        )
        results = self._llm.generate(list(prompts), params)
        # vLLM may reorder internally but returns results aligned to input order.
        return [r.outputs[0].text for r in results]

    def shutdown(self) -> None:
        """Release the engine + GPU memory so a subsequent HF model can load."""
        try:
            import contextlib
            import gc
            import torch
            del self._llm
            self._llm = None
            gc.collect()
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover
            logger.debug("vLLM shutdown cleanup issue: %s", exc)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

def resolve_backend(
    name: str,
    *,
    model_id: str,
    device: str = "cuda:0",
    dtype: str = "float16",
    load_in_8bit: bool = False,
    hf_model=None,
    hf_tokenizer=None,
    output_hidden_states: bool = False,
) -> GenerationBackend:
    """Return a ready backend honoring task suitability and hardware availability."""
    name = (name or "auto").lower()

    # Auto-select based on task suitability
    if name == "auto":
        if output_hidden_states:
            name = "hf"
            logger.info("Auto-selected HF backend because output_hidden_states=True is required for this task.")
        elif vllm_supported(device):
            name = "vllm"
            logger.info("Auto-selected vLLM backend for fast text-only generation.")
        else:
            name = "hf"
            logger.info("Auto-selected HF backend because vLLM is not supported/available on this hardware.")

    # Validate task suitability against explicit backend requests
    if name == "vllm" and output_hidden_states:
        raise ValueError("vLLM backend does not support output_hidden_states=True. Please use the HF backend for latent state extraction tasks.")

    if name == "vllm":
        if not vllm_supported(device):
            raise RuntimeError(
                "vLLM backend was requested but is not available on this GPU (e.g. V100/sm_70). "
                "Refusing to silently fall back to HF. Please configure the correct backend."
            )
        quant = "bitsandbytes" if load_in_8bit else None
        logger.info("Using vLLM backend for '%s' on %s.", model_id, device)
        return VLLMBackend(model_id=model_id, device=device, dtype=dtype, quantization=quant)

    if name == "hf":
        if hf_model is not None and hf_tokenizer is not None:
            return HFBackend(hf_model, hf_tokenizer, device=device)
        from shared.model_loader import ModelLoadSpec
        spec = ModelLoadSpec(
            model_id=model_id,
            device=device,
            dtype=dtype,
            load_in_8bit=load_in_8bit,
            output_hidden_states=output_hidden_states,
        )
        return HFBackend.from_spec(spec)
        
    raise ValueError(f"Unknown backend requested: {name}")
