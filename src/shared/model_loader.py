"""
Shared model loader — single source of truth for HuggingFace model/tokenizer loading
across all three projects (mrre_drift, latent_coordination, latent_coordination).

Design goals
------------
* **V100-safe by default.** Tesla V100 (compute capability 7.0) does **not** support
  bfloat16; requesting it auto-downgrades to float16 with a warning.
* **accelerate backbone.** Supports single-GPU placement, multi-GPU sharding
  (``device_map="auto"`` / explicit ``max_memory``), and 8-bit / 4-bit quantisation via
  bitsandbytes — the path needed to fit 8-9B SEA-LRL models on 16 GB cards.
* **Hook-friendly.** Returns standard ``transformers`` models so forward hooks on hidden
  states keep working (vLLM/llama.cpp/unsloth do not expose these; see DEV_DOC).
* **No fabrication.** Missing dependencies or unresolvable models raise with actionable
  messages rather than silently degrading.

This module deliberately has no project-specific imports so all three pipelines can share it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__status__ = "prototype"

logger = logging.getLogger(__name__)

# bitsandbytes int8/4bit kernels require the model be placed via device_map.
_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def set_alloc_conf() -> None:
    """Reduce CUDA fragmentation OOMs. Safe no-op if already set by the user."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _supports_bf16(device: Optional[str]) -> bool:
    """bfloat16 needs compute capability >= 8.0 (Ampere+). V100 is 7.0."""
    if not torch.cuda.is_available():
        return False
    try:
        idx = 0
        if device and device.startswith("cuda") and ":" in device:
            idx = int(device.split(":", 1)[1])
        major, _ = torch.cuda.get_device_capability(idx)
        return major >= 8
    except Exception:  # pragma: no cover - defensive
        return False


def resolve_dtype(dtype: str, device: Optional[str]) -> torch.dtype:
    """Map a dtype string to ``torch.dtype``.

    G2 compliance: raises ``AssertionError`` if bfloat16 is requested on a
    pre-Ampere GPU (compute capability < 8.0, e.g. V100).  This is a hard
    guard — callers must explicitly request float16 on V100 hardware.
    """
    resolved = _DTYPE_MAP.get(str(dtype).lower())
    if resolved is None:
        raise ValueError(
            f"Unknown dtype '{dtype}'. Valid: {sorted(set(_DTYPE_MAP))}."
        )
    if resolved is torch.bfloat16 and not _supports_bf16(device):
        raise AssertionError(
            f"bfloat16 requested on device '{device}' but GPU compute capability "
            "< 8.0 (V100 is 7.0 — no native bf16 support). "
            "Set dtype='float16' in your config. (G2 hardware envelope guard)"
        )
    return resolved


@dataclass
class ModelLoadSpec:
    """Declarative spec for loading a model. Mirrors the YAML config fields."""

    model_id: str
    device: Optional[str] = "cuda:0"      # ignored when device_map shards across GPUs
    dtype: str = "float16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    device_map: Optional[Any] = None      # None -> {"": device}; or "auto" / dict
    max_memory: Optional[Dict[Any, str]] = None
    output_hidden_states: bool = False
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None  # e.g. "eager" (V100 has no flash-attn-2)
    extra: Dict[str, Any] = field(default_factory=dict)


def load_model_and_tokenizer(
    spec: ModelLoadSpec,
) -> Tuple["torch.nn.Module", Any]:
    """Load a causal LM + tokenizer per ``spec``.

    Returns
    -------
    (model, tokenizer)
        ``model`` is in eval mode. For quantized/sharded loads the model is placed by
        ``accelerate`` (do **not** call ``.to(device)`` afterwards); for plain loads it is
        moved to ``spec.device``.

    Raises
    ------
    ImportError
        If ``transformers`` (or bitsandbytes for quantized loads) is unavailable.
    """
    set_alloc_conf()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required to load models. Install with: pip install transformers"
        ) from exc

    quantized = spec.load_in_8bit or spec.load_in_4bit
    torch_dtype = resolve_dtype(spec.dtype, spec.device)

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": spec.trust_remote_code,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "output_hidden_states": spec.output_hidden_states,
    }
    if spec.attn_implementation:
        load_kwargs["attn_implementation"] = spec.attn_implementation
    load_kwargs.update(spec.extra)

    if quantized:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "bitsandbytes is required for 8-bit/4-bit loading. "
                "Install with: pip install bitsandbytes"
            ) from exc
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=spec.load_in_8bit,
            load_in_4bit=spec.load_in_4bit,
            # fp16 compute for V100 (bf16 unsupported on compute cap 7.0).
            bnb_4bit_compute_dtype=torch_dtype,
        )

    # Quantized/sharded loads MUST go through device_map. Single-GPU default keeps the
    # whole model on one card.
    if spec.device_map is not None:
        load_kwargs["device_map"] = spec.device_map
    elif quantized:
        load_kwargs["device_map"] = {"": spec.device or "cuda:0"}
    if spec.max_memory is not None:
        load_kwargs["max_memory"] = spec.max_memory

    logger.info(
        "Loading model '%s' | dtype=%s 8bit=%s 4bit=%s device=%s device_map=%s",
        spec.model_id, torch_dtype, spec.load_in_8bit, spec.load_in_4bit,
        spec.device, load_kwargs.get("device_map", "(.to)"),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id, trust_remote_code=spec.trust_remote_code
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(spec.model_id, **load_kwargs)

    # Only move manually when accelerate did not place the model.
    if "device_map" not in load_kwargs and spec.device is not None:
        model = model.to(spec.device)

    model.eval()
    logger.info("Model '%s' loaded.", spec.model_id)
    return model, tokenizer
