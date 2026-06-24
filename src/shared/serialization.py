"""JSON-safe serialization shared across pipelines.

Reports embed agent outputs whose fields can include ``torch.Tensor`` (e.g.
``AgentResponse.latent_state``) and numpy scalars, which ``json.dump`` cannot serialize.
``to_json_safe`` recursively converts a structure into JSON-serialisable primitives.

Tensors/ndarrays are summarised as ``{"shape", "norm", "dtype"}`` rather than dumped in full —
raw hidden-state values would bloat the report and are not needed for analysis.
"""

from __future__ import annotations

from typing import Any

__all__ = ["to_json_safe"]


def to_json_safe(obj: Any) -> Any:
    """Recursively convert *obj* into JSON-serialisable values."""
    # Lazy imports so this module has no hard torch/numpy dependency.
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            t = obj.detach().float()
            return {
                "__tensor__": True,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "norm": float(t.norm().item()) if t.numel() else 0.0,
            }
    except ImportError:  # pragma: no cover
        pass

    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return {
                "__ndarray__": True,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "norm": float(np.linalg.norm(obj)) if obj.size else 0.0,
            }
        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:  # pragma: no cover
        pass

    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Fallback: stringify unknown objects rather than crash the report.
    return str(obj)
