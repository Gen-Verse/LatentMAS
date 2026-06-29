from __future__ import annotations

import torch
from typing import Dict, List, Optional, Union

from mrre_drift.models.layers import get_transformer_layers
from mrre_drift.mrre.vectors import EnhancementVectors

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


class InjectionHandle:
    """Holds registered forward hooks; call .remove() to deactivate all of them."""

    def __init__(self, handles: list) -> None:
        self._handles = handles

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def make_linear_alpha_ramp(
    layer_ids: List[int],
    alpha_min: float,
    alpha_max: float,
) -> Dict[int, float]:
    """Return a dict mapping each layer_id to a linearly ramped alpha value.

    The first layer gets ``alpha_min``, the last gets ``alpha_max``, and
    intermediate layers are interpolated linearly. For a single layer, returns
    ``alpha_min`` for that layer.

    Used by the surgical MRRE Stage-2 anchoring ramp (norms increasing across
    the tail-layer anchoring window).
    """
    n = len(layer_ids)
    if n == 0:
        return {}
    if n == 1:
        return {layer_ids[0]: alpha_min}
    return {
        lid: alpha_min + (alpha_max - alpha_min) * i / (n - 1)
        for i, lid in enumerate(layer_ids)
    }


def register_injection_hooks(
    model: torch.nn.Module,
    ev: EnhancementVectors,
    alpha: Union[float, Dict[int, float]] = 1.0,
    normalize: bool = False,
    eta: float = 0.1,
) -> InjectionHandle:
    """
    Register forward hooks that add a scaled steering vector to the hidden-state
    output of each targeted transformer layer. Returns an InjectionHandle; call
    .remove() when injection should stop.

    Magnitude normalization (``normalize=True``, default)
    -----------------------------------------------------
    Raw ``h + alpha * v`` over-steers: a mean-difference vector can have a norm
    comparable to or larger than the residual stream, and injecting at several
    layers compounds the perturbation, collapsing generation into degenerate
    repetition. Following the magnitude-distortion fix, the per-token injection is
    bounded relative to the hidden-state norm:

        gamma = eta * ||h|| / (||v|| + eps)
        h' = h + alpha * gamma * v

    so each layer perturbs the stream by ~``alpha * eta`` of its own magnitude.
    With ``normalize=False`` the legacy raw ``h + alpha * v`` is used.

    The hook preserves the full output tuple so downstream attention caches and
    other side-outputs are unaffected.
    """
    all_layers = get_transformer_layers(model)
    handles: List = []
    eps = 1e-6

    # Resolve alpha: uniform float → per-layer dict.
    alpha_map: Dict[int, float] = (
        alpha if isinstance(alpha, dict) else {lid: float(alpha) for lid in ev.vectors}
    )

    for lid, vec in ev.vectors.items():
        layer = all_layers[lid]
        layer_alpha = alpha_map.get(lid, float(alpha) if not isinstance(alpha, dict) else 1.0)

        # Capture by value to avoid late-binding bugs.
        def _hook(module, input, output, _v=vec, _a=layer_alpha, _norm=normalize, _eta=eta):
            hidden = output[0] if isinstance(output, tuple) else output
            v = _v.to(dtype=hidden.dtype, device=hidden.device)
            if _norm:
                v_norm = v.norm() + eps
                h_norm = hidden.norm(dim=-1, keepdim=True)   # (..., 1) per-token
                gamma = _eta * h_norm / v_norm
                injected = hidden + _a * gamma * v
            else:
                injected = hidden + _a * v
            if isinstance(output, tuple):
                return (injected,) + output[1:]
            return injected

        handles.append(layer.register_forward_hook(_hook))

    return InjectionHandle(handles)
