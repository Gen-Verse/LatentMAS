from __future__ import annotations

import torch
from typing import Dict, List

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"



class HiddenStateCapture:
    """
    Context manager that captures mean-pooled hidden states from specified
    transformer layer modules via forward hooks.

    States are keyed by position in the supplied layer list (0, 1, 2, …),
    shape (batch, hidden_dim) — mean-pooled over the sequence dimension.

    Note: This is intentionally a simple, surgical-MRRE-specific capture
    context. The Mechanistic Disentanglement pipeline's ActivationExtractor is a more fully-featured
    alternative that supports batching, pooling strategies, and attention masks.
    """

    def __init__(self, layer_modules: List[torch.nn.Module]) -> None:
        self._layers = layer_modules
        self.states: Dict[int, torch.Tensor] = {}
        self._handles: list = []

    def __enter__(self) -> "HiddenStateCapture":
        self.states.clear()
        for idx, layer in enumerate(self._layers):
            self._handles.append(layer.register_forward_hook(self._hook(idx)))
        return self

    def __exit__(self, *_) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _hook(self, idx: int):
        def _fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # (batch, seq_len, hidden_dim) → (batch, hidden_dim)
            self.states[idx] = hidden.detach().float().mean(dim=1)
        return _fn
