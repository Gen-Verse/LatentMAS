"""
Shared transformer layer resolution utilities.

All three projects (mrre_drift, latent_coordination, latent_coordination) need to
locate the list of transformer decoder blocks inside a HuggingFace model.
This module centralises that logic so divergence cannot occur.
"""

from __future__ import annotations

from typing import List, Sequence

import torch.nn as nn

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

# Attribute paths tried in order of preference.  Each entry is a dot-separated
# path walked via getattr.  The first one that resolves to a non-empty sequence
# wins.  Paths are ordered from most-common to most-obscure.
_LAYER_PATHS: tuple[str, ...] = (
    "model.layers",                  # LLaMA / Qwen2 / Mistral / Gemma / Phi
    "model.language_model.layers",   # Gemma4 / PaliGemma
    "language_model.model.layers",   # InternVL / LLaVA-style VLMs
    "llm.model.layers",              # custom VLMs
    "transformer.h",                 # GPT-2 / GPT-Neo / GPT-J / Falcon / BLOOM
    "model.decoder.layers",          # OPT / BART decoder
    "encoder.layer",                 # BERT / RoBERTa encoder
    "layers",                        # generic fallback
)


def get_transformer_layers(model: nn.Module) -> List[nn.Module]:
    """Return the ordered list of transformer block modules for any supported arch.

    Probes a set of common HuggingFace attribute paths.  Raises ``AttributeError``
    if none resolve — callers should add the new path to ``_LAYER_PATHS`` rather
    than handling it ad-hoc.

    Args:
        model: A HuggingFace model instance (causal LM, VLM, or encoder).

    Returns:
        Ordered list of transformer layer ``nn.Module`` objects.

    Raises:
        AttributeError: If no standard path resolves to a non-empty layer list.
    """
    for path in _LAYER_PATHS:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            layers = list(obj)
            if layers:
                return layers
        except AttributeError:
            continue

    raise AttributeError(
        f"Cannot locate transformer layer list in {type(model).__name__}. "
        f"Tried paths: {_LAYER_PATHS}. "
        "Add the correct path to shared.model_utils._LAYER_PATHS."
    )


def layer_ids_from_fractions(
    model: nn.Module,
    fractions: Sequence[float],
) -> List[int]:
    """Convert fractional depth positions (0.0–1.0) to absolute layer indices.

    Duplicates are removed and the result is sorted.  Values are clamped to
    ``[0, n_layers - 1]`` so a fraction of 1.0 maps to the last layer.

    Args:
        model: A HuggingFace model instance.
        fractions: Iterable of floats in [0.0, 1.0].

    Returns:
        Sorted, deduplicated list of absolute layer indices.
    """
    n = len(get_transformer_layers(model))
    ids = sorted({max(0, min(n - 1, int(f * n))) for f in fractions})
    return ids
