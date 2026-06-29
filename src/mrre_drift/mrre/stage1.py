"""
MRRE Stage 1 — Cross-Lingual Reasoning Enhancement

Steers non-English hidden states toward the English semantic subspace at
intermediate transformer layers, unlocking the English-trained reasoning
circuits for foreign-language inputs without any weight updates.

Usage
-----
    enhancer = CrossLingualEnhancer(model, tokenizer, layer_ids=[8, 12, 16])
    enhancer.fit(prompt_pairs)          # compute enhancement vectors

    with enhancer.apply():              # activate injection for this scope
        ids = tokenizer(query, return_tensors="pt").to(device)
        output = model.generate(**ids)
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Sequence

import torch

from mrre_drift.mrre.hooks import register_injection_hooks
from mrre_drift.mrre.vectors import EnhancementVectors, PromptPair, compute_enhancement_vectors
from mrre_drift.models.layers import layer_ids_from_fractions

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


class CrossLingualEnhancer:
    """
    MRRE Stage 1: Cross-Lingual Reasoning Enhancement.

    Parameters
    ----------
    model       : HuggingFace causal LM
    tokenizer   : matching tokenizer
    layer_ids   : absolute transformer layer indices for intermediate injection;
                  use layer_ids_from_fractions() to express as depth fractions —
                  typical values cover 40–70% of the network depth
    alpha       : injection scale factor (1.0 = full enhancement vector)
    device      : device string for calibration forward passes
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        layer_ids: List[int],
        alpha: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer_ids = layer_ids
        self.alpha = alpha
        self.device = device
        self._vectors: EnhancementVectors | None = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit(self, prompt_pairs: Sequence[PromptPair]) -> "CrossLingualEnhancer":
        """
        Compute enhancement vectors from semantically equivalent prompt pairs.

        Each pair is (english_prompt, target_language_prompt) with the same
        semantic content.  More pairs → more stable vectors; 20–50 is typical.
        """
        if not prompt_pairs:
            raise ValueError("prompt_pairs must not be empty")
        self._vectors = compute_enhancement_vectors(
            self.model, self.tokenizer, prompt_pairs, self.layer_ids, self.device
        )
        return self

    # ------------------------------------------------------------------
    # Inference-time injection
    # ------------------------------------------------------------------

    @contextmanager
    def apply(self) -> Generator["CrossLingualEnhancer", None, None]:
        """
        Context manager that activates cross-lingual enhancement injection for
        every forward pass inside the block.  Hooks are cleanly removed on exit.
        """
        if self._vectors is None:
            raise RuntimeError("Call .fit() before .apply()")
        handle = register_injection_hooks(self.model, self._vectors, self.alpha)
        try:
            yield self
        finally:
            handle.remove()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        if self._vectors is None:
            raise RuntimeError("No vectors to save — call .fit() first")
        self._vectors.save(path)

    def load(self, path: str | Path) -> "CrossLingualEnhancer":
        self._vectors = EnhancementVectors.load(path)
        return self

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def vectors(self) -> EnhancementVectors | None:
        return self._vectors

    def vector_norms(self) -> dict:
        if self._vectors is None:
            return {}
        return {lid: v.norm().item() for lid, v in self._vectors.vectors.items()}

    def __repr__(self) -> str:
        fitted = self._vectors is not None
        return (
            f"CrossLingualEnhancer(layers={self.layer_ids}, alpha={self.alpha}, "
            f"fitted={fitted})"
        )
