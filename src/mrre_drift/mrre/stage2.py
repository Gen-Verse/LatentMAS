"""
MRRE Stage 2 — Target-Language Output Anchoring

Counteracts output drift by injecting target-language anchoring vectors into
the final decoder layers.  Stage 1 steers intermediate hidden states toward
English to unlock reasoning circuits; Stage 2 steers the final layers back
toward the target language so the model outputs in the correct language.

Usage
-----
    anchorer = TargetLanguageAnchorer(model, tokenizer, layer_ids=[20, 22, 24])
    anchorer.fit(forcing_pairs)          # compute anchoring vectors

    with anchorer.apply():               # activate injection for this scope
        ids = tokenizer(query, return_tensors="pt").to(device)
        output = model.generate(**ids)

Combined Stage 1 + Stage 2
---------------------------
    with enhancer.apply():
        with anchorer.apply():
            output = model.generate(**ids)
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Sequence

import torch

from mrre_drift.mrre.anchoring import AnchoringVectors, LanguageForcingPair, compute_anchoring_vectors
from mrre_drift.mrre.hooks import register_injection_hooks
from mrre_drift.models.layers import layer_ids_from_fractions

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


class TargetLanguageAnchorer:
    """
    MRRE Stage 2: Target-Language Output Anchoring.

    Parameters
    ----------
    model       : HuggingFace causal LM
    tokenizer   : matching tokenizer
    layer_ids   : absolute transformer layer indices for final-layer injection;
                  typical values cover the last 20–30% of the network
    alpha       : injection scale factor (1.0 = full anchoring vector)
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
        self._vectors: AnchoringVectors | None = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit(self, forcing_pairs: Sequence[LanguageForcingPair]) -> "TargetLanguageAnchorer":
        """
        Compute anchoring vectors from language-forcing prompt pairs.

        Each pair is (english_forcing_prompt, target_language_forcing_prompt):
        prompts that explicitly instruct the model to respond in a given language.
        More pairs → more stable vectors; 20–50 is typically sufficient.
        """
        if not forcing_pairs:
            raise ValueError("forcing_pairs must not be empty")
        self._vectors = compute_anchoring_vectors(
            self.model, self.tokenizer, forcing_pairs, self.layer_ids, self.device
        )
        return self

    # ------------------------------------------------------------------
    # Inference-time injection
    # ------------------------------------------------------------------

    @contextmanager
    def apply(self) -> Generator["TargetLanguageAnchorer", None, None]:
        """
        Context manager that activates target-language anchoring injection
        for every forward pass inside the block.  Hooks are cleanly removed.
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

    def load(self, path: str | Path) -> "TargetLanguageAnchorer":
        self._vectors = AnchoringVectors.load(path)
        return self

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def vectors(self) -> AnchoringVectors | None:
        return self._vectors

    def vector_norms(self) -> dict:
        if self._vectors is None:
            return {}
        return {lid: v.norm().item() for lid, v in self._vectors.vectors.items()}

    def __repr__(self) -> str:
        fitted = self._vectors is not None
        return (
            f"TargetLanguageAnchorer(layers={self.layer_ids}, alpha={self.alpha}, "
            f"fitted={fitted})"
        )
