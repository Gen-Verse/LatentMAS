from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from mrre_drift.models.layers import get_transformer_layers
from mrre_drift.utils.capture import HiddenStateCapture

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


# (english_prompt, target_language_prompt)
PromptPair = Tuple[str, str]


@dataclass
class EnhancementVectors:
    """
    Precomputed cross-lingual enhancement vectors, one per targeted layer.

    v_l = mean_over_pairs[ mean_pool(h_l(en)) - mean_pool(h_l(tgt)) ]

    Injecting alpha * v_l into layer l steers non-English hidden states toward
    the English semantic subspace, unlocking English-trained reasoning circuits.
    """

    layer_ids: List[int]
    vectors: Dict[int, torch.Tensor]  # layer_id → (hidden_dim,)

    def save(self, path: str | Path) -> None:
        torch.save({"layer_ids": self.layer_ids, "vectors": self.vectors}, path)

    @classmethod
    def load(cls, path: str | Path) -> "EnhancementVectors":
        data = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**data)


def compute_enhancement_vectors(
    model: torch.nn.Module,
    tokenizer,
    prompt_pairs: Sequence[PromptPair],
    layer_ids: Sequence[int],
    device: str = "cpu",
) -> EnhancementVectors:
    """
    Compute cross-lingual enhancement vectors by contrasting mean-pooled hidden
    states of English vs target-language prompts at each specified layer.
    """
    all_layers = get_transformer_layers(model)
    target_layers = [all_layers[i] for i in layer_ids]

    model.eval()
    running: Dict[int, torch.Tensor] = {}

    with torch.no_grad():
        for en_text, tgt_text in prompt_pairs:
            en_h = _extract_states(model, tokenizer, en_text, target_layers, layer_ids, device)
            tgt_h = _extract_states(model, tokenizer, tgt_text, target_layers, layer_ids, device)

            for lid in layer_ids:
                delta = en_h[lid] - tgt_h[lid]  # (hidden_dim,)
                running[lid] = running.get(lid, torch.zeros_like(delta)) + delta

    n = len(prompt_pairs)
    vectors = {lid: running[lid] / n for lid in layer_ids}
    return EnhancementVectors(layer_ids=list(layer_ids), vectors=vectors)


def _extract_states(
    model,
    tokenizer,
    text: str,
    target_layers,
    layer_ids: Sequence[int],
    device: str,
) -> Dict[int, torch.Tensor]:
    """Run a single forward pass and return mean-pooled hidden states per layer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with HiddenStateCapture(target_layers) as cap:
        model(**inputs)

    return {lid: cap.states[pos].squeeze(0) for pos, lid in enumerate(layer_ids)}
