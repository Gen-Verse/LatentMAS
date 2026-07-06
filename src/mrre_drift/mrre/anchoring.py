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


# (english_language_forcing_prompt, target_language_forcing_prompt)
LanguageForcingPair = Tuple[str, str]


@dataclass
class AnchoringVectors:
    """
    Precomputed target-language output anchoring vectors, one per targeted layer.

    v_l = mean_over_pairs[ mean_pool(h_l(tgt_forcing)) - mean_pool(h_l(en_forcing)) ]

    Injecting alpha * v_l into the final layers pulls the model's output
    distribution away from English and toward the target language, counteracting
    the drift caused by Stage 1's intermediate-layer enhancement injection.
    """

    layer_ids: List[int]
    vectors: Dict[int, torch.Tensor]  # layer_id → (hidden_dim,)

    def save(self, path: str | Path) -> None:
        torch.save({"layer_ids": self.layer_ids, "vectors": self.vectors}, path)

    @classmethod
    def load(cls, path: str | Path) -> "AnchoringVectors":
        data = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**data)


def compute_anchoring_vectors(
    model: torch.nn.Module,
    tokenizer,
    forcing_pairs: Sequence[LanguageForcingPair],
    layer_ids: Sequence[int],
    device: str = "cpu",
) -> AnchoringVectors:
    """
    Compute target-language anchoring vectors by contrasting mean-pooled hidden
    states of target-language-forcing vs English-forcing prompts at each layer.
    """
    all_layers = get_transformer_layers(model)
    target_layers = [all_layers[i] for i in layer_ids]

    model.eval()
    running: Dict[int, torch.Tensor] = {}

    with torch.no_grad():
        for en_text, tgt_text in forcing_pairs:
            en_h = _extract_states(model, tokenizer, en_text, target_layers, layer_ids, device)
            tgt_h = _extract_states(model, tokenizer, tgt_text, target_layers, layer_ids, device)

            for lid in layer_ids:
                # tgt - en: opposite sign from enhancement vectors
                delta = tgt_h[lid] - en_h[lid]
                running[lid] = running.get(lid, torch.zeros_like(delta)) + delta

    n = len(forcing_pairs)
    vectors = {lid: running[lid] / n for lid in layer_ids}
    return AnchoringVectors(layer_ids=list(layer_ids), vectors=vectors)


def _extract_states(
    model,
    tokenizer,
    text: str,
    target_layers,
    layer_ids: Sequence[int],
    device: str,
) -> Dict[int, torch.Tensor]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with HiddenStateCapture(target_layers) as cap:
        model(**inputs)

    return {lid: cap.states[pos].squeeze(0) for pos, lid in enumerate(layer_ids)}
