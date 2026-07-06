"""
SteeringVectorBuilder: Construct cross-lingual steering vectors from contrastive
parallel activation pairs.

Supports:
  - Mean-difference (standard baseline)
  - Aggregated multilingual vectors averaged over multiple target languages
  - SVD subspace-projected steering vectors (orthogonal to language-specific axes)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import Tensor

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SteeringVectors:
    """Container for a set of per-layer steering vectors.

    Attributes
    ----------
    layer_ids : List[int]
        Layers for which vectors are defined.
    vectors : Dict[int, Tensor]
        Mapping from layer_id to steering vector of shape ``(hidden_dim,)``.
    method : str
        Method used to build the vectors (e.g. ``"mean_diff"``,
        ``"aggregated_multilingual"``, ``"subspace_steering"``).
    metadata : Dict
        Extra build metadata (languages used, n_pairs, etc.).
    """

    layer_ids: List[int]
    vectors: Dict[int, Tensor]
    method: str
    metadata: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.layer_ids)

    def save(self, path: Path | str) -> None:
        """Persist to disk using torch.save.

        Parameters
        ----------
        path : Path or str
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "layer_ids": self.layer_ids,
            "vectors": self.vectors,
            "method": self.method,
            "metadata": self.metadata,
        }
        torch.save(payload, path)
        logger.info("SteeringVectors saved to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "SteeringVectors":
        """Load from disk.

        Parameters
        ----------
        path : Path or str

        Returns
        -------
        SteeringVectors
        """
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(
            layer_ids=payload["layer_ids"],
            vectors=payload["vectors"],
            method=payload["method"],
            metadata=payload["metadata"],
        )
        logger.info("SteeringVectors loaded from %s", path)
        return instance


# ---------------------------------------------------------------------------
# Builder class
# ---------------------------------------------------------------------------

class SteeringVectorBuilder:
    """Construct cross-lingual steering vectors from hidden state dictionaries.

    All methods take ``Dict[int, Tensor]`` (layer_id → hidden states of shape
    ``(n_samples, hidden_dim)``) as input and return :class:`SteeringVectors`.

    Parameters
    ----------
    device : str, optional
        Torch device.  Defaults to ``"cpu"``.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        logger.info("SteeringVectorBuilder initialised | device=%s", device)

    # ------------------------------------------------------------------
    # Mean-difference steering (standard baseline)
    # ------------------------------------------------------------------

    def build_mean_diff(
        self,
        en_states: Dict[int, Tensor],
        tgt_states: Dict[int, Tensor],
        layer_ids: Optional[List[int]] = None,
    ) -> SteeringVectors:
        """Build mean-difference steering vectors: sv_l = mean(tgt_l) - mean(en_l).

        Parameters
        ----------
        en_states : Dict[int, Tensor]
            English hidden states per layer.
        tgt_states : Dict[int, Tensor]
            Target-language hidden states per layer.
        layer_ids : List[int], optional
            Subset of layers to build for.  Defaults to all shared keys.

        Returns
        -------
        SteeringVectors
        """
        shared_layers = sorted(set(en_states.keys()) & set(tgt_states.keys()))
        if layer_ids is not None:
            shared_layers = [l for l in layer_ids if l in shared_layers]

        if not shared_layers:
            raise ValueError("No shared layer ids between en_states and tgt_states.")

        vectors: Dict[int, Tensor] = {}
        for lid in shared_layers:
            en = en_states[lid].float().to(self.device)
            tgt = tgt_states[lid].float().to(self.device)

            if en.shape != tgt.shape:
                raise ValueError(
                    f"Shape mismatch at layer {lid}: en={en.shape} tgt={tgt.shape}"
                )

            sv = tgt.mean(dim=0) - en.mean(dim=0)
            vectors[lid] = sv.cpu()
            logger.debug(
                "mean_diff | layer=%d sv_norm=%.4f", lid, sv.norm().item()
            )

        logger.info(
            "Built mean-diff steering vectors | %d layers | sv_norm_range=[%.4f, %.4f]",
            len(vectors),
            min(v.norm().item() for v in vectors.values()),
            max(v.norm().item() for v in vectors.values()),
        )

        return SteeringVectors(
            layer_ids=shared_layers,
            vectors=vectors,
            method="mean_diff",
            metadata={
                "n_samples": next(iter(en_states.values())).shape[0],
                "hidden_dim": next(iter(en_states.values())).shape[1],
            },
        )

    # ------------------------------------------------------------------
    # Aggregated multilingual steering
    # ------------------------------------------------------------------

    def build_aggregated_multilingual(
        self,
        states_by_lang: Dict[str, Dict[int, Tensor]],
        en_states: Dict[int, Tensor],
        layer_ids: Optional[List[int]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> SteeringVectors:
        """Build aggregated steering vectors averaged over multiple target languages.

        For each layer l:
            sv_l = sum_lang( w_lang * (mean(tgt_lang_l) - mean(en_l)) ) / sum(w_lang)

        Parameters
        ----------
        states_by_lang : Dict[str, Dict[int, Tensor]]
            Mapping from language code to per-layer hidden states.
        en_states : Dict[int, Tensor]
            English per-layer hidden states.
        layer_ids : List[int], optional
            Subset of layers.
        weights : Dict[str, float], optional
            Per-language weights.  Defaults to uniform weighting.

        Returns
        -------
        SteeringVectors
        """
        languages = list(states_by_lang.keys())
        if not languages:
            raise ValueError("states_by_lang is empty.")

        if weights is None:
            weights = {lang: 1.0 for lang in languages}

        # Validate all languages have the same layer ids
        all_layer_sets = [set(states_by_lang[lang].keys()) for lang in languages]
        shared_layers = set(en_states.keys())
        for layer_set in all_layer_sets:
            shared_layers &= layer_set
        shared_layers = sorted(shared_layers)

        if layer_ids is not None:
            shared_layers = [l for l in layer_ids if l in shared_layers]

        if not shared_layers:
            raise ValueError("No shared layers across all languages.")

        total_weight = sum(weights.get(lang, 1.0) for lang in languages)
        vectors: Dict[int, Tensor] = {}

        for lid in shared_layers:
            en = en_states[lid].float().to(self.device)
            agg = torch.zeros(en.shape[1], device=self.device)

            for lang in languages:
                tgt = states_by_lang[lang][lid].float().to(self.device)
                lang_sv = tgt.mean(dim=0) - en.mean(dim=0)
                w = weights.get(lang, 1.0)
                agg = agg + (lang_sv * w)

            sv = agg / (total_weight + 1e-12)
            vectors[lid] = sv.cpu()
            logger.debug(
                "aggregated_multilingual | layer=%d sv_norm=%.4f",
                lid,
                sv.norm().item(),
            )

        logger.info(
            "Built aggregated multilingual steering | %d languages | %d layers",
            len(languages),
            len(vectors),
        )

        return SteeringVectors(
            layer_ids=shared_layers,
            vectors=vectors,
            method="aggregated_multilingual",
            metadata={
                "languages": languages,
                "weights": {k: float(v) for k, v in weights.items()},
                "n_samples": next(iter(en_states.values())).shape[0],
            },
        )

    # ------------------------------------------------------------------
    # Subspace steering
    # ------------------------------------------------------------------

    def build_subspace_steering(
        self,
        decomposer,
        en_states: Dict[int, Tensor],
        tgt_states: Dict[int, Tensor],
        layer_ids: Optional[List[int]] = None,
    ) -> SteeringVectors:
        """Build SVD-projected steering vectors.

        Projects both en and tgt mean states onto the reasoning subspace
        (language-component ablated) before computing the difference.
        This creates steering vectors that operate orthogonally to
        language-surface features.

        Parameters
        ----------
        decomposer : SVDSubspaceDecomposer
            A fitted decomposer.
        en_states : Dict[int, Tensor]
            English per-layer hidden states.
        tgt_states : Dict[int, Tensor]
            Target per-layer hidden states.
        layer_ids : List[int], optional
            Subset of layers.

        Returns
        -------
        SteeringVectors
        """
        shared_layers = sorted(set(en_states.keys()) & set(tgt_states.keys()))
        if layer_ids is not None:
            shared_layers = [l for l in layer_ids if l in shared_layers]

        vectors: Dict[int, Tensor] = {}
        for lid in shared_layers:
            en = en_states[lid].float()
            tgt = tgt_states[lid].float()

            # Project to reasoning subspace (ablate language-specific axes)
            en_proj = decomposer.project_to_reasoning(en)
            tgt_proj = decomposer.project_to_reasoning(tgt)

            sv = tgt_proj.mean(dim=0) - en_proj.mean(dim=0)
            vectors[lid] = sv.cpu()
            logger.debug(
                "subspace_steering | layer=%d sv_norm=%.4f",
                lid,
                sv.norm().item(),
            )

        logger.info(
            "Built subspace steering vectors | %d layers", len(vectors)
        )

        return SteeringVectors(
            layer_ids=shared_layers,
            vectors=vectors,
            method="subspace_steering",
            metadata={
                "n_components": decomposer.n_components,
                "language_variance_ratio": (
                    decomposer.get_result().language_specific_variance_ratio
                    if decomposer._result is not None else None
                ),
            },
        )
