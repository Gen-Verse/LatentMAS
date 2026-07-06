"""
Cross-Lingual Alignment Probe (CLAP).

Decomposes hidden-state matrices at each transformer layer via truncated SVD
to extract the principal concept directions that differentiate English from
target-language representations.  The projection of target-language hidden
states onto the dominant English-aligned direction quantifies how deeply the
model's internal representation has drifted from the target language at each
layer.

This is the diagnostic complement to MRRE Stage 1: Stage 1 corrects the drift
by injecting enhancement vectors; CLAP measures it to identify which layers
need intervention and to validate that the intervention is working.

The ``CLAP delta'' at each layer is defined as:
    delta(l) = mean_cos(en_states, u1) - mean_cos(tgt_states, u1)
where u1 is the top singular vector of the centred [en; tgt] hidden-state
matrix.  A positive delta indicates the English states project more strongly
onto the dominant cross-lingual direction --- i.e. the representation is
English-biased at that layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


try:
    import torch
    import torch.nn as nn
    from mrre_drift.models.layers import get_transformer_layers
    from mrre_drift.utils.capture import HiddenStateCapture
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConceptDirection:
    """A principal direction at one transformer layer (from SVD)."""
    layer_id: int
    direction: "torch.Tensor"       # (hidden_dim,) unit vector
    singular_value: float
    explained_variance_ratio: float


@dataclass
class LayerCLAPResult:
    """CLAP results for a single layer."""
    layer_id: int
    concept_directions: List[ConceptDirection]
    english_alignment: float   # mean cosine sim of en hidden states → direction[0]
    target_alignment: float    # mean cosine sim of tgt hidden states → direction[0]

    @property
    def alignment_delta(self) -> float:
        """High positive delta = representation space is English-biased at this layer."""
        return self.english_alignment - self.target_alignment


# Backwards-compatible alias used by existing pipeline code.
LayerCRAFResult = LayerCLAPResult


@dataclass
class CLAPProfile:
    """Full CLAP decomposition across all layers for a pair of language corpora."""
    layer_results: Dict[int, LayerCLAPResult] = field(default_factory=dict)

    @property
    def alignment_deltas(self) -> Dict[int, float]:
        return {lid: r.alignment_delta for lid, r in self.layer_results.items()}

    def peak_drift_layer(self) -> int:
        return max(self.layer_results, key=lambda lid: self.layer_results[lid].alignment_delta)

    def drift_onset_layer(self, threshold: float = 0.1) -> Optional[int]:
        for lid in sorted(self.layer_results):
            if self.layer_results[lid].alignment_delta >= threshold:
                return lid
        return None

    def safe_intervention_layers(
        self,
        early_layers_fraction: float = 0.3,
        n_layers_total: Optional[int] = None,
        # backwards-compatible alias for early_layers_fraction
        vision_fusion_fraction: Optional[float] = None,
    ) -> List[int]:
        """Layers after the early token-formation zone and above the drift-onset."""
        if vision_fusion_fraction is not None:
            early_layers_fraction = vision_fusion_fraction
        total = n_layers_total or (max(self.layer_results) + 1)
        early_cutoff = int(early_layers_fraction * total)
        onset = self.drift_onset_layer() or early_cutoff

        return [
            lid for lid in sorted(self.layer_results)
            if lid >= max(early_cutoff, onset)
        ]


# Backwards-compatible alias.
CRAFProfile = CLAPProfile


# ---------------------------------------------------------------------------
# CLAP
# ---------------------------------------------------------------------------

class CrossLingualAlignmentProbe:
    """
    Cross-Lingual Alignment Probe (CLAP).

    For each transformer layer, collects mean-pooled hidden states from English
    and target-language prompts, stacks them into a matrix, centres it, and
    performs truncated SVD to extract the principal directions that separate
    the two language spaces.

    Parameters
    ----------
    model        : causal LM
    tokenizer    : matching tokenizer
    n_components : number of SVD components to extract per layer
    device       : computation device
    """

    def __init__(
        self,
        model,
        tokenizer,
        n_components: int = 3,
        device: str = "cpu",
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required to use CrossLingualAlignmentProbe")
        self.model = model
        self.tokenizer = tokenizer
        self.n_components = n_components
        self.device = device

    def profile(
        self,
        en_texts: Sequence[str],
        tgt_texts: Sequence[str],
    ) -> CLAPProfile:
        if not en_texts or not tgt_texts:
            raise ValueError("Both en_texts and tgt_texts must be non-empty")
        with torch.no_grad():
            return self._profile_impl(en_texts, tgt_texts)

    def _profile_impl(self, en_texts, tgt_texts) -> CLAPProfile:
        layers = get_transformer_layers(self.model)
        n_layers = len(layers)

        en_states = self._collect_states(en_texts, layers)
        tgt_states = self._collect_states(tgt_texts, layers)

        profile = CLAPProfile()
        for pos in range(n_layers):
            profile.layer_results[pos] = self._decompose_layer(
                pos, en_states[pos], tgt_states[pos]
            )
        return profile

    def _collect_states(self, texts, layers) -> Dict[int, "torch.Tensor"]:
        per_layer: Dict[int, List["torch.Tensor"]] = {i: [] for i in range(len(layers))}

        self.model.eval()
        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with HiddenStateCapture(layers) as cap:
                self.model(**inputs)

            for pos in range(len(layers)):
                per_layer[pos].append(cap.states[pos].cpu())

        return {pos: torch.cat(vecs, dim=0) for pos, vecs in per_layer.items()}

    def _decompose_layer(
        self,
        layer_id: int,
        en_h: "torch.Tensor",
        tgt_h: "torch.Tensor",
    ) -> LayerCLAPResult:
        combined = torch.cat([en_h, tgt_h], dim=0).float()
        combined = combined - combined.mean(dim=0, keepdim=True)

        k = min(self.n_components, combined.shape[0], combined.shape[1])
        try:
            U, S, Vh = torch.linalg.svd(combined, full_matrices=False)
            directions = Vh[:k]
            singular_values = S[:k]
            total_var = float((S ** 2).sum())
        except Exception:
            diff = (en_h.mean(0) - tgt_h.mean(0)).float()
            norm = diff.norm()
            directions = (diff / norm.clamp(min=1e-8)).unsqueeze(0)
            singular_values = torch.tensor([norm])
            total_var = float(norm ** 2)

        concept_directions = []
        for i in range(len(singular_values)):
            sv = float(singular_values[i])
            evr = (sv ** 2) / total_var if total_var > 0 else 0.0
            direction = directions[i]
            direction = direction / direction.norm().clamp(min=1e-8)
            
            # Enforce sign convention: orient towards English centroid
            if i == 0:
                diff_mean = (en_h.mean(dim=0) - tgt_h.mean(dim=0)).float().to(direction.device)
                if (direction @ diff_mean) < 0:
                    direction = -direction

            concept_directions.append(ConceptDirection(
                layer_id=layer_id,
                direction=direction.cpu(),
                singular_value=sv,
                explained_variance_ratio=evr,
            ))

        d0 = concept_directions[0].direction.to(en_h.device)
        en_cos = _mean_cosine(en_h.float(), d0)
        tgt_cos = _mean_cosine(tgt_h.float(), d0)

        return LayerCLAPResult(
            layer_id=layer_id,
            concept_directions=concept_directions,
            english_alignment=en_cos,
            target_alignment=tgt_cos,
        )


# Backwards-compatible class alias so existing pipeline imports still work.
CRAF = CrossLingualAlignmentProbe


def _mean_cosine(matrix: "torch.Tensor", direction: "torch.Tensor") -> float:
    norms = matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
    d_norm = direction / direction.norm().clamp(min=1e-8)
    sims = (matrix / norms) @ d_norm
    return float(sims.mean())
