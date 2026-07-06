"""
MRRE Stage 1 + Stage 2 — Surgical Application.

Standard MRRE applies enhancement and anchoring vectors to fixed fractional
depth positions.  Surgical MRRE refines this by:

  1. Accepting a CollapseProfile (from Logit Lens + CRAF analysis) to determine
     which specific layers showed English collapse and are therefore prime targets
     for Stage 1 enhancement injection.

  2. Enforcing a hard exclusion zone covering the vision-fusion layers (the first
     N% of the network).  These layers are off-limits because they carry the
     visual token representations; injecting into them destroys visual alignment.
     Set vision_fusion_fraction=0.0 for text-only LLMs.

  3. Applying Stage 2 anchoring only at the final-layer tail, which is the
     correct semantic position for output-distribution steering.

Usage
-----
    detector = CollapseDetector(n_layers=32)
    collapse = detector.detect(logit_scan)

    surgical = SurgicalMRRE(model, tokenizer, collapse=collapse)
    surgical.fit(prompt_pairs, forcing_pairs)

    with surgical.apply():
        output = model.generate(**inputs)
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional, Sequence

import torch

from mrre_drift.interpret.collapse import CollapseProfile
from mrre_drift.mrre.anchoring import AnchoringVectors, LanguageForcingPair, compute_anchoring_vectors
from mrre_drift.mrre.hooks import make_linear_alpha_ramp, register_injection_hooks
from mrre_drift.mrre.vectors import EnhancementVectors, PromptPair, compute_enhancement_vectors
from mrre_drift.models.layers import get_transformer_layers, layer_ids_from_fractions

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"



@dataclass
class SurgicalMRREConfig:
    """
    Layer selection policy for Surgical MRRE.

    If a CollapseProfile is supplied to SurgicalMRRE, these fractions are used
    only as fallback defaults when the profile does not cover a given range.
    """
    vision_fusion_fraction: float = 0.30   # first N% are off-limits (vision fusion)
    enhancement_fractions: List[float] = field(
        default_factory=lambda: [0.40, 0.55, 0.65]
    )
    anchoring_fractions: List[float] = field(
        default_factory=lambda: [0.75, 0.875]
    )
    alpha_enhancement: float = 1.0
    alpha_anchoring: float = 1.0
    # Magnitude-normalized injection (bounds each layer's perturbation to ~alpha*eta of the
    # hidden-state norm) prevents the over-steering that collapses generation into garbage.
    magnitude_normalize: bool = True
    eta: float = 0.1
    # If the CollapseProfile marks more than this fraction of layers as "safe enhancement"
    # (i.e. no collapse was actually localized, so it flooded the network), ignore the
    # profile and use the discrete `enhancement_fractions` instead. Cross-lingual enhancement
    # is english-ward and degrades script fidelity, so injecting it at most layers destroys output.
    max_enhancement_layer_fraction: float = 0.5
    # Stage-2 anchoring ramp: alpha increases linearly across anchoring layers from
    # alpha_anchoring_min to alpha_anchoring_max (alpha_anchoring is used when ramp is off).
    # The increasing ramp matches the natural norm growth of tail-layer hidden states and
    # prevents the uniform-alpha (CLAS-like) fallback where all anchoring layers get
    # identical injection strength regardless of depth.
    anchoring_ramp: bool = True
    alpha_anchoring_min: float = 0.4
    alpha_anchoring_max: float = 0.8


class SurgicalMRRE:
    """
    Surgical application of MRRE Stage 1 (cross-lingual enhancement) and
    Stage 2 (target-language anchoring) with vision-fusion layer exclusion.

    Parameters
    ----------
    model        : causal LM
    tokenizer    : matching tokenizer
    collapse     : CollapseProfile from Logit Lens + CRAF analysis (optional;
                   if None, falls back to config fractions)
    config       : SurgicalMRREConfig controlling layer selection and alpha values
    device       : device string for calibration forward passes
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        collapse: Optional[CollapseProfile] = None,
        config: Optional[SurgicalMRREConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.collapse = collapse
        self.config = config or SurgicalMRREConfig()
        self.device = device

        self._enhancement_vectors: Optional[EnhancementVectors] = None
        self._anchoring_vectors: Optional[AnchoringVectors] = None

        self._enh_layer_ids = self._resolve_enhancement_layers()
        self._anch_layer_ids = self._resolve_anchoring_layers()

    # ------------------------------------------------------------------
    # Layer selection
    # ------------------------------------------------------------------

    def _resolve_enhancement_layers(self) -> List[int]:
        n_layers = len(get_transformer_layers(self.model))
        fusion_cutoff = int(n_layers * self.config.vision_fusion_fraction)

        profile_layers = self.collapse.safe_enhancement_layers if self.collapse is not None else []
        max_layers = int(n_layers * self.config.max_enhancement_layer_fraction)
        if profile_layers and len(profile_layers) <= max_layers:
            candidates = profile_layers
        else:
            # No localized collapse (profile flooded or empty) → use discrete fractions.
            candidates = layer_ids_from_fractions(
                self.model, self.config.enhancement_fractions
            )

        return [lid for lid in candidates if lid >= fusion_cutoff]

    def _resolve_anchoring_layers(self) -> List[int]:
        n_layers = len(get_transformer_layers(self.model))
        fusion_cutoff = int(n_layers * self.config.vision_fusion_fraction)

        if self.collapse is not None and self.collapse.safe_anchoring_layers:
            candidates = self.collapse.safe_anchoring_layers
        else:
            candidates = layer_ids_from_fractions(
                self.model, self.config.anchoring_fractions
            )

        return [lid for lid in candidates if lid >= fusion_cutoff]

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit(
        self,
        prompt_pairs: Sequence[PromptPair],
        forcing_pairs: Sequence[LanguageForcingPair],
    ) -> "SurgicalMRRE":
        """
        Compute Stage 1 enhancement vectors (from semantically equivalent pairs)
        and Stage 2 anchoring vectors (from language-forcing pairs).
        """
        if not prompt_pairs:
            raise ValueError("prompt_pairs must not be empty")
        if not forcing_pairs:
            raise ValueError("forcing_pairs must not be empty")
        if not self._enh_layer_ids:
            raise RuntimeError(
                "No valid enhancement layers after vision-fusion exclusion. "
                "Adjust config.enhancement_fractions or vision_fusion_fraction."
            )
        if not self._anch_layer_ids:
            raise RuntimeError(
                "No valid anchoring layers after vision-fusion exclusion. "
                "Adjust config.anchoring_fractions or vision_fusion_fraction."
            )

        self._enhancement_vectors = compute_enhancement_vectors(
            self.model, self.tokenizer, prompt_pairs, self._enh_layer_ids, self.device
        )
        self._anchoring_vectors = compute_anchoring_vectors(
            self.model, self.tokenizer, forcing_pairs, self._anch_layer_ids, self.device
        )
        return self

    # ------------------------------------------------------------------
    # Inference-time injection
    # ------------------------------------------------------------------

    def _anchoring_alpha(self) -> "float | dict":
        """Return per-layer alpha for Stage-2 anchoring.

        When ``anchoring_ramp=True`` (the surgical default), alpha increases
        linearly across the anchoring layers from ``alpha_anchoring_min`` to
        ``alpha_anchoring_max``.  This distinguishes the surgical method from a
        CLAS-like uniform injection where every anchoring layer gets the same
        strength regardless of depth.

        When ``anchoring_ramp=False``, the scalar ``alpha_anchoring`` is used
        for all layers (uniform / CLAS-like behaviour — ablation baseline).
        """
        if not self.config.anchoring_ramp:
            return self.config.alpha_anchoring
        return make_linear_alpha_ramp(
            self._anch_layer_ids,
            self.config.alpha_anchoring_min,
            self.config.alpha_anchoring_max,
        )

    @contextmanager
    def apply(self) -> Generator["SurgicalMRRE", None, None]:
        """
        Context manager that activates both Stage 1 (enhancement) and Stage 2
        (anchoring) injection simultaneously.  Both handle sets are removed
        cleanly on exit regardless of exceptions.

        Stage 2 uses a linearly ramped alpha across anchoring layers (surgical
        path) rather than uniform injection (CLAS-like fallback). The ramp is
        controlled by ``SurgicalMRREConfig.anchoring_ramp``.
        """
        if self._enhancement_vectors is None or self._anchoring_vectors is None:
            raise RuntimeError("Call .fit() before .apply()")

        enh_handle = register_injection_hooks(
            self.model, self._enhancement_vectors, self.config.alpha_enhancement,
            normalize=self.config.magnitude_normalize, eta=self.config.eta,
        )
        anch_handle = register_injection_hooks(
            self.model, self._anchoring_vectors, self._anchoring_alpha(),
            normalize=self.config.magnitude_normalize, eta=self.config.eta,
        )
        try:
            yield self
        finally:
            enh_handle.remove()
            anch_handle.remove()

    @contextmanager
    def apply_stage1_only(self) -> Generator["SurgicalMRRE", None, None]:
        """Activates only Stage 1 (cross-lingual enhancement) hooks.

        Ablation baseline: removes Stage 2 anchoring to isolate enhancement contribution.
        """
        if self._enhancement_vectors is None:
            raise RuntimeError("Call .fit() before .apply_stage1_only()")
        handle = register_injection_hooks(
            self.model, self._enhancement_vectors, self.config.alpha_enhancement,
            normalize=self.config.magnitude_normalize, eta=self.config.eta,
        )
        try:
            yield self
        finally:
            handle.remove()

    @contextmanager
    def apply_stage2_only(self) -> Generator["SurgicalMRRE", None, None]:
        """Activates only Stage 2 (target-language anchoring) hooks.

        Ablation baseline: removes Stage 1 enhancement to isolate anchoring contribution.
        Uses the same surgical ramp as :meth:`apply`.
        """
        if self._anchoring_vectors is None:
            raise RuntimeError("Call .fit() before .apply_stage2_only()")
        handle = register_injection_hooks(
            self.model, self._anchoring_vectors, self._anchoring_alpha(),
            normalize=self.config.magnitude_normalize, eta=self.config.eta,
        )
        try:
            yield self
        finally:
            handle.remove()

    @contextmanager
    def apply_uniform_anchor(self) -> Generator["SurgicalMRRE", None, None]:
        """Both stages with anchoring vectors norm-equalized across layers.

        Models vanilla MRRE (no depth-ramp): tests whether the ramped anchoring
        schedule (which produces naturally increasing vector norms at later layers)
        contributes beyond uniform-strength anchoring.
        """
        if self._enhancement_vectors is None or self._anchoring_vectors is None:
            raise RuntimeError("Call .fit() before .apply_uniform_anchor()")
        norms = [v.norm().item() for v in self._anchoring_vectors.vectors.values()]
        mean_norm = sum(norms) / len(norms) if norms else 1.0
        uniform_vecs = AnchoringVectors(
            layer_ids=self._anchoring_vectors.layer_ids,
            vectors={
                lid: v * (mean_norm / (v.norm().item() + 1e-8))
                for lid, v in self._anchoring_vectors.vectors.items()
            },
        )
        enh_handle = register_injection_hooks(
            self.model, self._enhancement_vectors, self.config.alpha_enhancement,
            normalize=self.config.magnitude_normalize, eta=self.config.eta,
        )
        anch_handle = register_injection_hooks(
            self.model, uniform_vecs, self.config.alpha_anchoring,
            normalize=self.config.magnitude_normalize, eta=self.config.eta,
        )
        try:
            yield self
        finally:
            enh_handle.remove()
            anch_handle.remove()

    def fit_randomized_layers(
        self,
        prompt_pairs: Sequence,
        forcing_pairs: Sequence,
        seed: int = 42,
        keep_fraction: float = 0.6,
    ) -> "SurgicalMRRE":
        """Fit a new SurgicalMRRE instance with randomly sampled layer subsets.

        Returns a fresh instance; ``self`` is not modified.  Used to test whether
        the specific layer choices (collapse-guided) matter versus a random draw
        from the same valid pool.

        Parameters
        ----------
        prompt_pairs, forcing_pairs
            Same calibration data used by :meth:`fit`.
        seed
            RNG seed for reproducibility.
        keep_fraction
            Fraction of each stage's layer pool to sample (default 0.6).
        """
        import random as _random
        rng = _random.Random(seed)

        def _sample(layer_ids: List[int]) -> List[int]:
            k = max(1, round(len(layer_ids) * keep_fraction))
            return sorted(rng.sample(layer_ids, min(k, len(layer_ids))))

        n_layers = len(get_transformer_layers(self.model))
        fusion_cutoff = int(n_layers * self.config.vision_fusion_fraction)
        valid_pool = [i for i in range(fusion_cutoff, n_layers)]

        rand_enh = _sample(valid_pool[: len(valid_pool) // 2])   # mid-stack half
        rand_anch = _sample(valid_pool[len(valid_pool) // 2 :])  # late-stack half

        rand_cfg = SurgicalMRREConfig(
            vision_fusion_fraction=self.config.vision_fusion_fraction,
            enhancement_fractions=self.config.enhancement_fractions,
            anchoring_fractions=self.config.anchoring_fractions,
            alpha_enhancement=self.config.alpha_enhancement,
            alpha_anchoring=self.config.alpha_anchoring,
            magnitude_normalize=self.config.magnitude_normalize,
            eta=self.config.eta,
        )
        instance = SurgicalMRRE(
            model=self.model,
            tokenizer=self.tokenizer,
            collapse=None,
            config=rand_cfg,
            device=self.device,
        )
        # Override layer ids with random subsets after construction.
        instance._enh_layer_ids = rand_enh
        instance._anch_layer_ids = rand_anch

        instance._enhancement_vectors = compute_enhancement_vectors(
            self.model, self.tokenizer, prompt_pairs, rand_enh, self.device
        )
        instance._anchoring_vectors = compute_anchoring_vectors(
            self.model, self.tokenizer, forcing_pairs, rand_anch, self.device
        )
        return instance

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if self._enhancement_vectors is None or self._anchoring_vectors is None:
            raise RuntimeError("No vectors to save — call .fit() first")
        self._enhancement_vectors.save(d / "enhancement.pt")
        self._anchoring_vectors.save(d / "anchoring.pt")

    def load(self, directory: str | Path) -> "SurgicalMRRE":
        d = Path(directory)
        self._enhancement_vectors = EnhancementVectors.load(d / "enhancement.pt")
        self._anchoring_vectors = AnchoringVectors.load(d / "anchoring.pt")
        return self

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def enhancement_layer_ids(self) -> List[int]:
        return self._enh_layer_ids

    @property
    def anchoring_layer_ids(self) -> List[int]:
        return self._anch_layer_ids

    def enhancement_norms(self) -> dict:
        if self._enhancement_vectors is None:
            return {}
        return {lid: v.norm().item() for lid, v in self._enhancement_vectors.vectors.items()}

    def anchoring_norms(self) -> dict:
        if self._anchoring_vectors is None:
            return {}
        return {lid: v.norm().item() for lid, v in self._anchoring_vectors.vectors.items()}

    def __repr__(self) -> str:
        fitted = self._enhancement_vectors is not None
        return (
            f"SurgicalMRRE("
            f"enh_layers={self._enh_layer_ids}, "
            f"anch_layers={self._anch_layer_ids}, "
            f"alpha_enh={self.config.alpha_enhancement}, "
            f"alpha_anch={self.config.alpha_anchoring}, "
            f"fitted={fitted})"
        )
