"""
LatentSteerer: Integrated module for Gaussian-scheduled, magnitude-normalised
cross-lingual activation steering.

Integrates:
  - GaussianDepthScheduler (per-layer injection weights)
  - MagnitudeNormalizer (adaptive scaling)
  - SVDSubspaceDecomposer (optional subspace projection before injection)

Uses PyTorch forward hooks for non-invasive latent injection without modifying
model weights.
"""

import json
import logging
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from latent_coordination.steering.gaussian_scheduler import GaussianDepthScheduler
from latent_coordination.steering.magnitude_norm import MagnitudeNormalizer

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
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SteeringConfig:
    """Complete configuration for a LatentSteerer instance.

    Attributes
    ----------
    layer_ids : List[int]
        Layers where hooks will be installed.
    eta : float
        Global injection fraction for magnitude normalisation.
    alpha_0 : float
        Gaussian peak amplitude.
    mu_frac : float
        Gaussian centre as fraction of depth.
    sigma_frac : float
        Gaussian spread as fraction of depth.
    apply_subspace_projection : bool
        Whether to ablate language-specific components before generation.
    n_layers : int
        Total transformer depth.
    language : str
        Target language being steered toward.
    method : str
        Steering vector construction method (e.g. ``"mean_diff"``).
    metadata : Dict
        Any additional metadata.
    """

    layer_ids: List[int]
    eta: float
    alpha_0: float
    mu_frac: float
    sigma_frac: float
    apply_subspace_projection: bool
    n_layers: int
    language: str
    method: str = "mean_diff"
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SteeringConfig":
        return cls(**d)


# ---------------------------------------------------------------------------
# Per-layer hook factory
# ---------------------------------------------------------------------------

class _SteeringHook:
    """A single-layer steering hook that injects a magnitude-normalised,
    Gaussian-weighted steering vector into the residual stream.

    Optionally applies orthogonal projection to ablate language-specific
    components before injection.

    Parameters
    ----------
    layer_id : int
    steering_vector : Tensor
        Shape ``(hidden_dim,)``.
    weight : float
        Gaussian schedule weight for this layer.
    eta : float
        Injection fraction.
    normalizer : MagnitudeNormalizer
    decomposer : optional
        SVDSubspaceDecomposer instance; if provided, applies subspace projection.
    log_injections : bool
        Whether to log the actual injection magnitude.
    """

    def __init__(
        self,
        layer_id: int,
        steering_vector: Tensor,
        weight: float,
        eta: float,
        normalizer: MagnitudeNormalizer,
        decomposer=None,
        log_injections: bool = True,
    ) -> None:
        self.layer_id = layer_id
        self.steering_vector = steering_vector.float()
        self.weight = weight
        self.eta = eta
        self.normalizer = normalizer
        self.decomposer = decomposer
        self.log_injections = log_injections

        # Diagnostics
        self.last_injection_norm: float = 0.0

    def __call__(self, module: nn.Module, inputs: tuple, output) -> Tuple:
        """Hook function applied after each targeted transformer layer."""
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        orig_dtype = hidden_states.dtype
        h = hidden_states.float()
        device = h.device

        sv = self.steering_vector.to(device)

        if h.numel() == 0:
            return output

        # (1) Optional subspace projection: ablate language-surface direction FROM
        #     THE STEERING VECTOR so only the reasoning-subspace component is
        #     injected. Projecting h instead would strip language identity from the
        #     model's hidden state, causing catastrophic all-English output (IFL~1.0)
        #     because the model can no longer use its language-specific activations.
        if self.decomposer is not None:
            sv = self.decomposer.project_to_reasoning(sv.unsqueeze(0)).squeeze(0)

        # (2) Magnitude-normalised, Gaussian-weighted injection vector
        #     Use mean over batch & sequence for scale computation
        h_flat = h.reshape(-1, h.shape[-1])
        scale = self.normalizer.compute_scale(h_flat, sv, self.eta)
        effective_sv = sv * scale * self.weight

        # (3) Inject: broadcast over (batch, seq_len)
        h_steered = h + effective_sv.unsqueeze(0).unsqueeze(0)

        injection_norm = (effective_sv.norm()).item()
        self.last_injection_norm = injection_norm

        if self.log_injections:
            logger.debug(
                "Layer %d | weight=%.4f scale=%.4f injection_norm=%.4f",
                self.layer_id,
                self.weight,
                scale,
                injection_norm,
            )

        h_steered = h_steered.to(orig_dtype)

        if rest is not None:
            return (h_steered,) + rest
        return h_steered


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LatentSteerer:
    """Orchestrate multi-layer activation steering with Gaussian scheduling.

    Parameters
    ----------
    scheduler : GaussianDepthScheduler
        Provides per-layer injection weights.
    normalizer : MagnitudeNormalizer
        Computes adaptive scale factors.
    decomposer : optional
        Fitted SVDSubspaceDecomposer for optional subspace projection.
    log_injections : bool, optional
        Whether to log per-layer injection magnitudes.  Defaults to True.

    Examples
    --------
    >>> steerer = LatentSteerer(scheduler, normalizer)
    >>> with steerer.apply(model, steering_vectors, layer_ids, eta=0.3):
    ...     output = model.generate(**inputs)
    """

    def __init__(
        self,
        scheduler: GaussianDepthScheduler,
        normalizer: MagnitudeNormalizer,
        decomposer=None,
        log_injections: bool = True,
    ) -> None:
        self.scheduler = scheduler
        self.normalizer = normalizer
        self.decomposer = decomposer
        self.log_injections = log_injections

        self._config: Optional[SteeringConfig] = None
        self._active_hooks: List[_SteeringHook] = []
        self._hook_handles = []

        logger.info("LatentSteerer initialised.")

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _decomposer_for(self, layer_id: int):
        """Resolve the decomposer for a layer.

        ``self.decomposer`` may be a single fitted decomposer (applied at every
        layer) or a ``{layer_id: decomposer}`` map (per-layer reasoning subspaces).
        Returns ``None`` when no decomposer is available for the layer.
        """
        if isinstance(self.decomposer, dict):
            return self.decomposer.get(layer_id)
        return self.decomposer

    def register_hooks(
        self,
        model: PreTrainedModel,
        steering_vectors: Dict[int, Tensor],
        layer_ids: List[int],
        eta: float,
        apply_subspace_projection: bool = False,
        use_schedule: bool = True,
    ) -> List:
        """Register forward hooks on specified model layers.

        Parameters
        ----------
        model : PreTrainedModel
            Target model (should be in eval mode).
        steering_vectors : Dict[int, Tensor]
            Layer-id -> steering vector.
        layer_ids : List[int]
            Layers to hook.
        eta : float
            Injection fraction.
        apply_subspace_projection : bool, optional
            Whether to apply SVD subspace projection before injection. No-op unless
            a decomposer (or per-layer decomposer map) was supplied at construction.
        use_schedule : bool, optional
            When True (default) per-layer weights come from the Gaussian depth
            scheduler; when False all layers use a flat weight of 1.0 (uniform
            single-stage steering, e.g. the ``standard_clas`` baseline).

        Returns
        -------
        List
            PyTorch hook handles (for manual removal).
        """
        decoder_layers = self._resolve_decoder_layers(model)
        handles = []
        self._active_hooks = []

        for lid in layer_ids:
            if lid not in steering_vectors:
                logger.warning("No steering vector for layer %d; skipping.", lid)
                continue

            sv = steering_vectors[lid]
            weight = self.scheduler.get_weight(lid) if use_schedule else 1.0

            if weight < 1e-6:
                logger.debug("Layer %d has near-zero weight (%.6f); skipping.", lid, weight)
                continue

            decomposer = self._decomposer_for(lid) if apply_subspace_projection else None
            hook = _SteeringHook(
                layer_id=lid,
                steering_vector=sv,
                weight=weight,
                eta=eta,
                normalizer=self.normalizer,
                decomposer=decomposer,
                log_injections=self.log_injections,
            )

            handle = decoder_layers[lid].register_forward_hook(hook)
            handles.append(handle)
            self._active_hooks.append(hook)

            logger.info(
                "Registered hook | layer=%d weight=%.4f", lid, weight
            )

        self._hook_handles = handles
        return handles

    def remove_hooks(self) -> None:
        """Remove all currently registered forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        n = len(self._hook_handles)
        self._hook_handles = []
        self._active_hooks = []
        logger.info("Removed %d steering hooks.", n)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def apply(
        self,
        model: PreTrainedModel,
        steering_vectors: Dict[int, Tensor],
        layer_ids: List[int],
        eta: float,
        apply_subspace_projection: bool = False,
        use_schedule: bool = True,
    ) -> Generator[None, None, None]:
        """Context manager: register hooks, yield, then clean up.

        Parameters
        ----------
        model : PreTrainedModel
        steering_vectors : Dict[int, Tensor]
        layer_ids : List[int]
        eta : float
        apply_subspace_projection : bool, optional

        Yields
        ------
        None
        """
        try:
            self.register_hooks(
                model,
                steering_vectors,
                layer_ids,
                eta,
                apply_subspace_projection,
                use_schedule=use_schedule,
            )
            logger.info(
                "Steering active on %d layers | eta=%.3f subspace_proj=%s schedule=%s",
                len(self._hook_handles),
                eta,
                apply_subspace_projection,
                use_schedule,
            )
            yield
        finally:
            self.remove_hooks()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        input_text: str,
        steering_vectors: Dict[int, Tensor],
        layer_ids: List[int],
        eta: float,
        max_new_tokens: int = 128,
        apply_subspace_projection: bool = False,
        use_schedule: bool = True,
        **generate_kwargs,
    ) -> str:
        """Generate text with activation steering applied.

        Parameters
        ----------
        model : PreTrainedModel
        tokenizer : PreTrainedTokenizerBase
        input_text : str
        steering_vectors : Dict[int, Tensor]
        layer_ids : List[int]
        eta : float
        max_new_tokens : int, optional
        apply_subspace_projection : bool, optional
        **generate_kwargs
            Additional kwargs forwarded to ``model.generate``.

        Returns
        -------
        str
            Generated text (decoded, without prompt).
        """
        inputs = tokenizer(input_text, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with self.apply(
            model,
            steering_vectors,
            layer_ids,
            eta,
            apply_subspace_projection,
            use_schedule=use_schedule,
        ):
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    **generate_kwargs,
                )

        # Decode only the newly generated tokens
        prompt_len = inputs["input_ids"].shape[1]
        new_ids = output_ids[0, prompt_len:]
        generated = tokenizer.decode(new_ids, skip_special_tokens=True)

        logger.info(
            "Generated %d new tokens | injection norms: %s",
            len(new_ids),
            {h.layer_id: f"{h.last_injection_norm:.4f}" for h in self._active_hooks},
        )

        return generated

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_injection_magnitudes(self) -> Dict[int, float]:
        """Return the last injection norm for each active hook."""
        return {h.layer_id: h.last_injection_norm for h in self._active_hooks}

    # ------------------------------------------------------------------
    # Config serialisation
    # ------------------------------------------------------------------

    def save_config(self, path: Path | str) -> None:
        """Save the SteeringConfig to JSON.

        Parameters
        ----------
        path : Path or str
        """
        if self._config is None:
            raise RuntimeError("No config set. Build one via SteeringConfig and assign to .config.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._config.to_dict(), f, indent=2)
        logger.info("SteeringConfig saved to %s", path)

    @classmethod
    def load_config(cls, path: Path | str) -> SteeringConfig:
        """Load a SteeringConfig from JSON.

        Parameters
        ----------
        path : Path or str

        Returns
        -------
        SteeringConfig
        """
        path = Path(path)
        with open(path) as f:
            d = json.load(f)
        config = SteeringConfig.from_dict(d)
        logger.info("SteeringConfig loaded from %s", path)
        return config

    def set_config(self, config: SteeringConfig) -> None:
        """Attach a SteeringConfig to this steerer."""
        self._config = config

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_decoder_layers(model: PreTrainedModel) -> List[nn.Module]:
        """Resolve decoder layer list from the model."""
        from shared.model_utils import get_transformer_layers
        return get_transformer_layers(model)
