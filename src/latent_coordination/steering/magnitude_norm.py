"""
Magnitude Normalizer for cross-lingual activation steering.

Addresses the Magnitude Distortion Paradox: low-resource language (LRL)
hidden states have significantly different L2 norms compared to English,
causing naive steering vector injection to over- or under-steer.

The normalizer scales a steering vector so that its magnitude is proportional
to the target hidden state norms, ensuring calibrated injection regardless of
language.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MagnitudeStats:
    """Statistics for a single (language, layer) pair.

    Attributes
    ----------
    language : str
    layer_id : int
    mean_norm : float
    std_norm : float
    min_norm : float
    max_norm : float
    """

    language: str
    layer_id: int
    mean_norm: float
    std_norm: float
    min_norm: float
    max_norm: float


@dataclass
class MagnitudeAnalysis:
    """Full magnitude analysis over multiple languages and layers.

    Attributes
    ----------
    stats : Dict[str, Dict[int, MagnitudeStats]]
        Nested dict: language -> layer_id -> MagnitudeStats.
    distortion_ratios : Dict[Tuple[str, str], Dict[int, float]]
        Pairwise distortion ratios ``(lang_a, lang_b) -> layer_id -> ratio``.
    reference_language : str
        Language used as the reference (numerator) in distortion ratios.
    """

    stats: Dict[str, Dict[int, MagnitudeStats]]
    distortion_ratios: Dict[Tuple[str, str], Dict[int, float]]
    reference_language: str

    def get_distortion_for(self, lang: str, layer_id: int) -> float:
        """Get the distortion ratio for a specific language and layer vs reference."""
        key = (self.reference_language, lang)
        if key not in self.distortion_ratios:
            raise KeyError(f"No distortion ratio found for pair {key}")
        return self.distortion_ratios[key].get(layer_id, float("nan"))

    def to_dict(self) -> Dict:
        result = {"reference_language": self.reference_language, "distortion_ratios": {}}
        for (la, lb), layer_dict in self.distortion_ratios.items():
            result["distortion_ratios"][f"{la}_vs_{lb}"] = {
                str(k): v for k, v in layer_dict.items()
            }
        return result


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MagnitudeNormalizer:
    """Scale steering vectors to compensate for cross-lingual magnitude distortion.

    This class implements the magnitude normalisation scheme from the paper:
    the steering vector is scaled by ``eta * (mean_norm(h) / norm(sv))``
    so that the injected perturbation is a fraction eta of the residual
    stream magnitude, not an absolute fixed magnitude.

    Parameters
    ----------
    device : str, optional
        Torch device.  Defaults to ``"cpu"``.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        logger.info("MagnitudeNormalizer initialised | device=%s", device)

    # ------------------------------------------------------------------
    # Core normalisation
    # ------------------------------------------------------------------

    def compute_scale(
        self,
        hidden_states: Tensor,
        steering_vector: Tensor,
        eta: float,
    ) -> float:
        """Compute the scalar scale factor for a steering vector.

        Scale = eta * (mean‖hidden‖) / ‖sv‖

        Parameters
        ----------
        hidden_states : Tensor
            Batch of hidden states, shape ``(batch, hidden_dim)`` or
            ``(n_samples, hidden_dim)``.
        steering_vector : Tensor
            Raw (un-normalised) steering vector, shape ``(hidden_dim,)``.
        eta : float
            Injection fraction.  Typical values ``[0.05, 0.5]``.

        Returns
        -------
        float
            The scale factor.
        """
        mean_hs_norm = hidden_states.float().norm(dim=-1).mean().item()
        sv_norm = steering_vector.float().norm().item()

        if sv_norm < 1e-12:
            logger.warning("Steering vector has near-zero norm; scale set to 0.")
            return 0.0

        scale = eta * (mean_hs_norm / sv_norm)
        logger.debug(
            "compute_scale | mean_hs_norm=%.4f sv_norm=%.4f eta=%.3f -> scale=%.4f",
            mean_hs_norm,
            sv_norm,
            eta,
            scale,
        )
        return scale

    def normalize(
        self,
        steering_vector: Tensor,
        hidden_states: Tensor,
        eta: float,
    ) -> Tensor:
        """Return a magnitude-normalised steering vector.

        Parameters
        ----------
        steering_vector : Tensor
            Raw steering vector, shape ``(hidden_dim,)``.
        hidden_states : Tensor
            Representative hidden states from the target context.
        eta : float
            Injection fraction.

        Returns
        -------
        Tensor
            Scaled steering vector, same shape as input.
        """
        scale = self.compute_scale(hidden_states, steering_vector, eta)
        normalised = steering_vector.float() * scale
        logger.debug(
            "Normalised sv norm: %.4f (was %.4f, scale=%.4f)",
            normalised.norm().item(),
            steering_vector.float().norm().item(),
            scale,
        )
        return normalised

    # ------------------------------------------------------------------
    # Distortion analysis
    # ------------------------------------------------------------------

    def compute_distortion_ratio(
        self,
        en_hidden: Tensor,
        lrl_hidden: Tensor,
        layer_id: int = -1,
    ) -> float:
        """Ratio of English to LRL mean norm at a specific layer.

        Parameters
        ----------
        en_hidden : Tensor
            English hidden states, shape ``(n, hidden_dim)``.
        lrl_hidden : Tensor
            LRL hidden states, same shape.
        layer_id : int, optional
            Layer index for logging only.

        Returns
        -------
        float
            Mean ratio ‖en‖ / ‖lrl‖.
        """
        en_norm = en_hidden.float().norm(dim=-1).mean().item()
        lrl_norm = lrl_hidden.float().norm(dim=-1).mean().item()
        ratio = en_norm / (lrl_norm + 1e-12)
        logger.debug(
            "MDR layer=%d | en_norm=%.4f lrl_norm=%.4f ratio=%.4f",
            layer_id,
            en_norm,
            lrl_norm,
            ratio,
        )
        return ratio

    def analyze_language_pairs(
        self,
        hidden_dict: Dict[str, Tensor],
        reference_language: str = "en",
    ) -> MagnitudeAnalysis:
        """Compute pairwise magnitude distortion ratios for all languages.

        Parameters
        ----------
        hidden_dict : Dict[str, Tensor]
            Mapping from language code to hidden state tensor.
            Tensors can be either:
            - ``(n_samples, hidden_dim)`` for a single layer, or
            - ``(n_layers, n_samples, hidden_dim)`` for multi-layer.
        reference_language : str, optional
            The base language (numerator).  Defaults to ``"en"``.

        Returns
        -------
        MagnitudeAnalysis
            Full analysis with per-language, per-layer statistics.
        """
        if reference_language not in hidden_dict:
            raise ValueError(
                f"Reference language '{reference_language}' not in hidden_dict. "
                f"Available: {list(hidden_dict.keys())}"
            )

        # Normalise tensor shapes: always work with (n_samples, hidden_dim)
        # by treating each input as a 2D matrix (possibly multi-layer stacked)
        stats: Dict[str, Dict[int, MagnitudeStats]] = {}
        distortion_ratios: Dict[Tuple[str, str], Dict[int, float]] = {}

        for lang, hs in hidden_dict.items():
            hs_f = hs.float()
            if hs_f.dim() == 2:
                # Single layer
                hs_f = hs_f.unsqueeze(0)  # (1, n_samples, hidden_dim)
            # hs_f: (n_layers, n_samples, hidden_dim)
            n_layers = hs_f.shape[0]
            stats[lang] = {}
            for lid in range(n_layers):
                norms = hs_f[lid].norm(dim=-1).numpy()
                stats[lang][lid] = MagnitudeStats(
                    language=lang,
                    layer_id=lid,
                    mean_norm=float(norms.mean()),
                    std_norm=float(norms.std()),
                    min_norm=float(norms.min()),
                    max_norm=float(norms.max()),
                )

        # Compute pairwise distortion ratios vs reference
        ref_hs = hidden_dict[reference_language].float()
        if ref_hs.dim() == 2:
            ref_hs = ref_hs.unsqueeze(0)

        for lang, hs in hidden_dict.items():
            if lang == reference_language:
                continue
            hs_f = hs.float()
            if hs_f.dim() == 2:
                hs_f = hs_f.unsqueeze(0)

            n_layers = min(ref_hs.shape[0], hs_f.shape[0])
            layer_ratios: Dict[int, float] = {}
            for lid in range(n_layers):
                ratio = self.compute_distortion_ratio(ref_hs[lid], hs_f[lid], lid)
                layer_ratios[lid] = ratio

            distortion_ratios[(reference_language, lang)] = layer_ratios
            logger.info(
                "MDR [%s vs %s] | mean_ratio=%.4f",
                reference_language,
                lang,
                np.mean(list(layer_ratios.values())),
            )

        return MagnitudeAnalysis(
            stats=stats,
            distortion_ratios=distortion_ratios,
            reference_language=reference_language,
        )

    def compute_per_layer_scales(
        self,
        hidden_states_by_layer: Dict[int, Tensor],
        steering_vectors_by_layer: Dict[int, Tensor],
        eta: float,
    ) -> Dict[int, float]:
        """Compute magnitude-normalised scales for each layer independently.

        Parameters
        ----------
        hidden_states_by_layer : Dict[int, Tensor]
            Layer-id -> hidden state matrix.
        steering_vectors_by_layer : Dict[int, Tensor]
            Layer-id -> steering vector.
        eta : float
            Global injection fraction.

        Returns
        -------
        Dict[int, float]
            Layer-id -> scale factor.
        """
        scales: Dict[int, float] = {}
        for lid in steering_vectors_by_layer:
            if lid not in hidden_states_by_layer:
                logger.warning("No hidden states for layer %d; skipping scale.", lid)
                continue
            scales[lid] = self.compute_scale(
                hidden_states_by_layer[lid],
                steering_vectors_by_layer[lid],
                eta,
            )
        return scales
