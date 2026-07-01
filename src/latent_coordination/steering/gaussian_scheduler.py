"""
Gaussian depth scheduling for layer-wise activation steering.

The Gaussian schedule assigns per-layer injection weights following a
Gaussian bell curve peaked at a specified depth fraction. This provides
smooth, anatomically-motivated injection that avoids disrupting early
feature-formation layers and late high-level reasoning layers.

The weight at layer l is:
    w(l) = alpha_0 * exp( -((l - mu_s)^2) / (2 * sigma_s^2) )
"""

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


logger = logging.getLogger(__name__)


@dataclass
class GaussianScheduleParams:
    """Serialisable parameter container for the Gaussian depth schedule.

    Attributes
    ----------
    alpha_0 : float
        Peak injection amplitude.
    mu_s : float
        Centre of the Gaussian in absolute layer indices.
    sigma_s : float
        Standard deviation in absolute layer indices.
    n_layers : int
        Total number of transformer layers (used for normalisation/plotting).
    """

    alpha_0: float
    mu_s: float
    sigma_s: float
    n_layers: int


class GaussianDepthScheduler:
    """Compute per-layer Gaussian injection weights for activation steering.

    Parameters
    ----------
    alpha_0 : float
        Peak amplitude of the Gaussian.  Typical range ``[0.5, 3.0]``.
    mu_s : float
        Centre layer index (absolute).  Should be in ``[0, n_layers-1]``.
    sigma_s : float
        Spread in layer indices.  Controls the width of the injection window.
    n_layers : int
        Total number of transformer layers in the target model.

    Examples
    --------
    >>> scheduler = GaussianDepthScheduler.from_fractions(
    ...     mu_frac=0.6, sigma_frac=0.15, alpha_0=1.5, n_layers=32
    ... )
    >>> weights = scheduler.get_schedule(32)
    """

    def __init__(
        self,
        alpha_0: float,
        mu_s: float,
        sigma_s: float,
        n_layers: int,
    ) -> None:
        self.alpha_0 = alpha_0
        self.mu_s = mu_s
        self.sigma_s = sigma_s
        self.n_layers = n_layers

        logger.info(
            "GaussianDepthScheduler | alpha_0=%.3f mu_s=%.2f sigma_s=%.2f n_layers=%d",
            alpha_0,
            mu_s,
            sigma_s,
            n_layers,
        )

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_fractions(
        cls,
        mu_frac: float,
        sigma_frac: float,
        alpha_0: float,
        n_layers: int,
    ) -> "GaussianDepthScheduler":
        """Construct a scheduler by specifying centre/spread as fractions of depth.

        Parameters
        ----------
        mu_frac : float
            Centre as fraction of n_layers, e.g. ``0.6`` means layer 60% of depth.
        sigma_frac : float
            Spread as fraction of n_layers.
        alpha_0 : float
            Peak amplitude.
        n_layers : int
            Total transformer depth.

        Returns
        -------
        GaussianDepthScheduler
        """
        if not (0.0 < mu_frac < 1.0):
            raise ValueError(f"mu_frac must be in (0, 1), got {mu_frac}")
        if not (0.0 < sigma_frac <= 1.0):
            raise ValueError(f"sigma_frac must be in (0, 1], got {sigma_frac}")

        mu_s = mu_frac * (n_layers - 1)
        sigma_s = sigma_frac * (n_layers - 1)
        logger.debug(
            "from_fractions | mu_frac=%.2f->mu_s=%.2f sigma_frac=%.2f->sigma_s=%.2f",
            mu_frac,
            mu_s,
            sigma_frac,
            sigma_s,
        )
        return cls(alpha_0=alpha_0, mu_s=mu_s, sigma_s=sigma_s, n_layers=n_layers)

    # ------------------------------------------------------------------
    # Core weight computation
    # ------------------------------------------------------------------

    def get_weight(self, layer_idx: int) -> float:
        """Return the Gaussian injection weight for a single layer.

        Parameters
        ----------
        layer_idx : int
            Zero-based layer index.

        Returns
        -------
        float
            Non-negative weight ``alpha_0 * exp(-((l - mu_s)^2) / (2*sigma_s^2))``.
        """
        if self.sigma_s < 1e-12:
            return self.alpha_0 if layer_idx == round(self.mu_s) else 0.0

        exponent = -((layer_idx - self.mu_s) ** 2) / (2.0 * self.sigma_s ** 2)
        weight = self.alpha_0 * math.exp(exponent)
        return weight

    def get_schedule(self, n_layers: Optional[int] = None) -> List[float]:
        """Return the full per-layer weight vector.

        Parameters
        ----------
        n_layers : int, optional
            Override the instance n_layers.

        Returns
        -------
        List[float]
            Per-layer weights, length ``n_layers``.

        Raises
        ------
        ValueError
            If the resolved ``n_layers`` is zero or negative.
        """
        n = n_layers if n_layers is not None else self.n_layers
        if n <= 0:
            raise ValueError(f"n_layers must be positive, got {n}")
        if self.mu_s >= n:
            logger.warning(
                "mu_s=%.2f is outside the layer range [0, %d); "
                "peak injection will fall at layer %d (clamped).",
                self.mu_s, n, n - 1,
            )
        schedule = [self.get_weight(l) for l in range(n)]
        logger.debug(
            "Schedule | peak_layer=%d peak_weight=%.4f sum=%.4f",
            int(np.argmax(schedule)),
            max(schedule),
            sum(schedule),
        )
        return schedule

    def get_active_layers(
        self,
        n_layers: Optional[int] = None,
        threshold: float = 0.01,
    ) -> List[int]:
        """Return layer indices where weight exceeds threshold.

        Parameters
        ----------
        n_layers : int, optional
            Total layers.
        threshold : float, optional
            Minimum weight to be considered active.

        Returns
        -------
        List[int]
            Active layer indices.
        """
        schedule = self.get_schedule(n_layers)
        active = [i for i, w in enumerate(schedule) if w > threshold]
        logger.info(
            "Active layers (threshold=%.2f): %s", threshold, active
        )
        return active

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_schedule(
        self,
        n_layers: Optional[int] = None,
        save_path: Optional[Path] = None,
        title: str = "Gaussian Depth Schedule",
    ) -> None:
        """Plot the per-layer Gaussian injection weights.

        Parameters
        ----------
        n_layers : int, optional
            Number of layers to plot.
        save_path : Path, optional
            If provided, saves the figure (both PNG and PDF).
        title : str, optional
            Figure title.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not installed; skipping plot.")
            return

        n = n_layers if n_layers is not None else self.n_layers
        schedule = self.get_schedule(n)
        layers = list(range(n))

        sns.set_theme(style="whitegrid", context="paper")
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.bar(layers, schedule, color=sns.color_palette("mako", n_colors=1)[0], alpha=0.85)
        ax.axvline(self.mu_s, color="crimson", linestyle="--", linewidth=1.5, label=f"μ_s = {self.mu_s:.1f}")
        ax.axvspan(
            self.mu_s - self.sigma_s,
            self.mu_s + self.sigma_s,
            alpha=0.12,
            color="crimson",
            label=f"±σ_s = {self.sigma_s:.1f}",
        )
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel("Injection Weight", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, self.alpha_0 * 1.15)
        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
            fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
            logger.info("Saved schedule plot to %s", save_path)
        else:
            plt.show()

        plt.close(fig)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialise the scheduler parameters to a plain dict."""
        return {
            "alpha_0": self.alpha_0,
            "mu_s": self.mu_s,
            "sigma_s": self.sigma_s,
            "n_layers": self.n_layers,
            "__class__": "GaussianDepthScheduler",
            "__version__": __version__,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "GaussianDepthScheduler":
        """Reconstruct a scheduler from a serialised dict.

        Parameters
        ----------
        d : Dict
            Output of :meth:`to_dict`.

        Returns
        -------
        GaussianDepthScheduler
        """
        return cls(
            alpha_0=float(d["alpha_0"]),
            mu_s=float(d["mu_s"]),
            sigma_s=float(d["sigma_s"]),
            n_layers=int(d["n_layers"]),
        )

    def __repr__(self) -> str:
        return (
            f"GaussianDepthScheduler("
            f"alpha_0={self.alpha_0}, mu_s={self.mu_s:.2f}, "
            f"sigma_s={self.sigma_s:.2f}, n_layers={self.n_layers})"
        )
