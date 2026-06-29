"""Geometry visualizations for analyzing representations and subspaces."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from shared.viz_base import VizConfig, setup_style, save_figure, LANGUAGE_COLOR_PALETTE, get_language_color

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger(__name__)


class GeometryPlotter:
    """Generates geometric analysis plots for representations and subspaces."""

    def __init__(self, config: Optional[VizConfig] = None) -> None:
        self.config = config or VizConfig()
        setup_style(self.config)

    def plot_svd_spectrum(
        self,
        singular_values_dict: Dict[str, np.ndarray],
        save_path: Path | str,
    ) -> None:
        """Plot the singular value spectra showing variance energy distribution."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        for i, (lang, s_vals) in enumerate(singular_values_dict.items()):
            color = get_language_color(lang)
            # Normalize to sum to 1 to show explained variance fraction
            var_fraction = (s_vals ** 2) / (np.sum(s_vals ** 2) + 1e-12)
            cum_var = np.cumsum(var_fraction)

            ax.plot(
                range(1, len(s_vals) + 1),
                cum_var,
                label=f"{lang.upper()} (Cumulative)",
                color=color,
                linestyle="--",
            )
            ax.bar(
                range(1, len(s_vals) + 1),
                var_fraction,
                alpha=0.4,
                color=color,
                label=f"{lang.upper()} (Individual)" if i == 0 else "",
            )

        ax.set_xlabel("Principal Component Index")
        ax.set_ylabel("Explained Variance Fraction")
        ax.set_title("Contrastive SVD Singular Value Spectra")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved SVD spectrum plot to %s", save_path)

    def plot_subspace_overlap(
        self,
        projections: Dict[str, Tensor],
        save_path: Path | str,
    ) -> None:
        """Plot a 2D PCA projection of multilingual representations showing overlap."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        for lang, proj in projections.items():
            proj_np = proj.numpy() if isinstance(proj, Tensor) else np.array(proj)
            color = get_language_color(lang)
            ax.scatter(
                proj_np[:, 0],
                proj_np[:, 1],
                label=lang.upper(),
                color=color,
                alpha=0.7,
                edgecolors="none",
            )

        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title("Multilingual Representation Subspace Projections")
        ax.legend()
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved subspace overlap scatter to %s", save_path)

    def plot_isomorphism_heatmap(
        self,
        isomorphism_matrix: np.ndarray,
        languages: List[str],
        metric_name: str,
        save_path: Path | str,
    ) -> None:
        """Plot pairwise geometric isomorphism as a heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            isomorphism_matrix,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            xticklabels=[l.upper() for l in languages],
            yticklabels=[l.upper() for l in languages],
            ax=ax,
            cbar_kws={"label": metric_name},
        )

        ax.set_title(f"Pairwise Cross-Lingual Geometric Isomorphism ({metric_name})")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved isomorphism heatmap to %s", save_path)

    def plot_magnitude_distortion_by_layer(
        self,
        ratios_by_layer: Dict[int, Dict[str, float]],
        languages: List[str],
        save_path: Path | str,
    ) -> None:
        """Plot per-layer magnitude distortion ratio for each language."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        layers = sorted(ratios_by_layer.keys())

        for lang in languages:
            color = get_language_color(lang)
            # Only plot layers where this language actually has a value. On --resume
            # runs the restored states may cover a subset of (layer, language) pairs,
            # so index defensively and skip missing points rather than crashing.
            pairs = [(l, ratios_by_layer[l][lang]) for l in layers if lang in ratios_by_layer.get(l, {})]
            if not pairs:
                continue
            xs, vals = zip(*pairs)
            ax.plot(
                xs,
                vals,
                marker="o",
                label=lang.upper(),
                color=color,
                linewidth=2,
            )

        ax.axhline(1.0, color="gray", linestyle=":", label="No Distortion (1.0)")
        ax.set_xlabel("Transformer Layer ID")
        ax.set_ylabel("Norm Ratio: ||EN|| / ||LRL||")
        ax.set_title("Layer-Wise Magnitude Distortion Paradox")
        ax.legend()
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved magnitude distortion plot to %s", save_path)

    def plot_cka_similarity(
        self,
        cka_matrix: np.ndarray,
        languages: List[str],
        save_path: Path | str,
    ) -> None:
        """Plot Centered Kernel Alignment similarity heatmap."""
        self.plot_isomorphism_heatmap(cka_matrix, languages, "CKA Similarity", save_path)
