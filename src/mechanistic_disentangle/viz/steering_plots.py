"""Steering visualization scripts generating Gaussian schedule curve and trajectory analysis."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy.special
import torch
from torch import Tensor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from shared.viz_base import VizConfig, setup_style, save_figure

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


logger = logging.getLogger(__name__)


class SteeringPlotter:
    """Generates visualizations of steering schedules and their effects on activations/outputs."""

    def __init__(self, config: Optional[VizConfig] = None) -> None:
        self.config = config or VizConfig()
        setup_style(self.config)

    def plot_gaussian_schedule(
        self,
        scheduler,
        n_layers: int,
        save_path: Path | str,
    ) -> None:
        """Plot the Gaussian scheduled injection weights across layers."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        layers = list(range(n_layers))
        weights = [scheduler.get_weight(l) for l in layers]

        ax.plot(layers, weights, marker="o", color="#D62728", linewidth=2.5, label="Gaussian Weight")
        ax.fill_between(layers, weights, alpha=0.15, color="#D62728")

        ax.set_xlabel("Transformer Layer ID")
        ax.set_ylabel("Steering Injection Weight (Multiplier)")
        ax.set_title(f"Gaussian Activation Steering Depth Schedule (n_layers={n_layers})")
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved Gaussian schedule plot to %s", save_path)

    def plot_activation_trajectories(
        self,
        before_states: Dict[int, Tensor],
        after_states: Dict[int, Tensor],
        layer_ids: List[int],
        save_path: Path | str,
    ) -> None:
        """Plot 2D PCA trajectory shift before and after activation steering."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        # Draw arrows connecting before and after states per layer
        for lid in sorted(layer_ids):
            if lid not in before_states or lid not in after_states:
                continue
            b_state = before_states[lid].mean(dim=0).numpy()
            a_state = after_states[lid].mean(dim=0).numpy()

            ax.arrow(
                b_state[0],
                b_state[1],
                a_state[0] - b_state[0],
                a_state[1] - b_state[1],
                head_width=0.05,
                head_length=0.1,
                fc="blue",
                ec="blue",
                alpha=0.6,
                length_includes_head=True,
            )
            ax.scatter(b_state[0], b_state[1], color="red", marker="o", s=50, zorder=3)
            ax.scatter(a_state[0], a_state[1], color="green", marker="X", s=55, zorder=3)
            ax.text(b_state[0], b_state[1], f" L{lid}", fontsize=9, va="bottom")

        # Create dummy plots for custom legend entries
        ax.scatter([], [], color="red", label="Vanilla (Before)", marker="o")
        ax.scatter([], [], color="green", label="Steered (After)", marker="X")

        ax.set_xlabel("PCA Dim 1")
        ax.set_ylabel("PCA Dim 2")
        ax.set_title("Steering Trajectory Shifts in Hidden Space")
        ax.legend()
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved activation trajectories plot to %s", save_path)

    def plot_injection_magnitudes_by_layer(
        self,
        magnitudes_by_layer: Dict[int, float],
        save_path: Path | str,
    ) -> None:
        """Plot a bar chart of injection magnitudes per layer."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        layers = sorted(magnitudes_by_layer.keys())
        norms = [magnitudes_by_layer[l] for l in layers]

        ax.bar(layers, norms, color="#1F77B4", alpha=0.8, edgecolor="black")
        ax.set_xlabel("Transformer Layer ID")
        ax.set_ylabel("Steering Vector L2 Norm")
        ax.set_title("Layer-wise Injection Magnitudes")
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved injection magnitudes plot to %s", save_path)

    def plot_logit_distributions(
        self,
        logits_before: np.ndarray,
        logits_after: np.ndarray,
        top_k: int = 20,
        save_path: Path | str = "",
    ) -> None:
        """Plot the token logit distribution shift before and after steering."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        # Plot logit histogram/density
        sns.kdeplot(logits_before.flatten(), label="Before Steering", fill=True, color="blue", ax=ax, alpha=0.3)
        sns.kdeplot(logits_after.flatten(), label="After Steering", fill=True, color="green", ax=ax, alpha=0.3)

        ax.set_xlabel("Logit Value")
        ax.set_ylabel("Density")
        ax.set_title("Vocabulary Logit Distributions Shift")
        ax.legend()
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved logit distributions plot to %s", save_path)

    def plot_softmax_drift(
        self,
        probs_before: Dict[int, float],
        probs_after: Dict[int, float],
        layer_ids: List[int],
        save_path: Path | str,
    ) -> None:
        """Plot softmax probability mass on target-language tokens by layer."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        layers = sorted(layer_ids)
        before_vals = [probs_before.get(l, 0.0) for l in layers]
        after_vals = [probs_after.get(l, 0.0) for l in layers]

        ax.plot(layers, before_vals, label="Before Steering", marker="o", color="blue", linewidth=1.8)
        ax.plot(layers, after_vals, label="After Steering (CLAS)", marker="x", color="green", linewidth=1.8)

        ax.set_xlabel("Transformer Layer ID")
        ax.set_ylabel("Probability Mass on Target Script Tokens")
        ax.set_title("Layer-wise Softmax Probability Drift")
        ax.legend()
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved softmax drift plot to %s", save_path)

    def plot_residual_stream_norms(
        self,
        hidden_states_by_layer: Dict[int, Tensor],
        save_path: Path | str,
    ) -> None:
        """Plot layer-wise residual stream norm evolution."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        layers = sorted(hidden_states_by_layer.keys())
        norms = [hidden_states_by_layer[l].float().norm(dim=-1).mean().item() for l in layers]

        ax.plot(layers, norms, marker="s", color="purple", linewidth=2.0)
        ax.set_xlabel("Transformer Layer ID")
        ax.set_ylabel("Average Residual Stream L2 Norm")
        ax.set_title("Layer-wise Residual Stream Norm Evolution")
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved residual stream norms plot to %s", save_path)

    def plot_entropy_heatmap(
        self,
        logits_dict: Dict[str, Tensor | np.ndarray],
        save_path: Path | str,
        max_positions: int = 64,
        max_examples: int = 16,
    ) -> None:
        """Plot a heatmap of Shannon entropy (bits) per position for steered/unsteered variants."""
        fig, axes = plt.subplots(1, len(logits_dict), figsize=(5 * len(logits_dict), 4), squeeze=False)

        for col, (label, lgt) in enumerate(logits_dict.items()):
            arr = lgt.detach().cpu().numpy() if isinstance(lgt, Tensor) else np.asarray(lgt)
            if arr.ndim == 2:
                arr = arr[np.newaxis]
            arr = arr[:max_examples, :max_positions]

            log_p = arr - scipy.special.logsumexp(arr, axis=-1, keepdims=True)
            p = np.exp(log_p)
            ent = -np.sum(np.where(p > 0, p * log_p, 0.0), axis=-1) / np.log(2)

            ax = axes[0, col]
            im = ax.imshow(ent, aspect="auto", cmap="hot_r", vmin=0, vmax=max(ent.max(), 1.0))
            ax.set_title(label)
            ax.set_xlabel("Token Position")
            ax.set_ylabel("Batch Example" if col == 0 else "")
            fig.colorbar(im, ax=ax, label="Entropy (bits)")

        fig.suptitle("Entropy Shifts (bits) per Position across Variants", fontsize=11, y=1.02)
        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved entropy heatmap to %s", save_path)

    def plot_logit_correlation(
        self,
        logits_before: Tensor | np.ndarray,
        logits_after: Tensor | np.ndarray,
        save_path: Path | str,
        n_positions: int = 5,
        top_k: int = 256,
    ) -> None:
        """Plot scatter of before-steering vs after-steering logit ranks at individual token positions."""
        from scipy.stats import spearmanr

        tb = logits_before.detach().cpu().numpy() if isinstance(logits_before, Tensor) else np.asarray(logits_before)
        ta = logits_after.detach().cpu().numpy() if isinstance(logits_after, Tensor) else np.asarray(logits_after)

        if tb.ndim == 1:
            tb = tb[np.newaxis]
            ta = ta[np.newaxis]

        n = min(n_positions, tb.shape[0])
        idx = np.linspace(0, tb.shape[0] - 1, n, dtype=int)

        ncols = min(n, 5)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows), squeeze=False)
        axes_flat = axes.ravel()

        for i in range(n):
            pos_idx = idx[i]
            val_b = tb[pos_idx]
            val_a = ta[pos_idx]

            k = min(top_k, len(val_b))
            top_idx = np.argsort(val_b)[-k:][::-1]

            v_b = val_b[top_idx]
            v_a = val_a[top_idx]

            rho, _ = spearmanr(v_b, v_a)
            ax = axes_flat[i]
            ax.scatter(v_b, v_a, s=4, alpha=0.5, c=np.arange(k), cmap="plasma")
            ax.set_title(f"Pos {pos_idx} | ρ={rho:.3f}", fontsize=9)
            ax.set_xlabel("Logits Before", fontsize=8)
            ax.set_ylabel("Logits After", fontsize=8)
            ax.tick_params(labelsize=7)

        for idx_to_hide in range(n, len(axes_flat)):
            axes_flat[idx_to_hide].set_visible(False)

        fig.suptitle("Logit Correlation Matrix (Before vs After Steering)", fontsize=11, y=1.02)
        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved logit correlation scatter to %s", save_path)

    def plot_topk_density(
        self,
        logits_list: List[Tensor | np.ndarray],
        labels: List[str],
        save_path: Path | str,
        k_values: Optional[List[int]] = None,
    ) -> None:
        """Plot cumulative probability mass captured by the top-k tokens for steering variants."""
        if k_values is None:
            k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

        fig, ax = plt.subplots(figsize=self.config.figsize_default)
        cmap = plt.cm.tab10.colors

        for idx, (lgt, label) in enumerate(zip(logits_list, labels)):
            arr = lgt.detach().cpu().numpy() if isinstance(lgt, Tensor) else np.asarray(lgt)
            if arr.ndim == 3:
                arr = arr.reshape(-1, arr.shape[-1])
            elif arr.ndim == 1:
                arr = arr[np.newaxis]

            masses = []
            for k in k_values:
                if k > arr.shape[-1]:
                    masses.append(1.0)
                    continue
                top_vals = np.partition(arr, -k, axis=-1)[..., -k:]
                top_max = top_vals.max(axis=-1, keepdims=True)
                exp_top = np.exp(top_vals - top_max)
                exp_all = np.exp(arr - arr.max(axis=-1, keepdims=True))
                mass = (exp_top.sum(axis=-1) / exp_all.sum(axis=-1)).mean()
                masses.append(float(mass))

            color = cmap[idx % len(cmap)]
            ax.plot(k_values, masses, "o-", label=label, color=color, lw=1.8, markersize=5)

        ax.set_xscale("log", base=2)
        ax.set_xlabel("k (number of top tokens)")
        ax.set_ylabel("Cumulative Probability Mass")
        ax.set_title("Probability Mass Distribution in Top-k Tokens")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.95, color="gray", ls=":", lw=1, alpha=0.5, label="95% Threshold")
        ax.legend(fontsize=9)
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved top-k density plot to %s", save_path)

    def plot_steering_shift(
        self,
        logits_before: Tensor | np.ndarray,
        logits_after: Tensor | np.ndarray,
        save_path: Path | str,
        top_k: int = 64,
    ) -> None:
        """Histogram shift showing logits before and after steering interventions."""
        lb = logits_before.detach().cpu().numpy() if isinstance(logits_before, Tensor) else np.asarray(logits_before)
        la = logits_after.detach().cpu().numpy() if isinstance(logits_after, Tensor) else np.asarray(logits_after)

        if lb.ndim > 1:
            lb = lb[0]
        if la.ndim > 1:
            la = la[0]

        sorted_idx = np.argsort(lb)[::-1]
        top_idx = sorted_idx[:top_k]
        tail_idx = sorted_idx[top_k:]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        def _draw(ax, vals, label_title, color):
            ax.hist(vals[tail_idx], bins=60, color="#cccccc", alpha=0.6, label="Tail")
            ax.hist(vals[top_idx], bins=min(top_k, 30), color=color, alpha=0.8, label=f"Top-{top_k}")
            ax.set_xlabel("Logit value")
            ax.set_ylabel("Count")
            ax.set_title(label_title)
            ax.legend(fontsize=8)
            stats_str = f"μ={vals.mean():.2f}  σ={vals.std():.2f}"
            ax.text(0.98, 0.96, stats_str, ha="right", va="top",
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
            ax.grid(True)

        _draw(axes[0], lb, "Before Steering", "#555555")
        _draw(axes[1], la, "After Steering", "#2ca02c")

        fig.suptitle("Logit Standardisation/Steering Shift (Vocabulary Distribution)", fontsize=11, y=1.01)
        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved steering shift plot to %s", save_path)
