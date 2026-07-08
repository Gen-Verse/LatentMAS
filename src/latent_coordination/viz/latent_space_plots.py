"""Latent space visualizations for universal space mapping and adapter transfers."""

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


class LatentSpacePlotter:
    """Generates plots showing universal latent space alignments, trajectories, and intent centroids."""

    def __init__(self, config: Optional[VizConfig] = None) -> None:
        self.config = config or VizConfig()
        setup_style(self.config)

    def plot_universal_space_embeddings(
        self,
        agent_states: Dict[str, Tensor],
        agent_names: List[str],
        save_path: Path | str,
    ) -> None:
        """Plot a 2D PCA representation of different agents mapped into the Universal Latent Space."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        from sklearn.decomposition import PCA
        # Collect and project
        all_states = []
        labels = []
        for name in agent_names:
            if name in agent_states:
                s = agent_states[name]
                s_np = s.numpy() if isinstance(s, Tensor) else np.array(s)
                all_states.append(s_np)
                labels.extend([name.upper()] * len(s_np))

        if all_states:
            X = np.concatenate(all_states, axis=0)
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)

            df = pd = None
            try:
                import pandas as pd
                df = pd.DataFrame(X_2d, columns=["PC1", "PC2"])
                df["Agent"] = labels
                sns.scatterplot(data=df, x="PC1", y="PC2", hue="Agent", style="Agent", s=60, alpha=0.8, ax=ax)
            except ImportError:
                # Basic scatter fallback
                ax.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.8)

        ax.set_xlabel("Universal space PC 1")
        ax.set_ylabel("Universal space PC 2")
        ax.set_title("Agent Hidden States Aligned in Universal Latent Space Hub")
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved universal space alignment plot to %s", save_path)

    def plot_transfer_trajectory(
        self,
        sender_states: Tensor,
        transferred_states: Tensor,
        receiver_states: Tensor,
        save_path: Path | str,
    ) -> None:
        """Plot a 3D scatter trajectory representing state transition sender -> universal -> receiver."""
        fig = plt.figure(figsize=self.config.figsize_default)
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax = fig.add_subplot(111, projection="3d")

        s_np = sender_states.numpy() if isinstance(sender_states, Tensor) else np.array(sender_states)
        t_np = transferred_states.numpy() if isinstance(transferred_states, Tensor) else np.array(transferred_states)
        r_np = receiver_states.numpy() if isinstance(receiver_states, Tensor) else np.array(receiver_states)

        # Truncate to first 3 dimensions or use simple PCA mapping
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        X_all = pca.fit_transform(np.concatenate([s_np, t_np, r_np], axis=0))
        n = len(s_np)

        s_3d = X_all[:n]
        t_3d = X_all[n:2*n]
        r_3d = X_all[2*n:]

        ax.scatter(s_3d[:, 0], s_3d[:, 1], s_3d[:, 2], color="red", label="Sender Domain", alpha=0.7)
        ax.scatter(t_3d[:, 0], t_3d[:, 1], t_3d[:, 2], color="gold", label="Universal Hub Space", alpha=0.8, s=50)
        ax.scatter(r_3d[:, 0], r_3d[:, 1], r_3d[:, 2], color="green", label="Receiver Domain", alpha=0.7)

        # Draw connecting line representing transfer trajectory path
        ax.plot(
            [s_3d.mean(axis=0)[0], t_3d.mean(axis=0)[0], r_3d.mean(axis=0)[0]],
            [s_3d.mean(axis=0)[1], t_3d.mean(axis=0)[1], r_3d.mean(axis=0)[1]],
            [s_3d.mean(axis=0)[2], t_3d.mean(axis=0)[2], r_3d.mean(axis=0)[2]],
            color="black", linestyle="--", linewidth=2.0, label="Transfer Path"
        )

        ax.set_xlabel("Latent PC 1")
        ax.set_ylabel("Latent PC 2")
        ax.set_zlabel("Latent PC 3")
        ax.set_title("Adapter State Transfer Trajectory (Sender -> Universal -> Receiver)")
        ax.legend()

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved state transfer trajectory plot to %s", save_path)

    def plot_intent_centroids(
        self,
        centroids: Tensor,
        task_embeddings: Tensor,
        task_labels: List[str],
        save_path: Path | str,
    ) -> None:
        """Plot a 2D scatter of query embeddings showing intent centroids mapping (Voronoi approximation)."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        from sklearn.decomposition import PCA
        c_np = centroids.numpy() if isinstance(centroids, Tensor) else np.array(centroids)
        t_np = task_embeddings.numpy() if isinstance(task_embeddings, Tensor) else np.array(task_embeddings)

        pca = PCA(n_components=2)
        X_all = pca.fit_transform(np.concatenate([c_np, t_np], axis=0))
        n_c = len(c_np)

        c_2d = X_all[:n_c]
        t_2d = X_all[n_c:]

        # Scatter historical queries
        ax.scatter(t_2d[:, 0], t_2d[:, 1], c="gray", alpha=0.4, label="Historical Task Queries")
        # Draw intent centroids
        ax.scatter(c_2d[:, 0], c_2d[:, 1], c="red", marker="D", s=150, edgecolors="black", label="Latent Intent Centroids (k-means)")

        for idx, (x, y) in enumerate(c_2d):
            ax.text(x, y, f" C{idx}", fontsize=12, fontweight="bold")

        ax.set_xlabel("Query Semantic PC 1")
        ax.set_ylabel("Query Semantic PC 2")
        ax.set_title("K-Means Centroid Intent Mapping of Task Latents")
        ax.legend()
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved intent centroids plot to %s", save_path)

    def plot_adapter_reconstruction_quality(
        self,
        original: Tensor,
        reconstructed: Tensor,
        agent_id: str,
        save_path: Path | str,
    ) -> None:
        """Plot reconstruction error metrics (bar chart showing dimensions/channels)."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        orig = original.float()
        recon = reconstructed.float()
        errors = torch.abs(orig - recon).mean(dim=0).cpu().numpy()

        ax.plot(errors, color="#D62728", label="Reconstruction Absolute Error")
        ax.fill_between(range(len(errors)), errors, color="#D62728", alpha=0.2)

        ax.set_xlabel("Latent State Vector Index (Dimension)")
        ax.set_ylabel("Mean Absolute Error (MAE)")
        ax.set_title(f"Universal Adapter Reconstruction Quality: Agent {agent_id.upper()}")
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved adapter reconstruction quality plot to %s", save_path)
