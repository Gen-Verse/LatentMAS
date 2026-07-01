"""Topology plots for CVAE graph priors and collaboration layouts."""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import networkx as nx
except ImportError:
    nx = None

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


class TopologyPlotter:
    """Generates visualizations of collaborative multi-agent topologies and priors."""

    def __init__(self, config: Optional[VizConfig] = None) -> None:
        self.config = config or VizConfig()
        setup_style(self.config)

    def plot_agent_topology(
        self,
        adj: Tensor | np.ndarray,
        agent_names: List[str],
        save_path: Path | str,
    ) -> None:
        """Draw agent communication layout graph with nodes colored by role."""
        if nx is None:
            logger.warning("networkx not installed; skipping agent topology drawing.")
            return

        fig, ax = plt.subplots(figsize=self.config.figsize_default)
        A = adj.numpy() if isinstance(adj, Tensor) else np.array(adj)

        G = nx.DiGraph()
        for idx, name in enumerate(agent_names):
            G.add_node(name)

        n = len(agent_names)
        for i in range(n):
            for j in range(n):
                if A[i, j] > 0.5:
                    G.add_edge(agent_names[i], agent_names[j], weight=float(A[i, j]))

        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color="skyblue", ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)
        nx.draw_networkx_edges(
            G, pos, arrowstyle="->", arrowsize=15, edge_color="gray", width=1.5, ax=ax
        )

        ax.set_title("Collaborative Multi-Agent Routing Topology")
        ax.axis("off")

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved agent topology plot to %s", save_path)

    def plot_cvae_latent_space(
        self,
        mu: np.ndarray,
        logvar: np.ndarray,
        queries: List[str],
        save_path: Path | str,
    ) -> None:
        """Scatter plot of CVAE latent space colored by query type."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        # Draw scatter colored by query category markers
        categories = ["math", "logic", "translation", "safety", "other"]
        colors = ["red", "blue", "green", "purple", "orange"]

        # Simple assignment for plotting
        assigned = []
        for q in queries:
            matched = False
            for cat in categories[:-1]:
                if cat in q.lower():
                    assigned.append(cat)
                    matched = True
                    break
            if not matched:
                assigned.append("other")

        for cat, color in zip(categories, colors):
            indices = [idx for idx, val in enumerate(assigned) if val == cat]
            if not indices:
                continue
            ax.scatter(
                mu[indices, 0],
                mu[indices, 1],
                color=color,
                label=cat.upper(),
                alpha=0.8,
                edgecolors="black"
            )

        ax.set_xlabel("Latent Dim Z1")
        ax.set_ylabel("Latent Dim Z2")
        ax.set_title("CVAE Latent Space Projection of Graph Priors")
        ax.legend()
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved CVAE latent space plot to %s", save_path)

    def plot_topology_evolution(
        self,
        adj_list: List[np.ndarray],
        timesteps: List[int],
        save_path: Path | str,
    ) -> None:
        """Plot a multi-panel layout showing how topology adapts across communication stages."""
        n_steps = len(adj_list)
        fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4))
        if n_steps == 1:
            axes = [axes]

        for idx, (A, t) in enumerate(zip(adj_list, timesteps)):
            ax = axes[idx]
            sns.heatmap(A, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax)
            ax.set_title(f"Stage t={t}")

        plt.suptitle("Topology Evolution Heatmaps Across Communication Rounds")

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved topology evolution panels to %s", save_path)

    def plot_transfer_matrix(
        self,
        transfer_costs: np.ndarray,
        agent_names: List[str],
        save_path: Path | str,
    ) -> None:
        """Plot a heatmap of pairwise communication weights or costs between agents."""
        fig, ax = plt.subplots(figsize=(7, 6))

        sns.heatmap(
            transfer_costs,
            annot=True,
            fmt=".2f",
            cmap="Oranges",
            xticklabels=agent_names,
            yticklabels=agent_names,
            ax=ax,
            cbar_kws={"label": "State Transfer Bandwidth / Latency (ms)"},
        )

        ax.set_title("Pairwise Latent Transfer Cost Matrix")

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved transfer cost matrix plot to %s", save_path)
