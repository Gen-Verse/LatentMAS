"""Efficiency plots comparing communication and scaling properties."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
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


class EfficiencyPlotter:
    """Generates plots showing system communication overhead, latency, and convergence."""

    def __init__(self, config: Optional[VizConfig] = None) -> None:
        self.config = config or VizConfig()
        setup_style(self.config)

    def plot_token_vs_latent_cost(
        self,
        ablation_report: Dict,
        save_path: Path | str,
    ) -> None:
        """Plot a side-by-side latency comparison across whichever modes are present in the
        report (mode names are not hard-coded, so single_agent_baseline / token_based_mas /
        latent_based_mas_ours all work)."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        metrics_by_mode = ablation_report.get("metrics_by_mode", {})
        modes = list(metrics_by_mode.keys())
        if not modes:
            logger.warning("plot_token_vs_latent_cost: no metrics_by_mode; skipping plot.")
            plt.close(fig)
            return
        latencies = [metrics_by_mode[m].get("avg_latency_ms", 0.0) for m in modes]
        labels = [m.replace("_", " ").title() for m in modes]

        palette = ["#1F77B4", "#2CA02C", "#FF7F0E", "#D62728", "#9467BD"]
        colors = [palette[i % len(palette)] for i in range(len(modes))]
        ax.bar(labels, latencies, color=colors, alpha=0.8, edgecolor="black")

        ax.set_ylabel("Average Latency (ms)")
        ax.set_xlabel("Communication Protocol")
        ax.set_title("Multi-Agent Latency Comparison")
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved token vs latent cost bar chart to %s", save_path)

    def plot_convergence_curves(
        self,
        accuracy_by_round_per_mode: Dict[str, List[float]],
        save_path: Path | str,
    ) -> None:
        """Plot task solving accuracy trajectories across interaction rounds."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        for mode, accs in accuracy_by_round_per_mode.items():
            rounds = list(range(1, len(accs) + 1))
            ax.plot(rounds, accs, marker="o", label=mode.replace("_", " ").title(), linewidth=2.0)

        ax.set_xlabel("Multi-Agent Coordination Round")
        ax.set_ylabel("Benchmark Accuracy")
        ax.set_title("MAS Collaboration Convergence Rate")
        ax.legend()
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved convergence curves plot to %s", save_path)

    def plot_scalability(
        self,
        n_agents_list: List[int],
        cost_by_mode: Dict[str, List[float]],
        save_path: Path | str,
    ) -> None:
        """Plot scaling cost comparison showing hub-and-spoke O(N) vs fully-connected O(N^2)."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        for mode, costs in cost_by_mode.items():
            label = "Universal Space (O(N) Ours)" if "latent" in mode.lower() else "Fully-Connected (O(N²))"
            linestyle = "-" if "latent" in mode.lower() else "--"
            ax.plot(n_agents_list, costs, marker="s", label=label, linestyle=linestyle, linewidth=2.0)

        ax.set_xlabel("Number of Registered Agents (N)")
        ax.set_ylabel("Communication Bandwidth Cost Index")
        ax.set_title("System Scalability: Adapter-Hub vs Peer-to-Peer")
        ax.legend()
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved scalability plot to %s", save_path)

    def plot_communication_overhead_breakdown(
        self,
        breakdown_dict: Dict[str, List[float]],
        save_path: Path | str,
    ) -> None:
        """Plot a stacked bar chart showing breakdown of overhead categories."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        categories = list(breakdown_dict.keys())
        stages = ["Stage 1", "Stage 2", "Stage 3"]

        bottom = np.zeros(len(stages))
        for cat in categories:
            vals = np.array(breakdown_dict[cat])
            ax.bar(stages, vals, bottom=bottom, label=cat.replace("_", " ").title(), alpha=0.85)
            bottom += vals

        ax.set_ylabel("FLOPs / Token Cost Overhead Metric")
        ax.set_title("Communication Resource Overhead Breakdown")
        ax.legend()
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved overhead breakdown stack bar to %s", save_path)

    def plot_accuracy_vs_latency_tradeoff(
        self,
        results: List[Dict],
        save_path: Path | str,
    ) -> None:
        """Plot a scatter comparing accuracy vs latency for different system configurations."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        for res in results:
            ax.scatter(
                res["latency_ms"],
                res["accuracy"],
                s=120,
                label=res["name"],
                alpha=0.9,
                edgecolors="black"
            )

        ax.set_xlabel("Execution Latency (ms)")
        ax.set_ylabel("Task Solving Accuracy")
        ax.set_title("System Efficiency: Accuracy-vs-Latency Tradeoffs")
        ax.legend()
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved accuracy vs latency scatter to %s", save_path)
