"""
Graph utility functions for multi-agent topology construction and analysis.

Provides conversions between adjacency matrices and edge lists, standard
topology constructors (hub-and-spoke, ring, fully-connected, Erdős-Rényi),
graph-theoretic property computation, and matplotlib/networkx visualisation.
"""

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Soft imports for visualisation
try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False
    logger.warning("networkx not installed; plot_topology will be unavailable.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    logger.warning("matplotlib not installed; plot_topology will be unavailable.")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GraphProperties:
    """Computed structural properties of an agent communication graph.

    Attributes:
        n_nodes: Number of nodes (agents).
        n_edges: Number of directed edges.
        density: Edge density = n_edges / (n_nodes * (n_nodes - 1)).
        avg_degree: Average out-degree.
        is_connected: Whether the underlying undirected graph is connected.
        diameter: Graph diameter (-1 if disconnected).
        clustering_coefficient: Average local clustering coefficient.
    """

    n_nodes: int
    n_edges: int
    density: float
    avg_degree: float
    is_connected: bool
    diameter: int
    clustering_coefficient: float


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

class GraphUtils:
    """Static utility methods for agent communication topology manipulation.

    All methods accept and return plain PyTorch tensors so they can be used
    inside a GPU-resident training loop without leaving the tensor ecosystem.
    For graph-theoretic properties (connectivity, diameter, clustering) we
    temporarily convert to ``networkx`` if available, falling back to pure
    PyTorch implementations otherwise.
    """

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def adjacency_to_edge_list(adj: Tensor) -> List[Tuple[int, int]]:
        """Convert a binary adjacency matrix to an edge list.

        Self-loops (diagonal entries) are excluded.

        Args:
            adj: Binary adjacency tensor of shape (N, N).

        Returns:
            List of (src, dst) integer index pairs for all edges where
            ``adj[src, dst] > 0.5``.

        Example::

            edges = GraphUtils.adjacency_to_edge_list(torch.eye(4))
            # edges == []  (no self-loops)
        """
        N = adj.shape[0]
        edges: List[Tuple[int, int]] = []
        for i in range(N):
            for j in range(N):
                if i != j and adj[i, j].item() > 0.5:
                    edges.append((i, j))
        return edges

    @staticmethod
    def edge_list_to_adjacency(
        edges: Sequence[Tuple[int, int]],
        n_nodes: int,
    ) -> Tensor:
        """Convert an edge list to a binary adjacency matrix.

        Args:
            edges: Sequence of (src, dst) integer pairs.
            n_nodes: Number of nodes (determines matrix size).

        Returns:
            Float binary adjacency tensor of shape (n_nodes, n_nodes).
        """
        adj = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
        for src, dst in edges:
            if 0 <= src < n_nodes and 0 <= dst < n_nodes:
                adj[src, dst] = 1.0
        return adj

    # ------------------------------------------------------------------
    # Graph property computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_graph_properties(adj: Tensor) -> GraphProperties:
        """Compute structural properties of an agent communication graph.

        Args:
            adj: Binary adjacency tensor of shape (N, N).  Off-diagonal
                entries > 0.5 are treated as edges; the diagonal is ignored.

        Returns:
            A :class:`GraphProperties` dataclass populated with computed values.
            If ``networkx`` is unavailable, ``is_connected`` defaults to ``True``
            and ``diameter`` / ``clustering_coefficient`` to ``-1`` / ``0.0``.
        """
        N = adj.shape[0]
        # Zero-out diagonal
        mask = 1.0 - torch.eye(N, device=adj.device)
        binary = ((adj * mask) > 0.5).float()
        n_edges = int(binary.sum().item())
        max_edges = N * (N - 1)
        density = n_edges / max_edges if max_edges > 0 else 0.0
        avg_degree = binary.sum(dim=1).mean().item()

        is_connected = True
        diameter = -1
        clustering = 0.0

        if _HAS_NX:
            G_nx = nx.from_numpy_array(binary.cpu().numpy())
            ug = G_nx.to_undirected()
            is_connected = nx.is_connected(ug)
            if is_connected:
                try:
                    diameter = nx.diameter(ug)
                except Exception:
                    diameter = -1
            try:
                clustering = nx.average_clustering(ug)
            except Exception:
                clustering = 0.0
        else:
            # Fallback: approximate connectivity via BFS (pure PyTorch)
            reachable = torch.zeros(N, dtype=torch.bool)
            reachable[0] = True
            for _ in range(N):
                new = (binary[reachable].sum(dim=0) > 0)
                reachable = reachable | new
            is_connected = bool(reachable.all().item())

        return GraphProperties(
            n_nodes=N,
            n_edges=n_edges,
            density=density,
            avg_degree=avg_degree,
            is_connected=is_connected,
            diameter=diameter,
            clustering_coefficient=clustering,
        )

    # ------------------------------------------------------------------
    # Topology constructors
    # ------------------------------------------------------------------

    @staticmethod
    def hub_spoke_topology(n_agents: int) -> Tensor:
        """Create a hub-and-spoke adjacency matrix.

        Node 0 is the hub; all other nodes are spokes.  The hub has
        bidirectional edges to all spokes; spokes do **not** connect to each other.

        Args:
            n_agents: Total number of agents (including the hub).

        Returns:
            Float adjacency tensor of shape (n_agents, n_agents).

        Example::

            adj = GraphUtils.hub_spoke_topology(4)
            # Hub (node 0) <-> spoke 1, 2, 3; no spoke-to-spoke edges.
        """
        adj = torch.zeros(n_agents, n_agents)
        for i in range(1, n_agents):
            adj[0, i] = 1.0  # hub -> spoke
            adj[i, 0] = 1.0  # spoke -> hub
        logger.debug("hub_spoke_topology: n_agents=%d, edges=%d", n_agents, int(adj.sum()))
        return adj

    @staticmethod
    def fully_connected_topology(n_agents: int) -> Tensor:
        """Create a fully-connected (complete) adjacency matrix.

        All agents can communicate with all other agents.  Self-loops are
        excluded.

        Args:
            n_agents: Number of agents.

        Returns:
            Float adjacency tensor of shape (n_agents, n_agents) with 0 on
            diagonal and 1 everywhere else.
        """
        adj = torch.ones(n_agents, n_agents) - torch.eye(n_agents)
        return adj.float()

    @staticmethod
    def ring_topology(n_agents: int) -> Tensor:
        """Create a directed ring adjacency matrix (each agent -> next agent).

        Args:
            n_agents: Number of agents.

        Returns:
            Float adjacency tensor of shape (n_agents, n_agents) representing
            a unidirectional ring: 0->1->2->...->n_agents-1->0.
        """
        adj = torch.zeros(n_agents, n_agents)
        for i in range(n_agents):
            adj[i, (i + 1) % n_agents] = 1.0
        return adj

    @staticmethod
    def sample_random_erdos_renyi(
        n_agents: int,
        p: float,
        seed: Optional[int] = None,
    ) -> Tensor:
        """Sample an Erdős-Rényi random graph G(n, p).

        Each possible directed edge (excluding self-loops) is included
        independently with probability ``p``.

        Args:
            n_agents: Number of agents / nodes.
            p: Edge inclusion probability in [0, 1].
            seed: Optional random seed for reproducibility.

        Returns:
            Binary float adjacency tensor of shape (n_agents, n_agents).
        """
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        rand = torch.rand(n_agents, n_agents, generator=gen)
        adj = (rand < p).float()
        # Remove self-loops
        adj.fill_diagonal_(0.0)
        logger.debug(
            "Erdős-Rényi G(%d, %.2f): %d edges", n_agents, p, int(adj.sum())
        )
        return adj

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def plot_topology(
        adj: Tensor,
        agent_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        title: str = "Agent Communication Topology",
        node_color: str = "#4C72B0",
        edge_color: str = "#555555",
        figsize: Tuple[int, int] = (8, 8),
    ) -> None:
        """Visualise an agent communication graph using networkx and matplotlib.

        Draws a directed graph where each node is an agent and each directed
        edge represents an allowed communication channel.

        Args:
            adj: Binary adjacency tensor of shape (N, N).
            agent_names: List of N agent name strings.  Defaults to
                ``['Agent 0', 'Agent 1', ...]``.
            save_path: If provided, saves the figure to this path (PNG/PDF).
            title: Figure title.
            node_color: Matplotlib colour for nodes.
            edge_color: Matplotlib colour for edges.
            figsize: Figure size in inches.

        Raises:
            ImportError: If ``networkx`` or ``matplotlib`` is not installed.
        """
        if not _HAS_NX or not _HAS_MPL:
            raise ImportError(
                "Both 'networkx' and 'matplotlib' are required for plot_topology."
            )

        N = adj.shape[0]
        if agent_names is None:
            agent_names = [f"Agent {i}" for i in range(N)]

        edges = GraphUtils.adjacency_to_edge_list(adj)
        G_nx = nx.DiGraph()
        G_nx.add_nodes_from(range(N))
        G_nx.add_edges_from(edges)

        fig, ax = plt.subplots(figsize=figsize)
        pos = nx.spring_layout(G_nx, seed=42)
        labels = {i: name for i, name in enumerate(agent_names)}

        nx.draw_networkx_nodes(G_nx, pos, node_color=node_color, node_size=800, ax=ax)
        nx.draw_networkx_labels(G_nx, pos, labels=labels, font_size=9, ax=ax)
        nx.draw_networkx_edges(
            G_nx,
            pos,
            edge_color=edge_color,
            arrows=True,
            arrowsize=20,
            ax=ax,
            connectionstyle="arc3,rad=0.08",
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Topology plot saved to %s", save_path)

        plt.close(fig)
