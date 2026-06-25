"""G-Designer and MasRouter Baselines: variational topology + cascaded routing.

This module provides simplified baselines for two direct topology/routing competitors:

**G-Designer** (Li et al., ICML 2025, arXiv:2410.11782):
    Uses a Variational Graph AutoEncoder (VGAE) with task-specific conditioning to
    decode query-adaptive communication topologies.  Reports 84.50% MMLU accuracy
    and 95.33% token reduction on HumanEval.  Key differences from our CVAE:
    • GNN-based encoder (not MLP) — graph-structure aware.
    • Node-level representations (not matrix-level adjacency).
    • Virtual task node injected into agent graph.
    Our simplified version uses MLP encoders (same as CVAE) but produces
    an inner-product decoded adjacency (VGAE style) rather than a direct MLP.

**MasRouter** (Wang et al., ACL 2025):
    Cascaded controller: first chooses collaboration mode (solo / pipeline /
    debate), then allocates roles, then selects LLMs per role.  Uses a
    variational latent-variable model for mode determination.  Key differences
    from our AttentionRouter:
    • Discrete collaboration mode (3-way choice) before role assignment.
    • Role allocation conditioned on chosen mode.
    Our simplified version implements the two-stage cascade logic.

References:
    Li et al. (2025) "G-Designer: VGAE-based Multi-Agent Topology Design"
        arXiv:2410.11782, ICML 2025.
    Wang et al. (2025) "MasRouter: Learning to Route LLMs in Multi-Agent Systems"
        ACL 2025.
"""

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# G-Designer simplified baseline
# ---------------------------------------------------------------------------

class _VGAEEncoder(nn.Module):
    """MLP-based VGAE encoder: adjacency + task query → (mu, logvar) per node."""

    def __init__(self, node_in_dim: int, query_dim: int, hidden_dim: int, z_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_in_dim + query_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu_proj = nn.Linear(hidden_dim, z_dim)
        self.lv_proj = nn.Linear(hidden_dim, z_dim)

    def forward(self, node_feats: Tensor, query_emb: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            node_feats: (B, N, node_in_dim)
            query_emb:  (B, query_dim)

        Returns:
            (mu, logvar) each (B, N, z_dim)
        """
        B, N, _ = node_feats.shape
        q = query_emb.unsqueeze(1).expand(B, N, -1)  # (B, N, query_dim)
        inp = torch.cat([node_feats, q], dim=-1)
        h = self.net(inp)
        return self.mu_proj(h), self.lv_proj(h)


class GDesignerBaseline:
    """Simplified G-Designer: VGAE with inner-product decoder for topology.

    Produces query-conditioned communication topologies.  The MLP node encoder
    is a simplification of G-Designer's GNN; the inner-product adjacency decoder
    is faithful to the VGAE formulation.

    Args:
        max_n_agents: Maximum agent count (adjacency matrix size).
        query_dim: Dimension of task query embedding.
        z_dim: VGAE latent dimension per node.
        hidden_dim: MLP hidden dimension.
        device: PyTorch device string.
    """

    def __init__(
        self,
        max_n_agents: int = 8,
        query_dim: int = 128,
        z_dim: int = 32,
        hidden_dim: int = 128,
        device: str = "cpu",
    ) -> None:
        self.max_n_agents = max_n_agents
        self.z_dim = z_dim
        self.device = torch.device(device)
        # Node features: one-hot agent ID (max_n_agents dims) + task virtual node
        node_in_dim = max_n_agents
        self.encoder = _VGAEEncoder(node_in_dim, query_dim, hidden_dim, z_dim).to(self.device)
        logger.info(
            "GDesignerBaseline: N=%d, z_dim=%d, query_dim=%d", max_n_agents, z_dim, query_dim
        )

    def _node_features(self, batch_size: int) -> Tensor:
        """One-hot node features for all agents, shape (B, N, N)."""
        feats = torch.eye(self.max_n_agents, device=self.device)
        return feats.unsqueeze(0).expand(batch_size, -1, -1)

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.encoder.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def sample_topology(
        self, query_emb: Tensor, threshold: float = 0.5
    ) -> Tuple[Tensor, Tensor]:
        """Sample a communication topology conditioned on the task query.

        Uses inner-product decoding (VGAE-style): P(A_ij=1) = σ(z_i · z_j).

        Args:
            query_emb: Task embedding, shape (B, query_dim).
            threshold: Probability threshold for binary adjacency.

        Returns:
            Tuple of (binary_adjacency, edge_probs), each shape (B, N, N).
        """
        B = query_emb.shape[0]
        node_feats = self._node_features(B)
        mu, logvar = self.encoder(node_feats, query_emb.to(self.device))
        z = self._reparameterize(mu, logvar)  # (B, N, z_dim)
        # Inner-product decoder: P(A_ij) = σ(z_i · z_j)
        edge_logits = torch.bmm(z, z.transpose(1, 2))  # (B, N, N)
        edge_probs = torch.sigmoid(edge_logits)
        binary_adj = (edge_probs > threshold).float()
        return binary_adj, edge_probs

    def kl_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Standard KL divergence from N(0,I) prior."""
        return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()


# ---------------------------------------------------------------------------
# MasRouter simplified baseline
# ---------------------------------------------------------------------------

COLLAB_MODES = ["solo", "pipeline", "debate"]


@dataclass
class MasRouterConfig:
    """Configuration for MasRouter baseline.

    Attributes:
        query_dim: Task query embedding dimension.
        mode_latent_dim: Latent dimension for collaboration mode VAE.
        n_roles: Maximum number of agent roles.
        hidden_dim: MLP hidden dimension.
    """
    query_dim: int = 128
    mode_latent_dim: int = 16
    n_roles: int = 4
    hidden_dim: int = 64


class _ModeController(nn.Module):
    """Variational mode selector: query -> collaboration mode distribution."""

    def __init__(self, config: MasRouterConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.query_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, len(COLLAB_MODES)),
        )

    def forward(self, query_emb: Tensor) -> Tensor:
        """Returns log-softmax over collaboration modes."""
        return F.log_softmax(self.net(query_emb), dim=-1)


class _RoleAllocator(nn.Module):
    """Assigns role weights conditioned on (query, mode)."""

    def __init__(self, config: MasRouterConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.query_dim + len(COLLAB_MODES), config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.n_roles),
        )

    def forward(self, query_emb: Tensor, mode_logits: Tensor) -> Tensor:
        """Returns soft role weights."""
        inp = torch.cat([query_emb, mode_logits.exp()], dim=-1)
        return F.softmax(self.net(inp), dim=-1)


class MasRouterBaseline:
    """Cascaded collaboration mode + role allocation router (MasRouter-style).

    Two-stage dispatch:
        1. **Mode controller** — chooses solo / pipeline / debate.
        2. **Role allocator** — allocates agent roles conditioned on chosen mode.

    This differs from the paper's AttentionRouter (which is single-stage,
    soft over roles only).  MasRouter's mode-then-role cascade is a stronger
    routing baseline.

    Args:
        config: :class:`MasRouterConfig`.
        device: PyTorch device string.
    """

    def __init__(self, config: MasRouterConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.mode_ctrl = _ModeController(config).to(self.device)
        self.role_alloc = _RoleAllocator(config).to(self.device)
        self._roles: List[str] = []
        logger.info("MasRouterBaseline initialized (modes=%s)", COLLAB_MODES)

    def set_roles(self, roles: List[str]) -> None:
        """Set the ordered list of agent roles this router dispatches to."""
        if len(roles) > self.config.n_roles:
            raise ValueError(
                f"Too many roles: {len(roles)} > config.n_roles={self.config.n_roles}"
            )
        self._roles = roles

    def route(
        self, query_emb: Tensor, threshold: float = 0.1
    ) -> Dict[str, object]:
        """Two-stage route: choose mode, then allocate roles.

        Args:
            query_emb: Task query embedding, shape (1, query_dim) or (query_dim,).
            threshold: Minimum role weight to include a role in the plan.

        Returns:
            Dict with ``collab_mode`` (str), ``role_weights`` (Tensor),
            ``selected_roles`` (List[str]), ``mode_probs`` (Tensor).
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        q = query_emb.to(self.device)

        with torch.no_grad():
            mode_log_probs = self.mode_ctrl(q)  # (1, 3)
            mode_idx = int(mode_log_probs.argmax(dim=-1).item())
            chosen_mode = COLLAB_MODES[mode_idx]

            role_weights = self.role_alloc(q, mode_log_probs)  # (1, n_roles)
            role_weights = role_weights.squeeze(0)

        n = len(self._roles)
        selected = [
            self._roles[i]
            for i in range(min(n, role_weights.shape[0]))
            if float(role_weights[i].item()) >= threshold
        ]
        if not selected and self._roles:
            selected = [self._roles[int(role_weights[:n].argmax().item())]]

        return {
            "collab_mode": chosen_mode,
            "role_weights": role_weights,
            "selected_roles": selected,
            "mode_probs": mode_log_probs.exp().squeeze(0),
        }
