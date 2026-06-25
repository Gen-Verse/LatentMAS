"""Cache-to-Cache Baseline: KV-cache projection with learnable gating.

Simplified implementation of the Cache-to-Cache (C2C) approach (Fu et al.,
arXiv:2510.03215).  Each agent projects its KV-cache vectors into a compatible
format for the receiver and fuses them via a learnable gate.

C2C is a *pairwise* (O(N²)) approach — each sender–receiver pair has a
dedicated projection.  This baseline demonstrates that O(N²) pairwise latent
communication exists as an alternative; the paper's O(N) hub-and-spoke design
should compare against it to show the scaling advantage.

Key differences from UniversalLatentSpace:
    • Pairwise projections (O(N²) params and O(N²) transfers per round).
    • Operates on KV-cache pairs (key + value) rather than last-layer hidden.
    • Learnable scalar gate per (sender, receiver) pair.

Reference:
    Fu et al. (2025) "Cache-to-Cache: Efficient KV-Cache Sharing for
    Heterogeneous Multi-Agent Systems" arXiv:2510.03215.
"""

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class _KVProjection(nn.Module):
    """Projects (key, value) pairs from sender dim to receiver dim with a gate."""

    def __init__(self, sender_dim: int, receiver_dim: int) -> None:
        super().__init__()
        self.key_proj = nn.Linear(sender_dim, receiver_dim, bias=False)
        self.val_proj = nn.Linear(sender_dim, receiver_dim, bias=False)
        # Learnable gate (initialised near 0 → minimal fusion at start)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self, sender_k: Tensor, sender_v: Tensor,
        receiver_k: Tensor, receiver_v: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Fuse projected sender KV with receiver KV via learned gate.

        Args:
            sender_k: Sender key cache, shape (B, L_s, D_s).
            sender_v: Sender value cache, shape (B, L_s, D_s).
            receiver_k: Receiver key cache, shape (B, L_r, D_r).
            receiver_v: Receiver value cache, shape (B, L_r, D_r).

        Returns:
            Tuple of fused (key, value) caches, each shape (B, L_r, D_r).
        """
        g = torch.sigmoid(self.gate)
        # Project sender → receiver dim (pool sender sequence to receiver length via mean)
        s_k = self.key_proj(sender_k.float().mean(dim=1, keepdim=True))  # (B, 1, D_r)
        s_v = self.val_proj(sender_v.float().mean(dim=1, keepdim=True))  # (B, 1, D_r)
        # Fuse: gate controls how much sender context to blend in
        fused_k = receiver_k + g * s_k.expand_as(receiver_k)
        fused_v = receiver_v + g * s_v.expand_as(receiver_v)
        return fused_k, fused_v


class CacheToCacheBaseline:
    """Pairwise KV-cache projection for cross-agent context sharing.

    Each registered (sender, receiver) pair gets a dedicated ``_KVProjection``
    module.  Communication is O(N²) because each sender–receiver pair needs its
    own projection parameters.

    Args:
        device: PyTorch device string.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self._agent_dims: Dict[str, int] = {}
        self._projections: Dict[Tuple[str, str], _KVProjection] = {}
        logger.info("CacheToCacheBaseline initialized (O(N²) pairwise)")

    def register_agent(self, agent_id: str, kv_dim: int) -> None:
        """Register an agent's KV-cache dimension.

        Args:
            agent_id: Unique agent identifier.
            kv_dim: Dimension of the agent's KV cache vectors.
        """
        self._agent_dims[agent_id] = kv_dim
        # Build pairwise projections for existing agents
        for other_id, other_dim in self._agent_dims.items():
            if other_id == agent_id:
                continue
            # sender=agent_id -> receiver=other_id
            key_fwd = (agent_id, other_id)
            if key_fwd not in self._projections:
                self._projections[key_fwd] = _KVProjection(kv_dim, other_dim).to(self.device)
            # sender=other_id -> receiver=agent_id
            key_bwd = (other_id, agent_id)
            if key_bwd not in self._projections:
                self._projections[key_bwd] = _KVProjection(other_dim, kv_dim).to(self.device)
        logger.info("CacheToCacheBaseline: registered '%s' (kv_dim=%d)", agent_id, kv_dim)

    def fuse_kv(
        self,
        sender_id: str,
        receiver_id: str,
        sender_k: Tensor,
        sender_v: Tensor,
        receiver_k: Tensor,
        receiver_v: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Fuse sender KV cache into receiver's KV cache via pairwise projection.

        Args:
            sender_id: Agent providing context.
            receiver_id: Agent receiving and integrating context.
            sender_k: Sender key cache, shape (B, L_s, D_s).
            sender_v: Sender value cache, shape (B, L_s, D_s).
            receiver_k: Receiver key cache, shape (B, L_r, D_r).
            receiver_v: Receiver value cache, shape (B, L_r, D_r).

        Returns:
            Tuple of fused (key, value) caches, shape (B, L_r, D_r).
        """
        proj = self._get_projection(sender_id, receiver_id)
        return proj(
            sender_k.to(self.device), sender_v.to(self.device),
            receiver_k.to(self.device), receiver_v.to(self.device),
        )

    def communication_complexity(self, n_agents: int) -> Dict[str, int]:
        """Report pairwise O(N²) complexity vs hub-and-spoke O(N).

        Args:
            n_agents: Number of registered agents.

        Returns:
            Dict comparing communication costs.
        """
        return {
            "n_agents": n_agents,
            "c2c_pairwise_projections": n_agents * (n_agents - 1),
            "hub_spoke_projections": 2 * n_agents,  # encoder + decoder per agent
            "c2c_transfers_per_round": n_agents * (n_agents - 1),
            "hub_spoke_transfers_per_round": 2 * n_agents,
            "complexity_ratio_params": n_agents * (n_agents - 1) // max(2 * n_agents, 1),
        }

    def _get_projection(self, sender_id: str, receiver_id: str) -> _KVProjection:
        key = (sender_id, receiver_id)
        if key not in self._projections:
            raise KeyError(
                f"No projection registered for ({sender_id}, {receiver_id}). "
                f"Call register_agent() for both before calling fuse_kv()."
            )
        return self._projections[key]
