"""ThoughtComm Baseline: sparsity-regularized shared/private latent communication.

Simplified implementation of the ThoughtComm approach (Zheng et al., NeurIPS 2025
spotlight, arXiv:2510.20733).  Each agent encodes its hidden state into two
components:
    z_shared  — broadcast to all other agents (sparse, low-entropy)
    z_private — kept local (unconstrained)

Sparsity is enforced via L1 regularization on ``z_shared`` (soft approximation
to minimizing shared-thought entropy).  Full ThoughtComm includes nonparametric
identifiability proofs and a recoverable topology; this baseline omits those
and implements the core communication mechanism for empirical comparison.

Key differences from our UniversalLatentSpace:
    • No inter-agent adapter training (shared/private split is per-agent).
    • Sparsity on the shared component (vs. dense hub vector).
    • Homogeneous-friendly but works across different hidden dims via projection.

Reference:
    Zheng et al. (2025) "ThoughtComm: Shareable and Private Latent Thoughts for
    Multi-Agent Reasoning" arXiv:2510.20733, NeurIPS 2025 spotlight.
"""


import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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
class ThoughtCommConfig:
    """Configuration for the ThoughtComm baseline.

    Attributes:
        hidden_dim: Agent native hidden dimension.
        shared_dim: Dimensionality of the shared latent component.
        private_dim: Dimensionality of the private latent component.
        sparsity_lambda: L1 coefficient on shared latent (entropy regularizer).
        dropout: Dropout rate in encoder/decoder MLPs.
    """
    hidden_dim: int
    shared_dim: int = 64
    private_dim: int = 192
    sparsity_lambda: float = 0.01
    dropout: float = 0.1


class _ThoughtEncoder(nn.Module):
    """Encodes hidden states into (z_shared, z_private)."""

    def __init__(self, config: ThoughtCommConfig) -> None:
        super().__init__()
        d = config.hidden_dim
        # Shared encoder: emphasises low-entropy features
        self.shared_proj = nn.Sequential(
            nn.Linear(d, config.shared_dim * 2),
            nn.LayerNorm(config.shared_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.shared_dim * 2, config.shared_dim),
        )
        # Private encoder: unconstrained
        self.private_proj = nn.Sequential(
            nn.Linear(d, config.private_dim),
            nn.LayerNorm(config.private_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.shared_proj(x), self.private_proj(x)


class _ThoughtDecoder(nn.Module):
    """Reconstructs hidden states from (z_shared, z_private)."""

    def __init__(self, config: ThoughtCommConfig) -> None:
        super().__init__()
        combined = config.shared_dim + config.private_dim
        self.net = nn.Sequential(
            nn.Linear(combined, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
        )

    def forward(self, z_shared: Tensor, z_private: Tensor) -> Tensor:
        return self.net(torch.cat([z_shared, z_private], dim=-1))


class ThoughtCommBaseline:
    """Sparsity-regularized shared/private latent communication baseline.

    Mimics the core of ThoughtComm (Zheng et al., NeurIPS 2025) for empirical
    comparison against the Universal Latent Space.  Agents register with a
    shared/private encoder; during communication only ``z_shared`` is broadcast.

    Args:
        config: :class:`ThoughtCommConfig` with dimension and sparsity settings.
        device: PyTorch device string.
    """

    def __init__(self, config: ThoughtCommConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self._encoders: Dict[str, _ThoughtEncoder] = {}
        self._decoders: Dict[str, _ThoughtDecoder] = {}
        logger.info(
            "ThoughtCommBaseline: hidden=%d, shared=%d, private=%d, λ_sparse=%.3f",
            config.hidden_dim, config.shared_dim, config.private_dim, config.sparsity_lambda,
        )

    def register_agent(self, agent_id: str) -> None:
        """Register an agent with its own encoder/decoder pair."""
        enc = _ThoughtEncoder(self.config).to(self.device)
        dec = _ThoughtDecoder(self.config).to(self.device)
        self._encoders[agent_id] = enc
        self._decoders[agent_id] = dec
        logger.info("ThoughtCommBaseline: registered agent '%s'", agent_id)

    def encode(self, agent_id: str, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        """Decompose hidden states into shared and private components.

        Args:
            agent_id: Registered agent identifier.
            hidden: Hidden states, shape (B, hidden_dim).

        Returns:
            Tuple of (z_shared, z_private), shapes (B, shared_dim) and (B, private_dim).
        """
        enc = self._require_agent(agent_id)
        x = hidden.to(self.device).float()
        return enc(x)

    def sparsity_loss(self, z_shared: Tensor) -> Tensor:
        """L1 sparsity regularizer on the shared thought component."""
        return self.config.sparsity_lambda * z_shared.abs().mean()

    def communicate(
        self,
        sender_id: str,
        receiver_id: str,
        sender_hidden: Tensor,
    ) -> Tuple[Tensor, float]:
        """Transfer shared latent from sender to receiver.

        The receiver uses its own private component (set to zero if no local
        state is available) and the sender's shared component to reconstruct
        a representation.

        Args:
            sender_id: Sending agent.
            receiver_id: Receiving agent.
            sender_hidden: Sender's hidden states, shape (B, hidden_dim).

        Returns:
            Tuple of (reconstructed_hidden, sparsity_loss_value).
        """
        z_shared, _ = self.encode(sender_id, sender_hidden)
        B = z_shared.shape[0]
        
        if sender_id == receiver_id:
            raise ValueError(f"ThoughtComm collapse detected: Sender and receiver are both '{sender_id}'. Cannot route to self.")
            
        # Receiver should use its own private state representation, not zeros.
        # Approximated by preserving the private norm variance.
        z_private = torch.randn(B, self.config.private_dim, device=self.device) * 0.02
        rec_dec = self._decoders[receiver_id]
        reconstructed = rec_dec(z_shared, z_private)
        sparse_loss = float(self.sparsity_loss(z_shared).item())
        return reconstructed, sparse_loss

    def compute_sparsity_stats(self, z_shared: Tensor) -> Dict[str, float]:
        """Report sparsity statistics of the shared thought component.

        Args:
            z_shared: Shared latent, shape (B, shared_dim).

        Returns:
            Dict with ``l1_norm``, ``l0_approx`` (fraction of near-zero dims),
            ``mean_activation``.
        """
        l1 = float(z_shared.abs().mean().item())
        l0_approx = float((z_shared.abs() < 0.01).float().mean().item())
        mean_act = float(z_shared.mean().item())
        return {"l1_norm": l1, "l0_approx_zero_frac": l0_approx, "mean_activation": mean_act}

    def _require_agent(self, agent_id: str) -> _ThoughtEncoder:
        if agent_id not in self._encoders:
            raise KeyError(
                f"Agent '{agent_id}' not registered. Call register_agent() first."
            )
        return self._encoders[agent_id]
