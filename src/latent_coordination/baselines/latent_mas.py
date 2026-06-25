"""LatentMAS Baseline: training-free last-layer hidden-state sharing.

Models the LatentMAS (Zou et al., arXiv:2511.20639; ICML 2026) approach:
agents share last-layer hidden states directly with no learned adapter,
relying on identical model architecture (homogeneous-only).

This baseline is used to:
    (a) Compare against our trained adapter to show adapter quality gains.
    (b) Validate the claim that LatentMAS is unstable on heterogeneous backbones
        (since without an adapter, mismatched hidden dims cause shape errors).
"""

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class LatentMASBaseline:
    """Training-free last-layer hidden-state transfer between homogeneous agents.

    For heterogeneous agents (different hidden dims), this baseline will raise
    a ``ValueError`` — which is by design and matches LatentMAS's documented
    limitation.

    Args:
        hidden_dim: Shared hidden dimension (all agents must match).
        device: PyTorch device string.
    """

    def __init__(self, hidden_dim: int, device: str = "cpu") -> None:
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self._kv_memory: Optional[Tensor] = None  # shared KV-cache working memory
        logger.info("LatentMASBaseline: hidden_dim=%d, device=%s", hidden_dim, device)

    def share_hidden_state(
        self,
        sender_hidden: Tensor,
        receiver_hidden_dim: int,
    ) -> Tensor:
        """Pass last-layer hidden states without any learned projection.

        Args:
            sender_hidden: Sender's last-layer hidden states, shape (B, D).
            receiver_hidden_dim: Receiver's expected hidden dim.

        Returns:
            Unmodified sender hidden states (assumes homogeneous architecture).

        Raises:
            ValueError: If sender and receiver hidden dims mismatch.
        """
        if sender_hidden.shape[-1] != receiver_hidden_dim:
            raise ValueError(
                f"LatentMASBaseline requires homogeneous hidden dims. "
                f"Sender has {sender_hidden.shape[-1]}, receiver expects {receiver_hidden_dim}. "
                f"Use UniversalLatentSpace for heterogeneous agents."
            )
        return sender_hidden.to(self.device)

    def update_kv_memory(self, hidden_states: Tensor) -> None:
        """Update the shared KV-cache working memory (running mean of states).

        Args:
            hidden_states: New hidden states to accumulate, shape (B, D).
        """
        hs = hidden_states.float().to(self.device)
        if self._kv_memory is None:
            self._kv_memory = hs.mean(dim=0, keepdim=True)
        else:
            self._kv_memory = 0.9 * self._kv_memory + 0.1 * hs.mean(dim=0, keepdim=True)

    def get_kv_memory(self) -> Optional[Tensor]:
        """Return the current KV working memory state."""
        return self._kv_memory

    def broadcast(
        self,
        sender_hidden: Tensor,
        n_receivers: int,
    ) -> Dict[int, Tensor]:
        """Broadcast sender states to all receivers (no adapter overhead).

        Args:
            sender_hidden: Sender's hidden states, shape (B, D).
            n_receivers: Number of receiver agents.

        Returns:
            Dict mapping receiver index -> hidden state tensor.
        """
        result = {}
        for i in range(n_receivers):
            result[i] = sender_hidden.to(self.device)
        return result

    def compute_fidelity(
        self,
        original: Tensor,
        received: Tensor,
    ) -> Dict[str, float]:
        """Compute transfer fidelity metrics (trivially perfect for this baseline).

        Args:
            original: Original sender states.
            received: Received states (same as original in this baseline).

        Returns:
            Dict with ``cosine_similarity`` and ``mse``.
        """
        o = original.float()
        r = received.float()
        cos = float(F.cosine_similarity(o, r, dim=-1).mean().item())
        mse = float(F.mse_loss(r, o).item())
        return {"cosine_similarity": cos, "mse": mse}
