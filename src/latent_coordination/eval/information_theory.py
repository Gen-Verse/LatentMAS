"""Information-theoretic analysis tools for the Universal Latent Space.

Provides:
    effective_rank   — information-theoretic rank proxy (Roy & Vetterli 2007)
    hsic_mi_proxy    — HSIC-based mutual information proxy (no MINE training)
    compression_ratio — latent bytes vs text-token bytes
    breakeven_n      — N at which latent communication is cheaper
    InfoTheoreticAnalyzer — unified class wrapping all metrics

These tools directly address the audit's highest-priority gap: converting the
"latent is richer than text" claim from hand-wavy to quantitatively grounded.
"""

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Effective rank
# ---------------------------------------------------------------------------

def effective_rank(matrix: Tensor) -> float:
    """Compute effective rank via the entropy of the singular value distribution.

    ``eff_rank = exp(H(p))`` where ``p_i = σ_i / Σ σ_j`` and H is Shannon
    entropy in nats (Roy & Vetterli 2007, IEEE Trans. Signal Processing).

    A full-rank matrix returns a value close to min(rows, cols); a degenerate
    matrix returns a value close to 1.

    Args:
        matrix: 2-D tensor of shape (M, N).

    Returns:
        Float effective rank in [1, min(M, N)].
    """
    if matrix.dim() != 2:
        raise ValueError(f"effective_rank expects a 2-D tensor, got shape {tuple(matrix.shape)}")
    sv = torch.linalg.svdvals(matrix.float())
    sv_sum = sv.sum().clamp(min=1e-12)
    p = sv / sv_sum
    h = -(p * (p + 1e-12).log()).sum().item()
    return float(math.exp(h))


# ---------------------------------------------------------------------------
# HSIC-based MI proxy
# ---------------------------------------------------------------------------

def hsic_mi_proxy(
    X: Tensor,
    Y: Tensor,
    kernel: str = "linear",
) -> float:
    """Hilbert-Schmidt Independence Criterion as a mutual-information proxy.

    HSIC(X, Y) = 0 iff X and Y are independent (for characteristic kernels).
    We use the unbiased estimator from Song et al. (2012).  For the linear
    kernel this reduces to ``Tr(K_X H K_Y H) / (n-1)^2`` which is fast and
    requires no training (unlike MINE).

    Args:
        X: Tensor of shape (N, D_x).
        Y: Tensor of shape (N, D_y).
        kernel: Currently only ``'linear'`` is supported.

    Returns:
        Float HSIC value (higher = more dependence).
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")
    n = X.shape[0]
    if n < 2:
        return float("nan")

    X = X.float()
    Y = Y.float()

    H = torch.eye(n, device=X.device) - torch.ones(n, n, device=X.device) / n
    Kx = X @ X.T  # linear kernel
    Ky = Y @ Y.T
    hsic = float(torch.trace(Kx @ H @ Ky @ H).item()) / max((n - 1) ** 2, 1)
    return hsic


# ---------------------------------------------------------------------------
# Compression ratio
# ---------------------------------------------------------------------------

def compression_ratio(
    latent_bytes: float,
    text_tokens: int,
    bytes_per_token: float = 4.0,
) -> float:
    """Ratio of text-serialized bytes to latent-channel bytes.

    Values > 1.0 indicate the latent channel is more compact than text.

    Args:
        latent_bytes: Total bytes transmitted via the latent channel.
        text_tokens: Number of tokens in the equivalent text message.
        bytes_per_token: Average bytes per token (default 4 = ~2 UTF-8 chars/token).

    Returns:
        Float compression ratio (text_bytes / latent_bytes).
    """
    text_bytes = text_tokens * bytes_per_token
    if latent_bytes <= 0:
        return float("inf")
    return text_bytes / latent_bytes


# ---------------------------------------------------------------------------
# Break-even analysis
# ---------------------------------------------------------------------------

def breakeven_n(
    adapter_forward_ms: float,
    token_gen_ms_per_token: float,
    avg_msg_len_tokens: int,
) -> float:
    """Return the N at which latent communication becomes cheaper than text.

    Each text transfer generates ``avg_msg_len_tokens`` tokens (cost:
    ``avg_msg_len_tokens * token_gen_ms_per_token``).  A latent transfer pays
    ``adapter_forward_ms`` regardless of message length.  The crossover N
    satisfies:

        N * adapter_forward_ms = N * (N-1) * avg_msg_len_tokens * token_gen_ms

    Solving: N_breakeven = adapter_forward_ms / (avg_msg_len_tokens * token_gen_ms_per_token) + 1

    Args:
        adapter_forward_ms: Wall-clock time for one adapter encode+decode (ms).
        token_gen_ms_per_token: Wall-clock time to generate one text token (ms).
        avg_msg_len_tokens: Average inter-agent message length in tokens.

    Returns:
        Float N_breakeven (fractional; round up for a practical threshold).
    """
    denominator = avg_msg_len_tokens * token_gen_ms_per_token
    if denominator <= 0:
        return float("inf")
    return adapter_forward_ms / denominator + 1.0


# ---------------------------------------------------------------------------
# Unified analyzer
# ---------------------------------------------------------------------------

class InfoTheoreticAnalyzer:
    """Unified information-theoretic analysis for the Universal Latent Space.

    Runs all metrics in one pass and returns a structured report.

    Args:
        hub_dim: Expected hub dimensionality (for logging).
    """

    def __init__(self, hub_dim: int = 512) -> None:
        self.hub_dim = hub_dim

    def analyze(
        self,
        uls,
        hidden_states: Tensor,
        agent_id: str,
        text_tokens_equivalent: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run all information-theoretic metrics for a single agent's hub mapping.

        Args:
            uls: Registered :class:`UniversalLatentSpace`.
            hidden_states: Sample hidden states, shape (B, D_agent).
            agent_id: Registered agent to analyse.
            text_tokens_equivalent: Number of tokens the message would occupy as
                text (used for compression ratio; skipped if None).

        Returns:
            Dict with ``effective_rank_hub``, ``effective_rank_original``,
            ``hsic_mi_proxy``, ``compression_ratio`` (if applicable),
            ``norm_ratio``, ``cosine_similarity``.
        """
        with torch.no_grad():
            x = hidden_states.float()
            u = uls.encode(agent_id, x)
            recon = uls.decode(agent_id, u)

            eff_rank_hub = effective_rank(u)
            eff_rank_orig = effective_rank(x)
            hsic = hsic_mi_proxy(u, x)
            cos_sim = float(F.cosine_similarity(x, recon, dim=-1).mean().item())
            norm_ratio = float((recon.norm(dim=-1) / (x.norm(dim=-1) + 1e-8)).mean().item())

        results: Dict[str, float] = {
            "effective_rank_hub": eff_rank_hub,
            "effective_rank_original": eff_rank_orig,
            "effective_rank_ratio": eff_rank_hub / max(eff_rank_orig, 1e-8),
            "hsic_mi_proxy": hsic,
            "cosine_similarity": cos_sim,
            "norm_ratio": norm_ratio,
        }

        if text_tokens_equivalent is not None:
            latent_bytes = float(u.numel() * 4)  # float32
            results["compression_ratio"] = compression_ratio(latent_bytes, text_tokens_equivalent)

        logger.info("InfoTheoreticAnalyzer[%s]: %s", agent_id, results)
        return results

    def breakeven_report(
        self,
        adapter_forward_ms: float,
        token_gen_ms_per_token: float,
        msg_len_sweep: List[int],
    ) -> Dict[str, object]:
        """Sweep message lengths and return breakeven N for each.

        Args:
            adapter_forward_ms: Measured adapter forward pass latency (ms).
            token_gen_ms_per_token: Measured token generation latency per token (ms).
            msg_len_sweep: List of message lengths (in tokens) to evaluate.

        Returns:
            Dict with ``adapter_forward_ms``, ``token_gen_ms_per_token``, and
            ``breakeven_by_msg_len`` mapping each msg_len to its breakeven N.
        """
        return {
            "adapter_forward_ms": adapter_forward_ms,
            "token_gen_ms_per_token": token_gen_ms_per_token,
            "breakeven_by_msg_len": {
                msg_len: breakeven_n(adapter_forward_ms, token_gen_ms_per_token, msg_len)
                for msg_len in msg_len_sweep
            },
        }
