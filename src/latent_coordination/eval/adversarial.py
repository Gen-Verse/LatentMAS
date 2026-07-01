"""Adversarial robustness evaluation for the Universal Latent Space channel.

Implements an agent-in-the-middle threat model (He et al., arXiv:2502.14847)
adapted to the hub latent channel: an adversary perturbs hub vectors within an
L2 ε-ball before they reach the receiver.  A NormMatch-based gate defense is
also provided.

References:
    He et al. (2025) "Agent-in-the-Middle Attacks" arXiv:2502.14847
    Vision Wormhole NormMatch (arXiv:2602.15382) for the gate defense design.
"""


import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from latent_coordination.latent_space.universal_space import UniversalLatentSpace

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bounded L2 attacker
# ---------------------------------------------------------------------------

class BoundedLatentAttacker:
    """Simulates an agent-in-the-middle adversary that perturbs hub vectors.

    The attacker intercepts the universal-space vector after encoding and
    adds adversarial noise bounded by an L2 ε-ball, then passes the
    corrupted vector to the receiver's decoder.

    Args:
        epsilon: L2 perturbation budget (absolute, in hub-vector norm units).
        seed: Random seed for reproducibility.
    """

    def __init__(self, epsilon: float = 0.1, seed: int = 0) -> None:
        self.epsilon = epsilon
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def perturb(self, hub_vector: Tensor) -> Tensor:
        """Add bounded random noise to a hub vector.

        Args:
            hub_vector: Universal-space tensor, shape (B, D) or (D,).

        Returns:
            Perturbed tensor of the same shape, with ‖δ‖₂ ≤ ε per sample.
        """
        delta = torch.randn_like(hub_vector, generator=self._rng)
        # Normalize to unit ball then scale by epsilon
        delta = delta / (delta.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        delta = delta * self.epsilon
        return hub_vector + delta

    def attack_transfer(
        self,
        uls: UniversalLatentSpace,
        sender_id: str,
        receiver_id: str,
        hidden_states: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Transfer with adversarial perturbation injected in hub space.

        Args:
            uls: Registered UniversalLatentSpace.
            sender_id: Sender agent ID.
            receiver_id: Receiver agent ID.
            hidden_states: Sender hidden states, shape (B, D_sender).

        Returns:
            Tuple of (clean_received, attacked_received) tensors of shape (B, D_receiver).
        """
        with torch.no_grad():
            hub = uls.encode(sender_id, hidden_states)
            clean = uls.decode(receiver_id, hub)
            attacked_hub = self.perturb(hub)
            attacked = uls.decode(receiver_id, attacked_hub)
        return clean, attacked


# ---------------------------------------------------------------------------
# NormMatch gate defense
# ---------------------------------------------------------------------------

class LatentGateDefense:
    """Gated norm-based defense against off-manifold hub injections.

    Rejects or clamps hub vectors whose RMS norm falls outside a learned
    [rms_lo, rms_hi] band, inspired by VW's NormMatch stabilizer.

    Args:
        rms_lo: Lower RMS threshold for acceptance.
        rms_hi: Upper RMS threshold for acceptance.
        clamp: If True, clamp instead of zeroing out-of-band vectors.
    """

    def __init__(
        self,
        rms_lo: float = 0.1,
        rms_hi: float = 10.0,
        clamp: bool = True,
    ) -> None:
        self.rms_lo = rms_lo
        self.rms_hi = rms_hi
        self.clamp = clamp

    def _rms(self, x: Tensor) -> Tensor:
        return x.norm(dim=-1) / (x.shape[-1] ** 0.5)

    def filter(self, hub_vector: Tensor) -> Tuple[Tensor, Tensor]:
        """Filter/clamp hub vectors outside the allowed norm range.

        Args:
            hub_vector: Tensor of shape (B, D).

        Returns:
            Tuple of (filtered_vector, mask) where mask is True for accepted rows.
        """
        rms = self._rms(hub_vector)  # (B,)
        in_band = (rms >= self.rms_lo) & (rms <= self.rms_hi)  # (B,)

        if self.clamp:
            rms_clamped = rms.clamp(self.rms_lo, self.rms_hi)
            scale = rms_clamped / rms.clamp(min=1e-8)
            filtered = hub_vector * scale.unsqueeze(-1)
        else:
            filtered = hub_vector * in_band.float().unsqueeze(-1)

        return filtered, in_band


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def run_adversarial_eval(
    uls: UniversalLatentSpace,
    agent_pairs: List[Tuple[str, str]],
    hidden_states: Tensor,
    epsilons: List[float],
    use_defense: bool = True,
    defense_rms_lo: float = 0.1,
    defense_rms_hi: float = 10.0,
) -> Dict[str, object]:
    """Evaluate adversarial robustness of the latent channel at multiple ε values.

    For each ε:
        1. Perturb hub vectors with bounded L2 noise.
        2. Optionally apply the NormMatch gate defense.
        3. Measure fidelity degradation (cosine similarity drop).

    Args:
        uls: Configured UniversalLatentSpace with registered agents.
        agent_pairs: List of (sender_id, receiver_id) tuples to evaluate.
        hidden_states: Sample hidden states to transfer, shape (B, D).
        epsilons: List of ε values to sweep.
        use_defense: Whether to apply the LatentGateDefense.
        defense_rms_lo: Lower gate threshold.
        defense_rms_hi: Upper gate threshold.

    Returns:
        Dict with per-ε results including ``cosine_similarity_clean``,
        ``cosine_similarity_attacked``, ``cosine_similarity_defended``, and
        ``defense_acceptance_rate``.
    """
    defense = LatentGateDefense(rms_lo=defense_rms_lo, rms_hi=defense_rms_hi)
    results: Dict[str, object] = {"epsilons": epsilons, "pairs": [], "per_epsilon": {}}

    for eps in epsilons:
        attacker = BoundedLatentAttacker(epsilon=eps)
        eps_results: Dict[str, List[float]] = {
            "cosine_clean": [],
            "cosine_attacked": [],
            "cosine_defended": [],
            "acceptance_rate": [],
        }

        for sender_id, receiver_id in agent_pairs:
            try:
                clean, attacked = attacker.attack_transfer(
                    uls, sender_id, receiver_id, hidden_states
                )
                # Fidelity: cosine similarity between clean and attacked/defended
                cos_clean = float(F.cosine_similarity(clean, clean, dim=-1).mean().item())
                cos_atk = float(F.cosine_similarity(clean, attacked, dim=-1).mean().item())

                if use_defense:
                    with torch.no_grad():
                        atk_hub = uls.encode(sender_id, hidden_states)
                        atk_hub_perturbed = attacker.perturb(atk_hub)
                        defended_hub, mask = defense.filter(atk_hub_perturbed)
                        defended = uls.decode(receiver_id, defended_hub)
                    cos_def = float(F.cosine_similarity(clean, defended, dim=-1).mean().item())
                    accept_rate = float(mask.float().mean().item())
                else:
                    cos_def = cos_atk
                    accept_rate = 1.0

                eps_results["cosine_clean"].append(cos_clean)
                eps_results["cosine_attacked"].append(cos_atk)
                eps_results["cosine_defended"].append(cos_def)
                eps_results["acceptance_rate"].append(accept_rate)

            except Exception as exc:
                logger.warning("Adversarial eval failed for pair (%s, %s): %s", sender_id, receiver_id, exc)

        import numpy as np
        results["per_epsilon"][eps] = {
            k: float(np.mean(v)) if v else float("nan")
            for k, v in eps_results.items()
        }
        logger.info(
            "ε=%.3f | cos_clean=%.4f | cos_attacked=%.4f | cos_defended=%.4f | accept=%.3f",
            eps,
            results["per_epsilon"][eps]["cosine_clean"],
            results["per_epsilon"][eps]["cosine_attacked"],
            results["per_epsilon"][eps]["cosine_defended"],
            results["per_epsilon"][eps]["acceptance_rate"],
        )

    return results
