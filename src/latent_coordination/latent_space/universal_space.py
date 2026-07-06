"""
Universal Latent Space: Hub-and-Spoke Latent State Transfer.

Maintains a registry of lightweight LatentAdapter modules (one encoder +
one decoder per registered agent) that all project to/from a shared
'universal_dim' space.  Enables text-free latent state exchange between
heterogeneous agents (different model families / hidden sizes).

Communication pattern (hub-and-spoke):
    Agent A  --[encode_A]--> universal space --[decode_B]--> Agent B

The class also tracks transfer history for analysis and provides
roundtrip fidelity metrics (encode -> decode cosine similarity).
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
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from latent_coordination.latent_space.adapter import AdapterConfig, LatentAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transfer record
# ---------------------------------------------------------------------------

@dataclass
class TransferRecord:
    """Log entry for a single latent state transfer event.

    Attributes:
        sender_id: Agent ID of the sender.
        receiver_id: Agent ID of the receiver (or 'universal' for encode-only).
        timestamp: Unix timestamp of the transfer.
        payload_shape: Shape of the transferred tensor as a tuple.
        cosine_similarity: Roundtrip cosine similarity if computed, else None.
        latency_ms: Wall-clock time for the transfer in milliseconds.
        effective_rank: Effective rank of the transferred hub vector if computed.
    """

    sender_id: str
    receiver_id: str
    timestamp: float
    payload_shape: Tuple[int, ...]
    cosine_similarity: Optional[float]
    latency_ms: float
    effective_rank: Optional[float] = None


# ---------------------------------------------------------------------------
# Agent record
# ---------------------------------------------------------------------------

@dataclass
class _AgentEntry:
    hidden_dim: int
    encoder: LatentAdapter  # agent hidden_dim -> universal_dim
    decoder: LatentAdapter  # universal_dim    -> agent hidden_dim


# ---------------------------------------------------------------------------
# Interlingua regularization losses (Module A+B)
# ---------------------------------------------------------------------------

def unbiased_hsic(K: Tensor, L: Tensor) -> Tensor:
    """Unbiased HSIC₁ estimator (Song et al. 2012) for kernel matrices K, L.

    The strategy audit specifically required the *unbiased* minibatch estimator
    for the CKA loss term — the naive biased estimator has batch-size-dependent
    gradient bias that corrupts small-batch adapter training. Requires n >= 4.
    """
    n = K.shape[0]
    if n < 4:
        raise ValueError(f"Unbiased HSIC needs batch size >= 4, got {n}.")
    K = K.clone().fill_diagonal_(0.0)
    L = L.clone().fill_diagonal_(0.0)
    ones = torch.ones(n, 1, device=K.device, dtype=K.dtype)
    term1 = torch.trace(K @ L)
    term2 = (ones.T @ K @ ones) * (ones.T @ L @ ones) / ((n - 1) * (n - 2))
    term3 = (ones.T @ K @ L @ ones) * (2.0 / (n - 2))
    return (term1 + term2.squeeze() - term3.squeeze()) / (n * (n - 3))


def cka_loss_unbiased(X: Tensor, Y: Tensor, eps: float = 1e-8) -> Tensor:
    """L_CKA = 1 − CKA(X, Y) with linear kernels and the unbiased HSIC₁ estimator.

    X, Y are row-aligned activation batches (same prompts / paired examples),
    shape (n, d_x) and (n, d_y).
    """
    K = X @ X.T
    L = Y @ Y.T
    hsic_xy = unbiased_hsic(K, L)
    hsic_xx = unbiased_hsic(K, K)
    hsic_yy = unbiased_hsic(L, L)
    cka = hsic_xy / (torch.sqrt(hsic_xx.clamp(min=eps) * hsic_yy.clamp(min=eps)))
    return 1.0 - cka


# ---------------------------------------------------------------------------
# UniversalLatentHub
# ---------------------------------------------------------------------------

class UniversalLatentHub:
    """Hub-and-spoke universal latent space for heterogeneous multi-agent systems.

    Each registered agent gets a pair of adapters:
        • **encoder**: maps the agent's hidden states to the shared universal space.
        • **decoder**: maps universal-space vectors back to the agent's hidden size.

    Transfer from agent A to agent B is: ``encode_A -> decode_B``.

    Args:
        universal_dim: Dimensionality of the shared universal latent space.
        adapter_hidden_dim: Hidden layer width inside each adapter MLP.
        dropout_rate: Dropout rate for adapters.
        device: PyTorch device string.

    Example::

        uls = UniversalLatentHub(universal_dim=256)
        uls.register_agent('llama', hidden_dim=4096)
        uls.register_agent('mistral', hidden_dim=4096)
        states = torch.randn(8, 4096)
        transferred = uls.transfer('llama', 'mistral', states)
    """

    def __init__(
        self,
        universal_dim: int = 256,
        adapter_hidden_dim: int = 512,
        dropout_rate: float = 0.1,
        device: str = "cpu",
    ) -> None:
        self.universal_dim = universal_dim
        self.adapter_hidden_dim = adapter_hidden_dim
        self.dropout_rate = dropout_rate
        self.device = torch.device(device)

        self._agents: Dict[str, _AgentEntry] = {}
        self._history: List[TransferRecord] = []

        logger.info(
            "UniversalLatentHub: universal_dim=%d, device=%s",
            universal_dim,
            device,
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_agent(
        self,
        agent_id: str,
        hidden_dim: int,
        encoder: Optional[LatentAdapter] = None,
        decoder: Optional[LatentAdapter] = None,
    ) -> None:
        """Register a new agent and create (or accept) its adapter pair.

        Args:
            agent_id: Unique string identifier for the agent.
            hidden_dim: The agent's native embedding dimension.
            encoder: Optional pre-built encoder adapter.  If ``None``, a fresh
                adapter is created with the default config.
            decoder: Optional pre-built decoder adapter.  If ``None``, created
                automatically.
        """
        if agent_id in self._agents:
            logger.warning("Agent '%s' already registered; overwriting.", agent_id)

        enc_cfg = AdapterConfig(
            in_dim=hidden_dim,
            out_dim=self.universal_dim,
            hidden_dim=self.adapter_hidden_dim,
            dropout_rate=self.dropout_rate,
            use_residual=False,
        )
        dec_cfg = AdapterConfig(
            in_dim=self.universal_dim,
            out_dim=hidden_dim,
            hidden_dim=self.adapter_hidden_dim,
            dropout_rate=self.dropout_rate,
            use_residual=False,
        )

        enc = encoder if encoder is not None else LatentAdapter(enc_cfg)
        dec = decoder if decoder is not None else LatentAdapter(dec_cfg)

        enc = enc.to(self.device)
        dec = dec.to(self.device)

        self._agents[agent_id] = _AgentEntry(
            hidden_dim=hidden_dim, encoder=enc, decoder=dec
        )
        logger.info(
            "Registered agent '%s' (hidden_dim=%d -> universal_dim=%d)",
            agent_id,
            hidden_dim,
            self.universal_dim,
        )

    def is_registered(self, agent_id: str) -> bool:
        """Return True if the agent is registered."""
        return agent_id in self._agents

    def list_agents(self) -> List[str]:
        """Return list of all registered agent IDs."""
        return list(self._agents.keys())

    # ------------------------------------------------------------------
    # Core projection operations
    # ------------------------------------------------------------------

    def encode(self, agent_id: str, hidden_states: Tensor) -> Tensor:
        """Project an agent's hidden states into the universal latent space.

        Args:
            agent_id: Registered agent identifier.
            hidden_states: Tensor of shape (B, hidden_dim) or (B, L, hidden_dim).
                If 3D, the last dimension is projected independently.

        Returns:
            Universal-space tensor of shape (B, universal_dim) or
            (B, L, universal_dim).

        Raises:
            KeyError: If the agent is not registered.
        """
        self._require_agent(agent_id)
        entry = self._agents[agent_id]
        # Cast to float32: adapters are always fp32; hidden states from bf16 models need conversion
        x = hidden_states.to(self.device).float()
        return entry.encoder(x)

    def decode(self, agent_id: str, universal_states: Tensor) -> Tensor:
        """Project universal-space states back to an agent's hidden space.

        Args:
            agent_id: Registered agent identifier.
            universal_states: Tensor of shape (B, universal_dim).

        Returns:
            Agent-space tensor of shape (B, hidden_dim).
        """
        self._require_agent(agent_id)
        entry = self._agents[agent_id]
        x = universal_states.to(self.device).float()
        return entry.decoder(x)

    def transfer(
        self,
        sender_id: str,
        receiver_id: str,
        hidden_states: Tensor,
        record_transfer: bool = True,
        norm_match: bool = False,
    ) -> Tensor:
        """Transfer latent states from one agent to another via universal space.

        Implements: ``encode(sender) -> decode(receiver)``

        Args:
            sender_id: Agent that produced the hidden states.
            receiver_id: Agent that will consume the transferred states.
            hidden_states: Sender's hidden state tensor, shape (B, sender_hidden_dim).
            record_transfer: If True, logs this transfer to history.
            norm_match: If True, rescales the decoded output so its RMS norm
                matches the source hidden states (prevents off-manifold injection).

        Returns:
            Receiver-compatible tensor of shape (B, receiver_hidden_dim).
        """
        from latent_coordination.latent_space.adapter import NormMatchLayer
        t_start = time.perf_counter()

        universal = self.encode(sender_id, hidden_states)
        received = self.decode(receiver_id, universal)

        if norm_match:
            _nm = NormMatchLayer()
            received = _nm(received, hidden_states.to(received.device).float())

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if record_transfer:
            with torch.no_grad():
                receiver_u = self.encode(receiver_id, received) if receiver_id in self._agents else universal
                cos_sim = float(
                    F.cosine_similarity(universal, receiver_u, dim=-1).mean().item()
                )
                try:
                    sv = torch.linalg.svdvals(universal.float())
                    sv_norm = sv / (sv.sum() + 1e-12)
                    h = -(sv_norm * (sv_norm + 1e-12).log()).sum().item()
                    eff_rank: Optional[float] = float(torch.exp(torch.tensor(h)).item())
                except Exception:
                    eff_rank = None
            self._history.append(
                TransferRecord(
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                    timestamp=time.time(),
                    payload_shape=tuple(hidden_states.shape),
                    cosine_similarity=cos_sim,
                    latency_ms=latency_ms,
                    effective_rank=eff_rank,
                )
            )
        logger.debug(
            "transfer: %s -> %s | shape=%s | %.2f ms",
            sender_id,
            receiver_id,
            tuple(hidden_states.shape),
            latency_ms,
        )
        return received

    def broadcast(
        self,
        sender_id: str,
        hidden_states: Tensor,
        target_ids: List[str],
    ) -> Dict[str, Tensor]:
        """Broadcast sender's hidden states to multiple receiver agents.

        Encodes the sender's states into universal space once, then decodes
        for each target agent.  This is the hub-spoke broadcast pattern where
        the orchestrator acts as hub.

        Args:
            sender_id: Agent that produced the hidden states.
            hidden_states: Sender's hidden state tensor, shape (B, sender_hidden_dim).
            target_ids: List of target agent IDs to broadcast to.

        Returns:
            Dict mapping each target agent ID to its received tensor.
        """
        universal = self.encode(sender_id, hidden_states)
        results: Dict[str, Tensor] = {}
        for target_id in target_ids:
            if target_id == sender_id:
                continue
            self._require_agent(target_id)
            received = self.decode(target_id, universal)
            results[target_id] = received
            self._history.append(
                TransferRecord(
                    sender_id=sender_id,
                    receiver_id=target_id,
                    timestamp=time.time(),
                    payload_shape=tuple(hidden_states.shape),
                    cosine_similarity=None,
                    latency_ms=0.0,
                )
            )
        logger.debug(
            "broadcast: %s -> %d targets", sender_id, len(results)
        )
        return results

    # ------------------------------------------------------------------
    # Adapter training (Module A+B: interlingua CKA-DAE regularization)
    # ------------------------------------------------------------------

    def fit_adapters(
        self,
        states_by_agent: Dict[str, Tensor],
        n_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        dae_sigma: float = 0.1,
        mu_cka: float = 1.0,
        gamma_dae: float = 1.0,
    ) -> Dict[str, float]:
        """Train all registered adapter pairs with the Module A+B objective.

        ``L_adapt = L_recon + μ·L_CKA + γ·L_DAE``:

        * **L_recon** — ``Σ_i ‖D_i(E_i(h_i)) − h_i‖²`` preserves each agent's
          local representation.
        * **L_DAE** — same reconstruction with ``h_i + ε, ε ~ N(0, σ²I)``
          corruption before encoding, forcing the hub to keep semantic content
          robust to surface variation.
        * **L_CKA** — ``1 − CKA(E_i(h_i), E_j(h_j))`` over row-aligned pairs of
          agents, computed with the **unbiased HSIC₁ estimator** (see
          :func:`cka_loss_unbiased`), aligning heterogeneous architectures'
          hub geometry.

        Args:
            states_by_agent: agent_id → (N, hidden_dim_i) real hidden states,
                row-aligned across agents (same prompts, or parallel
                translations for a cross-lingual variant). Every key must be a
                registered agent. This method refuses to run on unregistered
                agents or empty tensors — adapters must never be "trained" on
                fabricated data.
            n_epochs, lr, batch_size: optimization hyperparameters
                (configs/*.yaml ``universal_latent_space.adapter_training``).
            dae_sigma: Gaussian corruption σ for the DAE term.
            mu_cka, gamma_dae: loss weights (independently sweepable — the
                audit flagged a real DAE-vs-CKA conflict risk, so neither is
                hardcoded relative to the other).

        Returns:
            Dict of final loss components (``recon``, ``dae``, ``cka``,
            ``total``).
        """
        if not states_by_agent:
            raise ValueError("fit_adapters requires real hidden states; got none.")
        agent_ids = sorted(states_by_agent)
        for aid in agent_ids:
            self._require_agent(aid)
            t = states_by_agent[aid]
            if t is None or t.numel() == 0:
                raise ValueError(f"Empty hidden-state tensor for agent '{aid}'.")
            if t.shape[-1] != self._agents[aid].hidden_dim:
                raise ValueError(
                    f"Hidden states for '{aid}' have dim {t.shape[-1]}, but the agent "
                    f"is registered with hidden_dim={self._agents[aid].hidden_dim}."
                )
        n_rows = {aid: states_by_agent[aid].shape[0] for aid in agent_ids}
        if len(set(n_rows.values())) != 1:
            raise ValueError(f"States must be row-aligned across agents; got {n_rows}.")
        n = next(iter(n_rows.values()))
        batch_size = max(4, min(batch_size, n))  # unbiased HSIC needs >= 4

        states = {aid: states_by_agent[aid].to(self.device).float() for aid in agent_ids}
        params: List[nn.Parameter] = []
        for aid in agent_ids:
            params.extend(self._agents[aid].encoder.parameters())
            params.extend(self._agents[aid].decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        pairs = [
            (agent_ids[i], agent_ids[j])
            for i in range(len(agent_ids))
            for j in range(i + 1, len(agent_ids))
        ]
        final: Dict[str, float] = {}
        for epoch in range(1, n_epochs + 1):
            perm = torch.randperm(n, device=self.device)
            epoch_losses = {"recon": 0.0, "dae": 0.0, "cka": 0.0, "total": 0.0}
            n_batches = 0
            for start in range(0, n - batch_size + 1, batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                l_recon = torch.zeros((), device=self.device)
                l_dae = torch.zeros((), device=self.device)
                hub_z: Dict[str, Tensor] = {}
                for aid in agent_ids:
                    h = states[aid][idx]
                    entry = self._agents[aid]
                    z = entry.encoder(h)
                    hub_z[aid] = z
                    l_recon = l_recon + F.mse_loss(entry.decoder(z), h)
                    noisy = h + torch.randn_like(h) * dae_sigma
                    l_dae = l_dae + F.mse_loss(entry.decoder(entry.encoder(noisy)), h)
                l_cka = torch.zeros((), device=self.device)
                for a, b in pairs:
                    l_cka = l_cka + cka_loss_unbiased(hub_z[a], hub_z[b])
                if pairs:
                    l_cka = l_cka / len(pairs)
                loss = l_recon + gamma_dae * l_dae + mu_cka * l_cka
                loss.backward()
                optimizer.step()
                epoch_losses["recon"] += float(l_recon.item())
                epoch_losses["dae"] += float(l_dae.item())
                epoch_losses["cka"] += float(l_cka.item()) if pairs else 0.0
                epoch_losses["total"] += float(loss.item())
                n_batches += 1
            if n_batches:
                final = {k: v / n_batches for k, v in epoch_losses.items()}
            if epoch % max(1, n_epochs // 10) == 0:
                logger.info(
                    "Adapter training epoch %d/%d | recon=%.4f dae=%.4f cka=%.4f",
                    epoch, n_epochs,
                    final.get("recon", float("nan")),
                    final.get("dae", float("nan")),
                    final.get("cka", float("nan")),
                )
        return final

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------

    def compute_transfer_quality(
        self,
        agent_id: str,
        hidden_states: Tensor,
    ) -> Dict[str, float]:
        """Compute roundtrip fidelity for a single agent's representations.

        Encodes then decodes the hidden states and measures cosine similarity
        and MSE between original and reconstructed.

        Args:
            agent_id: Agent whose adapter pair to evaluate.
            hidden_states: Sample hidden states, shape (B, hidden_dim).

        Returns:
            Dict with ``cosine_similarity``, ``mse``, ``mean_norm_ratio`` metrics.
        """
        self._require_agent(agent_id)
        with torch.no_grad():
            x = hidden_states.to(self.device)
            u = self.encode(agent_id, x)
            recon = self.decode(agent_id, u)
            cos_sim = float(F.cosine_similarity(x, recon, dim=-1).mean().item())
            mse = float(F.mse_loss(recon, x).item())
            norm_ratio = float(
                (recon.norm(dim=-1) / (x.norm(dim=-1) + 1e-8)).mean().item()
            )
        metrics = {
            "cosine_similarity": cos_sim,
            "mse": mse,
            "mean_norm_ratio": norm_ratio,
        }
        logger.debug("Transfer quality for '%s': %s", agent_id, metrics)
        return metrics

    def compute_information_metrics(
        self,
        agent_id: str,
        hidden_states: Tensor,
    ) -> Dict[str, float]:
        """Compute information-theoretic quality metrics for a single agent's hub mapping.

        Supplements the cosine-only roundtrip check with effective rank (a proxy
        for information preservation) and an HSIC-based MI proxy between original
        and reconstructed states.

        Args:
            agent_id: Registered agent identifier.
            hidden_states: Sample hidden states, shape (B, hidden_dim).

        Returns:
            Dict with ``effective_rank``, ``norm_ratio``, ``hsic_mi_proxy``,
            ``cosine_similarity``, and ``mse``.
        """
        self._require_agent(agent_id)
        with torch.no_grad():
            x = hidden_states.to(self.device).float()
            u = self.encode(agent_id, x)
            recon = self.decode(agent_id, u)

            cos_sim = float(F.cosine_similarity(x, recon, dim=-1).mean().item())
            mse = float(F.mse_loss(recon, x).item())
            norm_ratio = float((recon.norm(dim=-1) / (x.norm(dim=-1) + 1e-8)).mean().item())

            # Effective rank of hub vectors
            try:
                sv = torch.linalg.svdvals(u)
                sv_norm = sv / (sv.sum() + 1e-12)
                h = -(sv_norm * (sv_norm + 1e-12).log()).sum().item()
                eff_rank = float(torch.exp(torch.tensor(h)).item())
            except Exception:
                eff_rank = float("nan")

            # HSIC-based MI proxy (linear kernel, O(B^2))
            try:
                B = x.shape[0]
                H = torch.eye(B, device=x.device) - torch.ones(B, B, device=x.device) / B
                Kx = x @ x.T
                Kr = recon @ recon.T
                hsic = float(torch.trace(Kx @ H @ Kr @ H).item()) / max((B - 1) ** 2, 1)
            except Exception:
                hsic = float("nan")

        metrics = {
            "effective_rank": eff_rank,
            "norm_ratio": norm_ratio,
            "hsic_mi_proxy": hsic,
            "cosine_similarity": cos_sim,
            "mse": mse,
        }
        logger.debug("Information metrics for '%s': %s", agent_id, metrics)
        return metrics

    @staticmethod
    def align_ridge(
        source_states: Tensor,
        target_states: Tensor,
        alpha: float = 1e-4,
    ) -> Tensor:
        """Closed-form affine ridge alignment (O(N) hub maps, like VW).

        Computes the least-squares affine map W* = (S^T S + αI)^{-1} S^T T
        and returns W* @ source_states^T.  Useful as an alternative to
        learned MLP adapters when anchor data is available.

        Args:
            source_states: Source embeddings, shape (N, D_src).
            target_states: Target embeddings, shape (N, D_tgt).
            alpha: Ridge regularization coefficient.

        Returns:
            Aligned source tensor of shape (N, D_tgt).
        """
        S = source_states.float()
        T = target_states.float()
        I = torch.eye(S.shape[1], device=S.device) * alpha
        W = torch.linalg.solve(S.T @ S + I, S.T @ T)  # (D_src, D_tgt)
        return S @ W

    # ------------------------------------------------------------------
    # History & persistence
    # ------------------------------------------------------------------

    def get_transfer_history(self) -> List[TransferRecord]:
        """Return the full list of recorded transfer events."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear the transfer history log."""
        self._history.clear()

    def save_adapters(self, save_dir: str) -> None:
        """Persist all registered agent adapters to disk.

        Creates one sub-directory per agent containing ``encoder.pt`` and
        ``decoder.pt`` files.

        Args:
            save_dir: Root directory for saved adapters.
        """
        root = Path(save_dir)
        root.mkdir(parents=True, exist_ok=True)
        for agent_id, entry in self._agents.items():
            agent_dir = root / agent_id
            agent_dir.mkdir(exist_ok=True)
            torch.save(entry.encoder.state_dict(), agent_dir / "encoder.pt")
            torch.save(entry.decoder.state_dict(), agent_dir / "decoder.pt")
            logger.info("Saved adapters for agent '%s' to %s", agent_id, agent_dir)

    def load_adapters(self, save_dir: str) -> None:
        """Load adapter weights from disk for all registered agents.

        Only loads adapters whose directories exist under ``save_dir``.
        Agents must already be registered (adapter architecture must exist).

        Args:
            save_dir: Root directory previously used with :meth:`save_adapters`.
        """
        root = Path(save_dir)
        for agent_id, entry in self._agents.items():
            agent_dir = root / agent_id
            enc_path = agent_dir / "encoder.pt"
            dec_path = agent_dir / "decoder.pt"
            if enc_path.exists():
                entry.encoder.load_state_dict(
                    torch.load(enc_path, map_location=self.device)
                )
                logger.info("Loaded encoder for '%s'", agent_id)
            if dec_path.exists():
                entry.decoder.load_state_dict(
                    torch.load(dec_path, map_location=self.device)
                )
                logger.info("Loaded decoder for '%s'", agent_id)

    def get_encoder(self, agent_id: str) -> LatentAdapter:
        """Return the encoder adapter for a registered agent."""
        self._require_agent(agent_id)
        return self._agents[agent_id].encoder

    def get_decoder(self, agent_id: str) -> LatentAdapter:
        """Return the decoder adapter for a registered agent."""
        self._require_agent(agent_id)
        return self._agents[agent_id].decoder

    def parameters(self) -> List[nn.Parameter]:
        """Return all adapter parameters for joint optimisation."""
        params: List[nn.Parameter] = []
        for entry in self._agents.values():
            params.extend(list(entry.encoder.parameters()))
            params.extend(list(entry.decoder.parameters()))
        return params

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_agent(self, agent_id: str) -> None:
        if agent_id not in self._agents:
            raise KeyError(
                f"Agent '{agent_id}' not registered. "
                f"Call register_agent() first. "
                f"Registered: {list(self._agents.keys())}"
            )

    def __repr__(self) -> str:
        agents = list(self._agents.keys())
        return (
            f"UniversalLatentHub(universal_dim={self.universal_dim}, "
            f"n_agents={len(agents)}, agents={agents})"
        )
