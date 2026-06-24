"""
CVAE-based Graph Topology Prior for Multi-Agent System Coordination.

Learns a conditional distribution p(G | Q) over agent communication graphs
conditioned on task queries. Enables zero-shot topology transfer to new tasks
and languages by sampling from the learned prior.

Architecture:
    Query Encoder  : LSTM/Transformer -> Q_emb (query_dim)
    Graph Encoder  : MLP -> G_emb (graph_dim)
    CVAE Encoder   : concat(G_emb, Q_emb) -> (mu, logvar)  in z_dim
    CVAE Decoder   : concat(z, Q_emb)     -> recon_G (adj logits)

References:
    - Sohn et al. (2015) "Learning Structured Output Representation using
      Deep Conditional Generative Models"
    - Higgins et al. (2017) "beta-VAE: Learning Basic Visual Concepts"
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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _try_import_wandb():
    try:
        import wandb
        return wandb
    except ImportError:
        return None


def _try_import_tb():
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Hyperparameters for CVAETopologyPrior training.

    Attributes:
        z_dim: Dimensionality of the CVAE latent space.
        query_dim: Output dimensionality of the query encoder.
        graph_hidden_dim: Hidden dim of the graph encoder MLP.
        encoder_hidden_dim: Hidden dim of the CVAE encoder MLP.
        decoder_hidden_dim: Hidden dim of the CVAE decoder MLP.
        max_n_agents: Maximum number of agents (determines adjacency matrix size).
        lstm_hidden_dim: Hidden dim for the LSTM query encoder.
        lstm_n_layers: Number of LSTM layers in the query encoder.
        query_vocab_size: Vocabulary size for query token embeddings.
        query_embed_dim: Token embedding dim fed to LSTM.
        lr: Learning rate for Adam optimizer.
        n_epochs: Total training epochs.
        batch_size: Mini-batch size.
        beta_max: Maximum beta value for beta-VAE annealing.
        warmup_epochs: Epochs before beta starts increasing.
        cycle_length: Number of epochs per beta annealing cycle.
        grad_clip: Gradient clipping norm (None disables clipping).
        checkpoint_interval: Save checkpoint every N epochs.
        use_wandb: Whether to log to Weights & Biases.
        use_tensorboard: Whether to log to TensorBoard.
        device: PyTorch device string.
    """

    z_dim: int = 64
    query_dim: int = 128
    graph_hidden_dim: int = 256
    encoder_hidden_dim: int = 256
    decoder_hidden_dim: int = 256
    max_n_agents: int = 8
    lstm_hidden_dim: int = 128
    lstm_n_layers: int = 2
    query_vocab_size: int = 30_522  # BERT-compatible default
    query_embed_dim: int = 64
    lr: float = 3e-4
    n_epochs: int = 100
    batch_size: int = 32
    beta_max: float = 4.0
    warmup_epochs: int = 10
    cycle_length: int = 20
    grad_clip: Optional[float] = 1.0
    checkpoint_interval: int = 10
    use_wandb: bool = False
    use_tensorboard: bool = False
    device: str = "cpu"
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Beta annealing schedule (cyclical)
# ---------------------------------------------------------------------------

def beta_annealing_schedule(
    epoch: int,
    max_epochs: int,
    beta_max: float = 4.0,
    warmup: int = 10,
    cycle_length: int = 20,
) -> float:
    """Compute cyclical beta for KL-annealing in beta-VAE training.

    During warm-up (epoch < warmup), beta = 0.0 to let the reconstruction
    stabilise before regularisation kicks in.  After warm-up the schedule
    follows a cosine ramp that resets every ``cycle_length`` epochs.

    Args:
        epoch: Current training epoch (0-indexed).
        max_epochs: Total number of training epochs.
        beta_max: Peak beta value.
        warmup: Number of epochs with beta = 0.
        cycle_length: Epochs per cosine cycle.

    Returns:
        Float beta value in [0, beta_max].

    Example:
        >>> betas = [beta_annealing_schedule(e, 100, 4.0) for e in range(100)]
    """
    if epoch < warmup:
        return 0.0
    adjusted = epoch - warmup
    cycle_pos = adjusted % cycle_length
    beta = beta_max * 0.5 * (1.0 - math.cos(math.pi * cycle_pos / cycle_length))
    return float(beta)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TopologyDataset(Dataset):
    """Dataset of (G, Q_tokens, metadata) triples for CVAE training.

    Each item represents a historical multi-agent task along with the
    agent communication topology that was used (or labelled as optimal).

    Args:
        graphs: List of adjacency matrices, each shape (max_n_agents, max_n_agents).
        query_tokens: List of query token ID tensors, shape (seq_len,).
        metadata: Optional list of metadata dicts (task type, language, etc.).
        max_n_agents: Maximum agent count; pads/truncates graphs to this size.
        max_seq_len: Maximum query token length; pads/truncates queries.
    """

    def __init__(
        self,
        graphs: List[Tensor],
        query_tokens: List[Tensor],
        metadata: Optional[List[Dict[str, Any]]] = None,
        max_n_agents: int = 8,
        max_seq_len: int = 64,
    ) -> None:
        if len(graphs) != len(query_tokens):
            raise ValueError(
                f"graphs and query_tokens must have the same length, "
                f"got {len(graphs)} vs {len(query_tokens)}"
            )
        self.graphs = graphs
        self.query_tokens = query_tokens
        self.metadata = metadata or [{} for _ in graphs]
        self.max_n_agents = max_n_agents
        self.max_seq_len = max_seq_len
        logger.info("TopologyDataset: %d samples", len(self))

    # ------------------------------------------------------------------

    def _pad_graph(self, G: Tensor) -> Tensor:
        """Pad or crop adjacency matrix to (max_n_agents, max_n_agents)."""
        n = G.shape[0]
        N = self.max_n_agents
        if n >= N:
            return G[:N, :N].float()
        padded = torch.zeros(N, N, dtype=torch.float32)
        padded[:n, :n] = G.float()
        return padded

    def _pad_query(self, q: Tensor) -> Tensor:
        """Pad or crop query token IDs to (max_seq_len,)."""
        L = q.shape[0]
        S = self.max_seq_len
        if L >= S:
            return q[:S].long()
        padded = torch.zeros(S, dtype=torch.long)
        padded[:L] = q.long()
        return padded

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        G = self._pad_graph(self.graphs[idx])
        Q = self._pad_query(self.query_tokens[idx])
        return {"G": G, "Q": Q, "meta": self.metadata[idx]}

    @classmethod
    def from_random(
        cls,
        n_samples: int,
        max_n_agents: int = 8,
        max_seq_len: int = 32,
        vocab_size: int = 1000,
        seed: int = 42,
    ) -> "TopologyDataset":
        """Create a synthetic random dataset for testing.

        Args:
            n_samples: Number of (G, Q) pairs to generate.
            max_n_agents: Adjacency matrix size.
            max_seq_len: Query token sequence length.
            vocab_size: Token vocabulary size.
            seed: Random seed.

        Returns:
            A ``TopologyDataset`` with randomly generated samples.
        """
        rng = torch.Generator()
        rng.manual_seed(seed)
        graphs = [
            (torch.rand(max_n_agents, max_n_agents, generator=rng) > 0.5).float()
            for _ in range(n_samples)
        ]
        queries = [
            torch.randint(0, vocab_size, (max_seq_len,), generator=rng)
            for _ in range(n_samples)
        ]
        return cls(graphs, queries, max_n_agents=max_n_agents, max_seq_len=max_seq_len)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class _QueryEncoder(nn.Module):
    """LSTM-based query encoder: token IDs -> query embedding.

    Args:
        vocab_size: Size of token vocabulary.
        embed_dim: Token embedding dimension.
        hidden_dim: LSTM hidden dimension.
        n_layers: Number of LSTM layers.
        output_dim: Final linear projection size (query_dim).
        dropout: Dropout rate (applied between LSTM layers).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        n_layers: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Args:
            token_ids: Long tensor of shape (B, seq_len).

        Returns:
            Query embedding of shape (B, output_dim).
        """
        emb = self.embedding(token_ids)          # (B, L, embed_dim)
        _, (h_n, _) = self.lstm(emb)             # h_n: (2*n_layers, B, hidden)
        # Concatenate forward and backward final hidden states
        fwd = h_n[-2]                            # (B, hidden)
        bwd = h_n[-1]                            # (B, hidden)
        cat = torch.cat([fwd, bwd], dim=-1)      # (B, 2*hidden)
        return self.proj(cat)                    # (B, output_dim)


class _GraphEncoder(nn.Module):
    """MLP graph encoder: flattened adjacency matrix -> graph embedding.

    Args:
        max_n_agents: Adjacency matrix side length; input dim = max_n_agents^2.
        hidden_dim: Hidden layer dimension.
        output_dim: Output embedding dimension.
    """

    def __init__(self, max_n_agents: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        in_dim = max_n_agents * max_n_agents
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, G: Tensor) -> Tensor:
        """
        Args:
            G: Float adjacency tensor of shape (B, N, N).

        Returns:
            Graph embedding of shape (B, output_dim).
        """
        B = G.shape[0]
        flat = G.view(B, -1)   # (B, N*N)
        return self.net(flat)  # (B, output_dim)


class _VAEEncoder(nn.Module):
    """Maps concatenated (graph_emb, query_emb) -> (mu, logvar).

    Args:
        input_dim: Dimension of concatenated input.
        hidden_dim: Hidden layer dimension.
        z_dim: Latent space dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Concatenated conditioning tensor, shape (B, input_dim).

        Returns:
            Tuple of (mu, logvar), each shape (B, z_dim).
        """
        h = self.shared(x)
        return self.fc_mean(h), self.fc_logvar(h)


class _VAEDecoder(nn.Module):
    """Maps concatenated (z, query_emb) -> adjacency logits.

    Args:
        input_dim: z_dim + query_dim.
        hidden_dim: Hidden layer dimension.
        max_n_agents: Adjacency matrix side; output dim = max_n_agents^2.
    """

    def __init__(self, input_dim: int, hidden_dim: int, max_n_agents: int) -> None:
        super().__init__()
        out_dim = max_n_agents * max_n_agents
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.max_n_agents = max_n_agents

    def forward(self, z: Tensor, q_emb: Tensor) -> Tensor:
        """
        Args:
            z: Latent vector, shape (B, z_dim).
            q_emb: Query embedding, shape (B, query_dim).

        Returns:
            Reconstructed adjacency probabilities, shape (B, N, N).
        """
        inp = torch.cat([z, q_emb], dim=-1)           # (B, z+q)
        logits = self.net(inp)                         # (B, N*N)
        probs = torch.sigmoid(logits)
        return probs.view(-1, self.max_n_agents, self.max_n_agents)


# ---------------------------------------------------------------------------
# Main Module
# ---------------------------------------------------------------------------

class CVAETopologyPrior(nn.Module):
    """Conditional VAE for learning transferable agent communication topologies.

    Models ``p(G | Q)`` where ``G`` is an adjacency matrix and ``Q`` is a
    task query.  After training, new topologies for unseen queries can be
    sampled by drawing ``z ~ N(0, I)`` and decoding with
    :py:meth:`sample_topology`.

    Architecture summary::

        Q_tokens  --[QueryEncoder]--> Q_emb  ─────────────────────────┐
        G_adj     --[GraphEncoder]--> G_emb ──┐                        │
                                              ├-> [VAEEncoder] -> z    │
                                              └────────────────────────┘
        z, Q_emb ──────────────────────────-> [VAEDecoder] -> recon_G

    Args:
        config: :class:`TrainingConfig` holding all hyperparameters.

    Example::

        cfg = TrainingConfig(z_dim=32, max_n_agents=6)
        model = CVAETopologyPrior(cfg)
        G = torch.zeros(2, 6, 6)  # batch of adjacency matrices
        Q = torch.randint(0, 1000, (2, 16))  # batch of query tokens
        recon_G, mu, logvar = model(G, Q)
    """

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        N = config.max_n_agents

        # Sub-modules
        self.query_encoder = _QueryEncoder(
            vocab_size=config.query_vocab_size,
            embed_dim=config.query_embed_dim,
            hidden_dim=config.lstm_hidden_dim,
            n_layers=config.lstm_n_layers,
            output_dim=config.query_dim,
        )
        self.graph_encoder = _GraphEncoder(
            max_n_agents=N,
            hidden_dim=config.graph_hidden_dim,
            output_dim=config.graph_hidden_dim,
        )
        enc_input_dim = config.graph_hidden_dim + config.query_dim
        self.vae_encoder = _VAEEncoder(
            input_dim=enc_input_dim,
            hidden_dim=config.encoder_hidden_dim,
            z_dim=config.z_dim,
        )
        dec_input_dim = config.z_dim + config.query_dim
        self.vae_decoder = _VAEDecoder(
            input_dim=dec_input_dim,
            hidden_dim=config.decoder_hidden_dim,
            max_n_agents=N,
        )

        # Expose fc_mean and fc_logvar at top level for compatibility
        self.fc_mean = self.vae_encoder.fc_mean
        self.fc_logvar = self.vae_encoder.fc_logvar

        self._device = torch.device(config.device)
        self.to(self._device)

        logger.info(
            "CVAETopologyPrior initialised | z_dim=%d, max_agents=%d, params=%d",
            config.z_dim,
            N,
            sum(p.numel() for p in self.parameters()),
        )

    # ------------------------------------------------------------------
    # Core CVAE methods
    # ------------------------------------------------------------------

    def encode(self, G: Tensor, Q: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode (G, Q) pair into latent space parameters.

        Args:
            G: Adjacency matrix batch, shape (B, N, N).
            Q: Query token batch, shape (B, seq_len).

        Returns:
            Tuple ``(mu, logvar)`` each of shape (B, z_dim).
        """
        q_emb = self.query_encoder(Q)       # (B, query_dim)
        g_emb = self.graph_encoder(G)       # (B, graph_hidden_dim)
        combined = torch.cat([g_emb, q_emb], dim=-1)
        return self.vae_encoder(combined)   # (mu, logvar)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample z via the reparameterization trick.

        At training time samples ``z = mu + eps * exp(0.5 * logvar)`` where
        ``eps ~ N(0, I)``.  At evaluation time returns ``mu`` directly.

        Args:
            mu: Mean tensor, shape (B, z_dim).
            logvar: Log-variance tensor, shape (B, z_dim).

        Returns:
            Latent sample of shape (B, z_dim).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: Tensor, Q: Tensor) -> Tensor:
        """Decode latent z conditioned on query Q.

        Args:
            z: Latent vector batch, shape (B, z_dim).
            Q: Query token batch, shape (B, seq_len).

        Returns:
            Reconstructed adjacency probability matrix, shape (B, N, N).
        """
        q_emb = self.query_encoder(Q)        # (B, query_dim)
        return self.vae_decoder(z, q_emb)   # (B, N, N)

    def forward(
        self, G: Tensor, Q: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Full CVAE forward pass.

        Args:
            G: Adjacency matrix batch, shape (B, N, N).
            Q: Query token batch, shape (B, seq_len).

        Returns:
            Tuple ``(recon_G, mu, logvar)`` where ``recon_G`` is shape (B, N, N)
            and ``mu``, ``logvar`` are each shape (B, z_dim).
        """
        mu, logvar = self.encode(G, Q)
        z = self.reparameterize(mu, logvar)
        recon_G = self.decode(z, Q)
        return recon_G, mu, logvar

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        recon_G: Tensor,
        G: Tensor,
        mu: Tensor,
        logvar: Tensor,
        beta: float = 1.0,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute beta-CVAE ELBO loss.

        Loss = BCE(recon_G, G)  +  beta * KL( q(z|G,Q) || p(z) )

        Args:
            recon_G: Reconstructed adjacency, shape (B, N, N), values in [0,1].
            G: Ground-truth adjacency, shape (B, N, N), binary.
            mu: Encoder mean, shape (B, z_dim).
            logvar: Encoder log-variance, shape (B, z_dim).
            beta: KL weight for beta-VAE regularisation.

        Returns:
            Tuple of total loss tensor and dict of component scalars.
        """
        # Binary cross-entropy reconstruction loss (averaged over all edges)
        recon_loss = F.binary_cross_entropy(recon_G, G, reduction="mean")
        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + beta * kl_loss
        return total, {
            "total": total.item(),
            "recon": recon_loss.item(),
            "kl": kl_loss.item(),
            "beta": beta,
        }

    # ------------------------------------------------------------------
    # Sampling / inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_topology(
        self,
        Q: Tensor,
        n_samples: int = 1,
        threshold: float = 0.5,
    ) -> Tensor:
        """Sample agent communication topologies for new queries.

        Draws ``z ~ N(0, I)`` and decodes conditioned on ``Q``.  Returns a
        binary adjacency matrix obtained by thresholding the decoder output.

        Args:
            Q: Query token tensor, shape (1, seq_len) or (B, seq_len).
            n_samples: Number of topology samples per query.
            threshold: Probability threshold for binarising adjacency.

        Returns:
            Binary adjacency tensor, shape (B * n_samples, N, N).
        """
        self.eval()
        B, seq_len = Q.shape
        Q_rep = Q.repeat_interleave(n_samples, dim=0)  # (B*n_samples, seq_len)
        z = torch.randn(B * n_samples, self.config.z_dim, device=self._device)
        probs = self.decode(z, Q_rep)                  # (B*n_samples, N, N)
        return (probs > threshold).float()

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        beta: float = 1.0,
        grad_clip: Optional[float] = 1.0,
    ) -> Dict[str, float]:
        """Run one full training epoch.

        Args:
            dataloader: DataLoader yielding ``{G, Q, meta}`` batches.
            optimizer: PyTorch optimiser instance.
            beta: KL weight for this epoch (from annealing schedule).
            grad_clip: Gradient clipping max norm (None to disable).

        Returns:
            Dict with mean ``total``, ``recon``, and ``kl`` losses for the epoch.
        """
        self.train()
        epoch_totals: Dict[str, float] = {"total": 0.0, "recon": 0.0, "kl": 0.0}
        n_batches = 0

        for batch in dataloader:
            G = batch["G"].to(self._device)   # (B, N, N)
            Q = batch["Q"].to(self._device)   # (B, seq_len)

            optimizer.zero_grad()
            recon_G, mu, logvar = self(G, Q)
            loss, components = self.compute_loss(recon_G, G, mu, logvar, beta=beta)
            loss.backward()

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

            optimizer.step()

            for k, v in components.items():
                if k in epoch_totals:
                    epoch_totals[k] += v
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in epoch_totals.items()}

    def fit(
        self,
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Callable] = None,
        checkpoint_dir: Optional[str] = None,
        resume_from: Optional[str] = None,
        wandb_run=None,
        tb_writer=None,
    ) -> List[Dict[str, float]]:
        """Full training loop with annealing, logging, and checkpointing.

        Args:
            dataloader: DataLoader yielding topology dataset batches.
            optimizer: Optimizer (defaults to Adam with config LR).
            scheduler: Optional LR scheduler; called each epoch.
            checkpoint_dir: Directory to save epoch checkpoints.
            resume_from: Path to checkpoint to resume from.
            wandb_run: Active wandb run object (optional).
            tb_writer: TensorBoard SummaryWriter (optional).

        Returns:
            List of per-epoch loss dicts.
        """
        cfg = self.config
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)

        start_epoch = 0
        history: List[Dict[str, float]] = []

        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from, optimizer)
            logger.info("Resuming from checkpoint at epoch %d", start_epoch)

        for epoch in range(start_epoch, cfg.n_epochs):
            beta = beta_annealing_schedule(
                epoch,
                cfg.n_epochs,
                beta_max=cfg.beta_max,
                warmup=cfg.warmup_epochs,
                cycle_length=cfg.cycle_length,
            )
            metrics = self.train_epoch(
                dataloader,
                optimizer,
                beta=beta,
                grad_clip=cfg.grad_clip,
            )
            metrics["epoch"] = epoch
            metrics["beta"] = beta
            history.append(metrics)

            if scheduler is not None:
                scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "Epoch %03d/%03d | beta=%.3f | total=%.4f | recon=%.4f | kl=%.4f",
                    epoch + 1,
                    cfg.n_epochs,
                    beta,
                    metrics["total"],
                    metrics["recon"],
                    metrics["kl"],
                )

            # Optional logging
            if wandb_run is not None:
                wandb_run.log(metrics, step=epoch)
            if tb_writer is not None:
                for k, v in metrics.items():
                    if isinstance(v, float):
                        tb_writer.add_scalar(f"cvae/{k}", v, epoch)

            # Checkpointing
            if checkpoint_dir and (epoch + 1) % cfg.checkpoint_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"cvae_epoch_{epoch+1:04d}.pt")
                self.save_checkpoint(ckpt_path, epoch, optimizer)

        return history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Save model, config, and optionally optimizer state.

        Args:
            path: Destination file path (e.g. ``checkpoints/cvae_e10.pt``).
            epoch: Current epoch (saved for resuming).
            optimizer: Optional optimizer whose state is also saved.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "config": self.config,
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(payload, path)
        logger.info("CVAETopologyPrior checkpoint saved -> %s", path)

    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> int:
        """Load checkpoint from disk and restore model (and optionally optimizer).

        Args:
            path: Path to saved checkpoint ``.pt`` file.
            optimizer: If provided, restores optimizer state too.

        Returns:
            The epoch stored in the checkpoint (useful for resuming).
        """
        payload = torch.load(path, map_location=self._device)
        self.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        epoch = payload.get("epoch", 0)
        logger.info("CVAETopologyPrior loaded from %s (epoch %d)", path, epoch)
        return epoch

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def to_device(self, device: str) -> "CVAETopologyPrior":
        """Move the model to a new device and update internal device reference.

        Args:
            device: PyTorch device string, e.g. ``'cuda:0'``.

        Returns:
            Self, to allow chaining.
        """
        self._device = torch.device(device)
        return self.to(self._device)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"CVAETopologyPrior(z_dim={cfg.z_dim}, query_dim={cfg.query_dim}, "
            f"max_n_agents={cfg.max_n_agents}, params={self.count_parameters():,})"
        )
