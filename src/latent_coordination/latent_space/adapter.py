"""
Latent Adapter: lightweight MLP-based projection between agent embedding spaces.

Each LatentAdapter maps from an agent's native hidden_dim to a shared
universal_dim (or vice versa), enabling text-free latent state transfer
across heterogeneous agents.

Design:
    Two-layer MLP with LayerNorm + GELU activations.
    Optional residual connection when in_dim == out_dim.
    Offline training utilities (train_adapter, compute_reconstruction_error).
    AdapterBank for named lazy-loading collection of adapters.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# InfoNCE contrastive alignment loss
# ---------------------------------------------------------------------------

def infonce_loss(
    anchor: Tensor,
    positive: Tensor,
    temperature: float = 0.07,
    negatives: Optional[Tensor] = None,
) -> Tensor:
    """InfoNCE contrastive loss for hub alignment (NT-Xent style).

    Treats each (anchor[i], positive[i]) pair as a positive pair and all
    other positives in the batch as negatives.  Optional extra negatives can
    be appended.

    Args:
        anchor: Tensor of shape (B, D) — e.g. encoded universal vectors.
        positive: Tensor of shape (B, D) — e.g. reconstructed then re-encoded.
        temperature: Softmax temperature τ (lower = sharper contrast).
        negatives: Optional extra negatives, shape (K, D).

    Returns:
        Scalar InfoNCE loss averaged over the batch.
    """
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)

    if negatives is not None:
        negatives = F.normalize(negatives, dim=-1)
        keys = torch.cat([positive, negatives], dim=0)  # (B+K, D)
    else:
        keys = positive  # (B, D)

    logits = torch.mm(anchor, keys.T) / temperature  # (B, B+K)
    labels = torch.arange(anchor.shape[0], device=anchor.device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# NormMatch stabilization layer
# ---------------------------------------------------------------------------

class NormMatchLayer(nn.Module):
    """Scales output to match the RMS norm of a reference input.

    Prevents off-manifold injection by keeping hub tokens on the same norm
    manifold as the source embedding space.  Mirrors VW's NormMatch + RMS
    stabilizer.

    Args:
        eps: Small constant to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, output: Tensor, reference: Tensor) -> Tensor:
        """Scale output RMS to match reference RMS.

        Args:
            output: Projected tensor to rescale, shape (..., D).
            reference: Source tensor whose RMS sets the target scale, shape (..., D).

        Returns:
            Rescaled tensor of the same shape as output.
        """
        ref_rms = reference.norm(dim=-1, keepdim=True) / (reference.shape[-1] ** 0.5)
        out_rms = output.norm(dim=-1, keepdim=True) / (output.shape[-1] ** 0.5)
        scale = (ref_rms + self.eps) / (out_rms + self.eps)
        return output * scale


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AdapterConfig:
    """Configuration for a single LatentAdapter.

    Attributes:
        in_dim: Input dimension (agent hidden dim or universal dim).
        out_dim: Output dimension.
        hidden_dim: Intermediate MLP width.
        dropout_rate: Dropout probability applied after each activation.
        use_residual: Whether to add a residual skip connection
            (only active when in_dim == out_dim).
        use_norm_match: Whether to apply NormMatch scaling at output to keep
            hub tokens on the source embedding norm manifold.
    """

    in_dim: int
    out_dim: int
    hidden_dim: int = 256
    dropout_rate: float = 0.1
    use_residual: bool = True
    use_norm_match: bool = False


# ---------------------------------------------------------------------------
# LatentAdapter module
# ---------------------------------------------------------------------------

class LatentAdapter(nn.Module):
    """Lightweight two-layer MLP adapter for latent space projection.

    Maps tensors from ``in_dim`` to ``out_dim`` using:
        x -> LayerNorm -> Linear(in, hidden) -> GELU -> Dropout
          -> Linear(hidden, out) -> LayerNorm
          (+ residual if in_dim == out_dim and use_residual)

    Args:
        config: :class:`AdapterConfig` specifying dimensions and regularisation.

    Example::

        cfg = AdapterConfig(in_dim=768, out_dim=256)
        adapter = LatentAdapter(cfg)
        out = adapter(torch.randn(4, 768))   # shape (4, 256)
    """

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.in_norm = nn.LayerNorm(config.in_dim)
        self.fc1 = nn.Linear(config.in_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc2 = nn.Linear(config.hidden_dim, config.out_dim)
        self.out_norm = nn.LayerNorm(config.out_dim)
        self.use_residual = config.use_residual and (config.in_dim == config.out_dim)
        self.norm_match: Optional[NormMatchLayer] = (
            NormMatchLayer() if config.use_norm_match else None
        )

        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

        logger.debug(
            "LatentAdapter: %d -> %d (hidden=%d, residual=%s, norm_match=%s)",
            config.in_dim,
            config.out_dim,
            config.hidden_dim,
            self.use_residual,
            config.use_norm_match,
        )

    def forward(self, x: Tensor, reference: Optional[Tensor] = None) -> Tensor:
        """Project input tensor through the adapter.

        Args:
            x: Input tensor of shape (..., in_dim).
            reference: Optional source tensor for NormMatch scaling.
                Required when ``use_norm_match=True``; if None, NormMatch uses x.

        Returns:
            Projected tensor of shape (..., out_dim).
        """
        residual = x
        h = self.in_norm(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.out_norm(h)
        if self.use_residual:
            h = h + residual
        if self.norm_match is not None:
            ref = reference if reference is not None else x
            # ref may differ in last dim; use its norm magnitude regardless
            h = self.norm_match(h, ref)
        return h


# ---------------------------------------------------------------------------
# Named adapter wrapper (for AdapterBank)
# ---------------------------------------------------------------------------

@dataclass
class NamedAdapter:
    """Container pairing an adapter with its identifier and disk path.

    Attributes:
        adapter_id: Unique string identifier (e.g. ``'translation_to_hub'``).
        config: AdapterConfig used to build the adapter.
        adapter: Optional loaded :class:`LatentAdapter` (None = not yet loaded).
        checkpoint_path: Optional filesystem path to the saved ``.pt`` file.
    """

    adapter_id: str
    config: AdapterConfig
    adapter: Optional[LatentAdapter] = None
    checkpoint_path: Optional[str] = None


# ---------------------------------------------------------------------------
# AdapterBank
# ---------------------------------------------------------------------------

class AdapterBank:
    """Manages a named collection of :class:`LatentAdapter` objects.

    Supports lazy loading from disk: adapters are only loaded into memory when
    first accessed, avoiding OOM when managing many adapters.

    Args:
        device: PyTorch device to load adapters onto.

    Example::

        bank = AdapterBank(device='cpu')
        cfg = AdapterConfig(in_dim=768, out_dim=256)
        bank.register('my_adapter', cfg)
        adapter = bank.get('my_adapter')
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self._registry: Dict[str, NamedAdapter] = {}

    # ------------------------------------------------------------------

    def register(
        self,
        adapter_id: str,
        config: AdapterConfig,
        checkpoint_path: Optional[str] = None,
        adapter: Optional[LatentAdapter] = None,
    ) -> None:
        """Register a new adapter (with optional pre-loaded instance or path).

        Args:
            adapter_id: Unique string key.
            config: Adapter configuration.
            checkpoint_path: Path to saved ``.pt`` file for lazy loading.
            adapter: Pre-built adapter instance (bypasses lazy loading).
        """
        self._registry[adapter_id] = NamedAdapter(
            adapter_id=adapter_id,
            config=config,
            adapter=adapter,
            checkpoint_path=checkpoint_path,
        )
        logger.info("AdapterBank: registered '%s'", adapter_id)

    def get(self, adapter_id: str) -> LatentAdapter:
        """Retrieve adapter, loading from disk if necessary.

        Args:
            adapter_id: Registered adapter key.

        Returns:
            Loaded and eval-mode :class:`LatentAdapter`.

        Raises:
            KeyError: If the adapter ID is not registered.
        """
        if adapter_id not in self._registry:
            raise KeyError(f"Adapter '{adapter_id}' not registered in AdapterBank.")

        named = self._registry[adapter_id]
        if named.adapter is None:
            if named.checkpoint_path and Path(named.checkpoint_path).exists():
                named.adapter = LatentAdapter(named.config)
                state = torch.load(named.checkpoint_path, map_location=self.device)
                named.adapter.load_state_dict(state)
                logger.info("AdapterBank: lazy-loaded '%s' from %s", adapter_id, named.checkpoint_path)
            else:
                logger.warning(
                    "AdapterBank: '%s' has no checkpoint, creating fresh instance.", adapter_id
                )
                named.adapter = LatentAdapter(named.config)
        named.adapter = named.adapter.to(self.device).eval()
        return named.adapter

    def save(self, adapter_id: str, save_dir: str) -> str:
        """Save an adapter's state dict to disk.

        Args:
            adapter_id: Key of the adapter to save.
            save_dir: Directory in which to create ``<adapter_id>.pt``.

        Returns:
            Full path to the saved file.
        """
        adapter = self.get(adapter_id)
        path = os.path.join(save_dir, f"{adapter_id}.pt")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(adapter.state_dict(), path)
        self._registry[adapter_id].checkpoint_path = path
        logger.info("AdapterBank: saved '%s' -> %s", adapter_id, path)
        return path

    def list_adapters(self) -> List[str]:
        """Return list of all registered adapter IDs."""
        return list(self._registry.keys())

    def __contains__(self, adapter_id: str) -> bool:
        return adapter_id in self._registry

    def __len__(self) -> int:
        return len(self._registry)


# ---------------------------------------------------------------------------
# Standalone training utilities
# ---------------------------------------------------------------------------

def train_adapter(
    adapter: LatentAdapter,
    source_states: Tensor,
    target_states: Tensor,
    n_epochs: int = 50,
    lr: float = 1e-3,
    loss_fn: Optional[Union[Callable[[Tensor, Tensor], Tensor], str]] = None,
    batch_size: int = 64,
    device: str = "cpu",
    verbose: bool = True,
    infonce_temperature: float = 0.07,
) -> List[float]:
    """Offline supervised training of a LatentAdapter.

    Trains the adapter to map ``source_states`` -> ``target_states`` using
    the given loss function (default: MSE).

    Args:
        adapter: :class:`LatentAdapter` to train in-place.
        source_states: Input tensor of shape (N, in_dim).
        target_states: Target tensor of shape (N, out_dim).
        n_epochs: Number of full passes over the data.
        lr: Learning rate for Adam.
        loss_fn: Loss function ``f(pred, target) -> scalar``, or the string
            ``'infonce'`` to use InfoNCE contrastive alignment.
            Defaults to ``F.mse_loss``.
        batch_size: Mini-batch size.
        device: PyTorch device string.
        verbose: Whether to print epoch-level loss.
        infonce_temperature: Temperature for InfoNCE (only used when
            ``loss_fn='infonce'``).

    Returns:
        List of per-epoch mean losses.
    """
    use_infonce = loss_fn == "infonce"
    if loss_fn is None or use_infonce:
        _loss_fn: Callable[[Tensor, Tensor], Tensor] = F.mse_loss
    else:
        _loss_fn = loss_fn  # type: ignore[assignment]

    dev = torch.device(device)
    adapter = adapter.to(dev).train()
    X = source_states.to(dev)
    Y = target_states.to(dev)
    N = X.shape[0]

    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    epoch_losses: List[float] = []

    for epoch in range(n_epochs):
        perm = torch.randperm(N, device=dev)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            x_b = X[idx]
            y_b = Y[idx]
            optimizer.zero_grad()
            pred = adapter(x_b)
            if use_infonce:
                loss = infonce_loss(pred, y_b, temperature=infonce_temperature)
            else:
                loss = _loss_fn(pred, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(mean_loss)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            logger.info("train_adapter epoch %03d/%03d | loss=%.6f", epoch + 1, n_epochs, mean_loss)

    adapter.eval()
    return epoch_losses


def compute_reconstruction_error(
    adapter: LatentAdapter,
    source: Tensor,
    target: Tensor,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate adapter reconstruction quality.

    Computes MSE and cosine similarity between the adapter output and
    the ground-truth target states.

    Args:
        adapter: Trained :class:`LatentAdapter` in eval mode.
        source: Source state tensor of shape (N, in_dim).
        target: Target state tensor of shape (N, out_dim).
        device: PyTorch device string.

    Returns:
        Dict with keys ``mse`` and ``cosine_similarity`` (mean over samples).
    """
    dev = torch.device(device)
    adapter = adapter.to(dev).eval()
    with torch.no_grad():
        pred = adapter(source.to(dev))
        target_d = target.to(dev)
        mse = F.mse_loss(pred, target_d).item()
        cos_sim = F.cosine_similarity(pred, target_d, dim=-1).mean().item()
        # Effective rank of predictions: exp(H(singular value distribution))
        try:
            sv = torch.linalg.svdvals(pred.float())
            sv_norm = sv / (sv.sum() + 1e-12)
            h = -(sv_norm * (sv_norm + 1e-12).log()).sum().item()
            eff_rank = float(torch.exp(torch.tensor(h)).item())
        except Exception:
            eff_rank = float("nan")
    logger.debug("Reconstruction error: MSE=%.6f, CosSim=%.4f, EffRank=%.2f", mse, cos_sim, eff_rank)
    return {"mse": mse, "cosine_similarity": cos_sim, "effective_rank": eff_rank}
