import torch
import torch.nn as nn

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

class LatentDriftException(Exception):
    """Raised when semantic drift and language confusion is detected inside the continuous hub."""
    pass

class QueryReconstructionProbe(nn.Module):
    """Module E: Closed-Loop Test-Time Reconstruction Probe.

    Two decoder variants per strategy.md §4.4: ``arch='linear'`` (default —
    linear probes are competitive per precedent) and ``arch='mlp'`` (shallow
    two-layer GELU MLP), so the linear-vs-MLP comparison (ablation row 7e) is
    measured on this stack, not assumed.

    NOTE: the probe's ``decoder`` must be TRAINED (fit_decoder) on real
    (hub-state, query-embedding) pairs before its drift scores mean anything —
    a random-init linear map gives near-zero cosine similarity for every input,
    i.e. it flags 100% of clean states as drifted. ``forward`` refuses to gate
    on an untrained decoder for exactly that reason.
    """

    def __init__(
        self,
        hub_dim: int = 512,
        query_dim: int = 1024,
        tau_drift: float = 0.5,
        arch: str = "linear",
        mlp_hidden_dim: int = 256,
    ):
        super().__init__()
        if arch == "linear":
            self.decoder = nn.Linear(hub_dim, query_dim)
        elif arch == "mlp":
            self.decoder = nn.Sequential(
                nn.Linear(hub_dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(mlp_hidden_dim, query_dim),
            )
        else:
            raise ValueError(f"Unknown probe arch '{arch}'. Valid: 'linear', 'mlp'.")
        self.arch = arch
        self.hub_dim = hub_dim
        # Explicit attribute rather than reaching into decoder internals:
        # with arch='mlp' the decoder is a Sequential and has no out_features.
        self.query_dim = query_dim
        self.tau_drift = tau_drift
        self._is_fitted = False

    def fit_decoder(
        self,
        hub_states: torch.Tensor,
        query_embeddings: torch.Tensor,
        n_epochs: int = 100,
        lr: float = 1e-3,
    ) -> float:
        """Train the reconstruction decoder on real paired (z, q) data.

        Returns the final cosine-reconstruction loss.
        """
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        loss_val = float("nan")
        for _ in range(n_epochs):
            optimizer.zero_grad()
            q_rec = self.decoder(hub_states)
            loss = (1.0 - torch.nn.functional.cosine_similarity(q_rec, query_embeddings, dim=-1)).mean()
            loss.backward()
            optimizer.step()
            loss_val = float(loss.item())
        self._is_fitted = True
        return loss_val

    def forward(self, z_t: torch.Tensor, q_orig: torch.Tensor, raise_on_drift: bool = True):
        """Per-sample Test-Time Fidelity Drift Score (1 - cosine similarity).

        Raises :class:`LatentDriftException` when any sample in the batch exceeds
        ``tau_drift`` (the exception message reports how many, so a caller
        repairing per-sample can inspect the returned scores with
        ``raise_on_drift=False`` instead of losing the whole batch).
        """
        if not self._is_fitted:
            raise RuntimeError(
                "QueryReconstructionProbe.decoder is untrained; call fit_decoder() "
                "on real (hub_state, query_embedding) pairs first. An untrained "
                "probe flags every clean state as drifted."
            )
        q_rec = self.decoder(z_t)

        # Test-Time Fidelity Drift Score
        cos_sim = torch.nn.functional.cosine_similarity(q_rec, q_orig, dim=-1)
        drift_score = 1.0 - cos_sim

        # Error Mitigation
        drifted = drift_score > self.tau_drift
        if raise_on_drift and drifted.any():
            n_bad = int(drifted.sum().item())
            n_all = int(drifted.numel())
            raise LatentDriftException(
                f"Semantic drift detected: {n_bad}/{n_all} sample(s) exceeded "
                f"tau_drift={self.tau_drift}."
            )

        return drift_score
