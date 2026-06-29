"""
SVD-based subspace decomposition for disentangling language-specific and
reasoning-agnostic subspaces from multilingual hidden states.

The core idea: given paired (English, target-language) hidden state matrices,
compute the *contrastive* matrix (difference of representations) and perform
truncated SVD. The leading singular vectors of the contrastive matrix span
the **language-specific subspace** U_L; the orthogonal complement spans the
**reasoning subspace** U_R. Projecting onto U_R ablates language-surface
artifacts while preserving semantic content.
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DecompositionResult:
    """Stores all artefacts produced by a single SVD decomposition run.

    Attributes
    ----------
    U_L : Tensor
        Language-specific subspace basis, shape ``(hidden_dim, n_components)``.
        Columns are orthonormal.
    U_R : Tensor
        Reasoning (language-agnostic) subspace basis, shape
        ``(hidden_dim, hidden_dim - n_components)``.  Orthogonal complement of
        U_L in the full embedding space.
    P_L : Tensor
        Orthogonal projection matrix onto U_L,
        ``P_L = U_L @ U_L.T``, shape ``(hidden_dim, hidden_dim)``.
    singular_values : Tensor
        Singular values of the contrastive matrix in descending order.
    variance_explained : Tensor
        Fraction of total variance explained by each singular value.
    cumulative_variance : Tensor
        Cumulative variance explained.
    n_components : int
        Number of language-specific components extracted.
    hidden_dim : int
        Dimensionality of the hidden states used during fit.
    language_specific_variance_ratio : float
        Total fraction of contrastive variance captured by U_L.
    fit_metadata : Dict
        Extra metadata (number of samples, norms, etc.).
    """

    U_L: Tensor
    U_R: Tensor
    P_L: Tensor
    singular_values: Tensor
    variance_explained: Tensor
    cumulative_variance: Tensor
    n_components: int
    hidden_dim: int
    language_specific_variance_ratio: float
    fit_metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SVDSubspaceDecomposer:
    """Decompose multilingual hidden states into language-specific and
    reasoning-agnostic subspaces using contrastive truncated SVD.

    This class is **model-agnostic** — it operates purely on raw
    ``torch.Tensor`` objects of shape ``(n_samples, hidden_dim)``.

    Parameters
    ----------
    n_components : int, optional
        Default number of language-specific singular vectors to retain.
        Can be overridden in :meth:`fit`.  Defaults to 32.
    center : bool, optional
        Whether to mean-center each matrix before SVD.  Defaults to ``True``.
    device : str, optional
        Torch device for computations.  Defaults to ``"cpu"``.

    Examples
    --------
    >>> decomposer = SVDSubspaceDecomposer(n_components=16)
    >>> result = decomposer.fit(en_states, th_states)
    >>> projected = decomposer.project_to_reasoning(hidden)
    """

    def __init__(
        self,
        n_components: int = 32,
        center: bool = True,
        device: str = "cpu",
    ) -> None:
        self.n_components = n_components
        self.center = center
        self.device = torch.device(device)

        # Set after fit()
        self._result: Optional[DecompositionResult] = None
        logger.info(
            "SVDSubspaceDecomposer initialised | n_components=%d center=%s device=%s",
            n_components,
            center,
            device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        en_states: Tensor,
        tgt_states: Tensor,
        n_components: Optional[int] = None,
    ) -> DecompositionResult:
        """Fit the decomposition on paired (English, target) hidden states.

        The contrastive matrix ``C = tgt_states - en_states`` is constructed
        and subjected to truncated SVD.  The top-``n_components`` left singular
        vectors become U_L; U_R is their orthogonal complement within the
        full space.

        Parameters
        ----------
        en_states : Tensor
            English hidden states, shape ``(n_samples, hidden_dim)``.
        tgt_states : Tensor
            Target-language hidden states, shape ``(n_samples, hidden_dim)``.
            Must be aligned (sentence-level parallel pairs).
        n_components : int, optional
            Override the instance-level ``n_components``.

        Returns
        -------
        DecompositionResult
            Full decomposition artefacts.

        Raises
        ------
        ValueError
            If tensors have mismatched shapes or ``n_components`` exceeds rank.
        """
        n_comp = n_components if n_components is not None else self.n_components

        en_states = en_states.float().to(self.device)
        tgt_states = tgt_states.float().to(self.device)

        if en_states.shape != tgt_states.shape:
            raise ValueError(
                f"Shape mismatch: en_states {en_states.shape} vs "
                f"tgt_states {tgt_states.shape}"
            )

        n_samples, hidden_dim = en_states.shape

        if n_comp > min(n_samples, hidden_dim):
            raise ValueError(
                f"n_components={n_comp} exceeds rank={min(n_samples, hidden_dim)}"
            )

        logger.info(
            "Fitting SVD | n_samples=%d hidden_dim=%d n_components=%d",
            n_samples,
            hidden_dim,
            n_comp,
        )

        # Optional mean-centering per matrix
        if self.center:
            en_states = en_states - en_states.mean(dim=0, keepdim=True)
            tgt_states = tgt_states - tgt_states.mean(dim=0, keepdim=True)

        # Contrastive matrix: captures what *differs* between languages
        C = tgt_states - en_states  # (n_samples, hidden_dim)

        logger.debug(
            "Contrastive matrix stats | mean_norm=%.4f max_norm=%.4f",
            C.norm(dim=1).mean().item(),
            C.norm(dim=1).max().item(),
        )

        # Truncated SVD via torch.linalg.svd (economy form on the smaller dim)
        # We operate on C.T so U gives us hidden_dim basis vectors.
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        # U : (n_samples, k), S : (k,), Vh : (k, hidden_dim)
        # The right singular vectors Vh.T[:, :n_comp] span U_L in hidden space

        V = Vh.T  # (hidden_dim, k)
        U_L = V[:, :n_comp]  # (hidden_dim, n_comp) — language-specific basis

        # Projection matrix P_L = U_L @ U_L.T
        P_L = U_L @ U_L.T  # (hidden_dim, hidden_dim)

        # U_R: orthogonal complement — use full SVD truncated at n_comp
        U_R = V[:, n_comp:]  # (hidden_dim, hidden_dim - n_comp)

        # Variance explained
        total_variance = (S ** 2).sum()
        variance_explained = (S ** 2) / (total_variance + 1e-12)
        cumulative_variance = variance_explained.cumsum(dim=0)
        lang_var_ratio = variance_explained[:n_comp].sum().item()

        logger.info(
            "Decomposition complete | language_variance_ratio=%.4f top_sv=%.4f",
            lang_var_ratio,
            S[0].item() if len(S) > 0 else 0.0,
        )

        result = DecompositionResult(
            U_L=U_L.cpu(),
            U_R=U_R.cpu(),
            P_L=P_L.cpu(),
            singular_values=S.cpu(),
            variance_explained=variance_explained.cpu(),
            cumulative_variance=cumulative_variance.cpu(),
            n_components=n_comp,
            hidden_dim=hidden_dim,
            language_specific_variance_ratio=lang_var_ratio,
            fit_metadata={
                "n_samples": n_samples,
                "center": self.center,
                "contrastive_mean_norm": C.norm(dim=1).mean().item(),
                "contrastive_max_norm": C.norm(dim=1).max().item(),
                "en_mean_norm": en_states.norm(dim=1).mean().item(),
                "tgt_mean_norm": tgt_states.norm(dim=1).mean().item(),
            },
        )
        self._result = result
        return result

    def project_to_reasoning(self, hidden: Tensor) -> Tensor:
        """Project hidden states onto the reasoning subspace.

        Applies the orthogonal complement projection:
        ``h_proj = h - P_L @ (P_L.T @ h)``

        Because ``P_L`` is symmetric (``P_L = U_L @ U_L.T``), this simplifies
        to ``h_proj = (I - P_L) @ h``.

        Parameters
        ----------
        hidden : Tensor
            Hidden states to project, shape ``(..., hidden_dim)``.

        Returns
        -------
        Tensor
            Projected hidden states, same shape as input, with
            language-specific components ablated.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        self._require_fit()
        result = self._result
        P_L = result.P_L.to(hidden.device, dtype=hidden.dtype)

        # h_proj = h - P_L @ h  (shape-agnostic via einsum / matmul)
        orig_shape = hidden.shape
        h = hidden.reshape(-1, result.hidden_dim)
        lang_component = h @ P_L.T  # (batch, hidden_dim)
        h_proj = h - lang_component
        return h_proj.reshape(orig_shape)

    def compute_explained_variance(
        self,
        states: Tensor,
        decompose_on: str = "full",
    ) -> Dict[str, float]:
        """Measure variance captured by each subspace for a given set of states.

        Parameters
        ----------
        states : Tensor
            Hidden states, shape ``(n_samples, hidden_dim)``.
        decompose_on : str
            One of ``"language"``, ``"reasoning"``, or ``"full"``.

        Returns
        -------
        Dict[str, float]
            Dictionary with keys:
            - ``"language_var"``: variance in U_L subspace
            - ``"reasoning_var"``: variance in U_R subspace
            - ``"total_var"``: total variance
            - ``"language_ratio"``: fraction in language subspace
            - ``"reasoning_ratio"``: fraction in reasoning subspace
        """
        self._require_fit()
        result = self._result
        states = states.float().to(self.device)
        if self.center:
            states = states - states.mean(dim=0, keepdim=True)

        U_L = result.U_L.to(self.device)
        U_R = result.U_R.to(self.device)

        # Project onto each subspace
        proj_L = states @ U_L  # (n, n_comp)
        proj_R = states @ U_R  # (n, hidden_dim - n_comp)

        var_L = proj_L.var(dim=0).sum().item()
        var_R = proj_R.var(dim=0).sum().item()
        total_var = var_L + var_R + 1e-12

        return {
            "language_var": var_L,
            "reasoning_var": var_R,
            "total_var": total_var,
            "language_ratio": var_L / total_var,
            "reasoning_ratio": var_R / total_var,
        }

    def get_result(self) -> DecompositionResult:
        """Return the stored :class:`DecompositionResult`.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        self._require_fit()
        return self._result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Persist the fitted decomposition to disk.

        Parameters
        ----------
        path : Path or str
            File path ending in ``.pt`` or ``.ckpt``.
        """
        self._require_fit()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_components": self.n_components,
            "center": self.center,
            "device": str(self.device),
            "result": {
                "U_L": self._result.U_L,
                "U_R": self._result.U_R,
                "P_L": self._result.P_L,
                "singular_values": self._result.singular_values,
                "variance_explained": self._result.variance_explained,
                "cumulative_variance": self._result.cumulative_variance,
                "n_components": self._result.n_components,
                "hidden_dim": self._result.hidden_dim,
                "language_specific_variance_ratio": self._result.language_specific_variance_ratio,
                "fit_metadata": self._result.fit_metadata,
            },
        }
        torch.save(payload, path)
        logger.info("SVDSubspaceDecomposer saved to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "SVDSubspaceDecomposer":
        """Load a fitted decomposer from disk.

        Parameters
        ----------
        path : Path or str
            Path to a file previously saved with :meth:`save`.

        Returns
        -------
        SVDSubspaceDecomposer
            Reconstructed instance with the stored result attached.
        """
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        instance = cls(
            n_components=payload["n_components"],
            center=payload["center"],
            device=payload["device"],
        )
        r = payload["result"]
        instance._result = DecompositionResult(
            U_L=r["U_L"],
            U_R=r["U_R"],
            P_L=r["P_L"],
            singular_values=r["singular_values"],
            variance_explained=r["variance_explained"],
            cumulative_variance=r["cumulative_variance"],
            n_components=r["n_components"],
            hidden_dim=r["hidden_dim"],
            language_specific_variance_ratio=r["language_specific_variance_ratio"],
            fit_metadata=r["fit_metadata"],
        )
        logger.info("SVDSubspaceDecomposer loaded from %s", path)
        return instance

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def select_n_components_by_variance(
        self,
        target_variance: float = 0.90,
    ) -> int:
        """Return the minimum n_components that explains target_variance.

        Parameters
        ----------
        target_variance : float
            Cumulative variance threshold in ``(0, 1)``.  Defaults to 0.90.

        Returns
        -------
        int
            Minimum index k such that cumulative_variance[k] >= target_variance.
        """
        self._require_fit()
        cum_var = self._result.cumulative_variance.numpy()
        indices = np.where(cum_var >= target_variance)[0]
        if len(indices) == 0:
            return len(cum_var)
        n = int(indices[0]) + 1
        logger.info(
            "Components for %.0f%% variance: %d", target_variance * 100, n
        )
        return n

    def summary(self) -> str:
        """Return a human-readable summary of the decomposition."""
        self._require_fit()
        r = self._result
        lines = [
            "=" * 60,
            "SVDSubspaceDecomposer — Decomposition Summary",
            "=" * 60,
            f"  hidden_dim               : {r.hidden_dim}",
            f"  n_components (U_L)       : {r.n_components}",
            f"  U_R dim                  : {r.U_R.shape[1]}",
            f"  language variance ratio  : {r.language_specific_variance_ratio:.4f}",
            f"  top singular value       : {r.singular_values[0].item():.4f}",
            f"  50% cumulative var @ k   : {(r.cumulative_variance >= 0.5).nonzero(as_tuple=True)[0][0].item() + 1}",
            f"  90% cumulative var @ k   : {(r.cumulative_variance >= 0.9).nonzero(as_tuple=True)[0][0].item() + 1}",
            "=" * 60,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_fit(self) -> None:
        if self._result is None:
            raise RuntimeError(
                "SVDSubspaceDecomposer has not been fitted yet. "
                "Call fit(en_states, tgt_states) first."
            )
