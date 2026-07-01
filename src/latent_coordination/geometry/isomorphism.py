"""
Geometric Isomorphism Analysis for cross-lingual representation spaces.

Measures the structural alignment between English and target-language hidden
state geometries using:
  - Centered Kernel Alignment (CKA)
  - Representational Similarity Analysis (RSA / RDM Spearman)
  - Orthogonal Procrustes alignment
  - Magnitude Distortion Ratio (MDR)

These metrics together quantify the *Geometric Isomorphism Hypothesis*:
that multilingual LLMs maintain structurally similar (but magnitude-skewed)
representation geometries across languages.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
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


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class IsomorphismReport:
    """Comprehensive isomorphism metrics for a single language pair.

    Attributes
    ----------
    language : str
        ISO 639-1 code of the target language compared against English.
    cka : float
        Centered Kernel Alignment score in ``[0, 1]`` (1 = identical geometry).
    rsa_spearman : float
        Spearman correlation of the representational dissimilarity matrices.
    procrustes_error : float
        Residual Frobenius norm after optimal orthogonal alignment.
    procrustes_disparity : float
        Normalised Procrustes disparity (0 = perfect alignment).
    magnitude_distortion_ratio : float
        Mean ratio ``‖h_en‖ / ‖h_tgt‖`` across samples.
    magnitude_distortion_per_sample : np.ndarray
        Per-sample MDR values.
    n_samples : int
        Number of paired samples used.
    hidden_dim : int
        Dimensionality of hidden states.
    extra : Dict
        Any additional computed values.
    """

    language: str
    cka: float
    rsa_spearman: float
    procrustes_error: float
    procrustes_disparity: float
    magnitude_distortion_ratio: float
    magnitude_distortion_per_sample: np.ndarray
    n_samples: int
    hidden_dim: int
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "language": self.language,
            "cka": self.cka,
            "rsa_spearman": self.rsa_spearman,
            "procrustes_error": self.procrustes_error,
            "procrustes_disparity": self.procrustes_disparity,
            "magnitude_distortion_ratio": self.magnitude_distortion_ratio,
            "n_samples": self.n_samples,
            "hidden_dim": self.hidden_dim,
            **self.extra,
        }

    def summary(self) -> str:
        lines = [
            f"IsomorphismReport [{self.language}]",
            f"  CKA                : {self.cka:.4f}",
            f"  RSA (Spearman ρ)   : {self.rsa_spearman:.4f}",
            f"  Procrustes disparity: {self.procrustes_disparity:.4f}",
            f"  Magnitude Distortion: {self.magnitude_distortion_ratio:.4f}",
            f"  Samples            : {self.n_samples}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GeometricIsomorphismAnalyzer:
    """Compute geometric isomorphism metrics between multilingual representation spaces.

    All methods accept raw ``torch.Tensor`` matrices and are model-agnostic.

    Parameters
    ----------
    device : str, optional
        Torch device for tensor operations.  Defaults to ``"cpu"``.
    kernel : str, optional
        Kernel for CKA computation: ``"linear"`` or ``"rbf"``.
        Defaults to ``"linear"``.
    rbf_sigma : float, optional
        Bandwidth for RBF kernel when ``kernel="rbf"``.  Defaults to 1.0.
    """

    def __init__(
        self,
        device: str = "cpu",
        kernel: str = "linear",
        rbf_sigma: float = 1.0,
    ) -> None:
        self.device = torch.device(device)
        self.kernel = kernel
        self.rbf_sigma = rbf_sigma
        logger.info(
            "GeometricIsomorphismAnalyzer | device=%s kernel=%s", device, kernel
        )

    # ------------------------------------------------------------------
    # CKA
    # ------------------------------------------------------------------

    def compute_cka(self, X: Tensor, Y: Tensor) -> float:
        """Compute Centered Kernel Alignment (CKA) between two representation matrices.

        Uses the unbiased HSIC estimator from Kornblith et al. (2019).

        Parameters
        ----------
        X : Tensor
            Representation matrix, shape ``(n_samples, d_X)``.
        Y : Tensor
            Representation matrix, shape ``(n_samples, d_Y)``.

        Returns
        -------
        float
            CKA similarity in ``[0, 1]``.  Higher means more similar geometry.

        Notes
        -----
        For linear CKA, this is equivalent to the normalised Frobenius inner
        product of the Gram matrices after centering.
        """
        X = X.float().to(self.device)
        Y = Y.float().to(self.device)

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Sample count mismatch: X has {X.shape[0]} rows, Y has {Y.shape[0]}."
            )

        if self.kernel == "linear":
            K = X @ X.T
            L = Y @ Y.T
        elif self.kernel == "rbf":
            K = self._rbf_kernel(X)
            L = self._rbf_kernel(Y)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        # Centre the Gram matrices
        K_c = self._centre_kernel(K)
        L_c = self._centre_kernel(L)

        hsic_kl = (K_c * L_c).sum() / ((X.shape[0] - 1) ** 2)
        hsic_kk = (K_c * K_c).sum() / ((X.shape[0] - 1) ** 2)
        hsic_ll = (L_c * L_c).sum() / ((X.shape[0] - 1) ** 2)

        denom = torch.sqrt(hsic_kk * hsic_ll).clamp(min=1e-12)
        cka = (hsic_kl / denom).item()
        cka = float(np.clip(cka, 0.0, 1.0))

        logger.debug("CKA = %.4f", cka)
        return cka

    # ------------------------------------------------------------------
    # RSA
    # ------------------------------------------------------------------

    def compute_rsa(self, X: Tensor, Y: Tensor) -> float:
        """Representational Similarity Analysis via Spearman correlation of RDMs.

        Constructs pairwise Euclidean distance matrices (RDMs) for X and Y,
        then computes the Spearman rank correlation between their upper
        triangular elements.

        Parameters
        ----------
        X : Tensor
            Representation matrix, shape ``(n_samples, d)``.
        Y : Tensor
            Representation matrix, shape ``(n_samples, d')``.

        Returns
        -------
        float
            Spearman correlation in ``[-1, 1]``.
        """
        X_np = X.float().cpu().numpy()
        Y_np = Y.float().cpu().numpy()

        rdm_X = squareform(pdist(X_np, metric="euclidean"))
        rdm_Y = squareform(pdist(Y_np, metric="euclidean"))

        # Upper triangle (excluding diagonal)
        n = rdm_X.shape[0]
        idx = np.triu_indices(n, k=1)
        rho, _ = spearmanr(rdm_X[idx], rdm_Y[idx])

        logger.debug("RSA Spearman ρ = %.4f", rho)
        return float(rho)

    # ------------------------------------------------------------------
    # Procrustes
    # ------------------------------------------------------------------

    def compute_procrustes(
        self, X: Tensor, Y: Tensor
    ) -> Tuple[float, float, Tensor]:
        """Orthogonal Procrustes alignment of Y onto X.

        Finds the orthogonal matrix Q that minimises ``‖X - Y @ Q‖_F`` and
        returns the residual error and normalised disparity.

        Parameters
        ----------
        X : Tensor
            Reference representation matrix, shape ``(n, d)``.
        Y : Tensor
            Matrix to align, shape ``(n, d)``.

        Returns
        -------
        error : float
            Frobenius norm of the residual ``X - Y @ Q``.
        disparity : float
            Normalised disparity in ``[0, 1]`` (0 = perfect alignment).
        Q : Tensor
            Optimal rotation matrix, shape ``(d, d)``.
        """
        X = X.float().to(self.device)
        Y = Y.float().to(self.device)

        # Standardise (zero-mean, unit Frobenius norm)
        X_n = X - X.mean(dim=0, keepdim=True)
        Y_n = Y - Y.mean(dim=0, keepdim=True)
        X_n = X_n / (X_n.norm() + 1e-12)
        Y_n = Y_n / (Y_n.norm() + 1e-12)

        # Procrustes: M = X.T @ Y, SVD(M) = U S V.T, Q = V U.T
        M = X_n.T @ Y_n
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        Q = Vh.T @ U.T

        # Residual
        Y_aligned = Y_n @ Q
        residual = (X_n - Y_aligned).norm(p="fro").item()

        # Disparity: 1 - sum(S)^2
        disparity = max(0.0, 1.0 - (S.sum().item() ** 2))

        logger.debug(
            "Procrustes | error=%.4f disparity=%.4f", residual, disparity
        )
        return residual, disparity, Q.cpu()

    # ------------------------------------------------------------------
    # Magnitude Distortion
    # ------------------------------------------------------------------

    def compute_magnitude_distortion_ratio(
        self, en_states: Tensor, lrl_states: Tensor
    ) -> Tuple[float, np.ndarray]:
        """Compute per-sample and mean magnitude distortion ratio.

        Quantifies the *Magnitude Distortion Paradox*: the observation that
        structurally similar LRL representations nevertheless have
        significantly different L2 norms compared to English.

        Parameters
        ----------
        en_states : Tensor
            English hidden states, shape ``(n_samples, hidden_dim)``.
        lrl_states : Tensor
            Low-resource language hidden states, same shape.

        Returns
        -------
        mean_ratio : float
            Mean of ``‖h_en_i‖ / ‖h_lrl_i‖`` across samples.
        per_sample : np.ndarray
            Array of per-sample ratios, shape ``(n_samples,)``.
        """
        en_norms = en_states.float().norm(dim=1).cpu().numpy()
        lrl_norms = lrl_states.float().norm(dim=1).cpu().numpy()

        per_sample = en_norms / (lrl_norms + 1e-12)
        mean_ratio = float(per_sample.mean())

        logger.debug(
            "MDR | mean=%.4f std=%.4f min=%.4f max=%.4f",
            per_sample.mean(),
            per_sample.std(),
            per_sample.min(),
            per_sample.max(),
        )
        return mean_ratio, per_sample

    # ------------------------------------------------------------------
    # Omnibus compute
    # ------------------------------------------------------------------

    def compute_all(
        self,
        en_states: Tensor,
        tgt_states: Tensor,
        language_name: str,
    ) -> IsomorphismReport:
        """Compute all isomorphism metrics and return a consolidated report.

        Parameters
        ----------
        en_states : Tensor
            English hidden states, shape ``(n_samples, hidden_dim)``.
        tgt_states : Tensor
            Target-language hidden states, same shape.
        language_name : str
            ISO 639-1 or human-readable language name.

        Returns
        -------
        IsomorphismReport
            Full report with all metrics populated.
        """
        logger.info(
            "Computing full isomorphism report for language '%s' | n=%d dim=%d",
            language_name,
            en_states.shape[0],
            en_states.shape[1],
        )

        cka = self.compute_cka(en_states, tgt_states)
        rsa = self.compute_rsa(en_states, tgt_states)
        proc_error, proc_disp, _ = self.compute_procrustes(en_states, tgt_states)
        mdr_mean, mdr_per_sample = self.compute_magnitude_distortion_ratio(
            en_states, tgt_states
        )

        report = IsomorphismReport(
            language=language_name,
            cka=cka,
            rsa_spearman=rsa,
            procrustes_error=proc_error,
            procrustes_disparity=proc_disp,
            magnitude_distortion_ratio=mdr_mean,
            magnitude_distortion_per_sample=mdr_per_sample,
            n_samples=en_states.shape[0],
            hidden_dim=en_states.shape[1],
            extra={
                "mdr_std": float(mdr_per_sample.std()),
                "mdr_min": float(mdr_per_sample.min()),
                "mdr_max": float(mdr_per_sample.max()),
            },
        )

        logger.info(report.summary())
        return report

    def compute_pairwise(
        self,
        states_by_language: Dict[str, Tensor],
        metric: str = "cka",
    ) -> np.ndarray:
        """Compute a pairwise isomorphism matrix for multiple languages.

        Parameters
        ----------
        states_by_language : Dict[str, Tensor]
            Mapping from language code to hidden state matrix.
        metric : str
            One of ``"cka"``, ``"rsa"``, ``"procrustes_disparity"``.

        Returns
        -------
        np.ndarray
            Square matrix of shape ``(n_langs, n_langs)`` with pairwise scores.
        """
        langs = list(states_by_language.keys())
        n = len(langs)
        matrix = np.zeros((n, n))

        for i, lang_i in enumerate(langs):
            for j, lang_j in enumerate(langs):
                if i == j:
                    matrix[i, j] = 1.0 if metric != "procrustes_disparity" else 0.0
                    continue
                if j < i:
                    matrix[i, j] = matrix[j, i]
                    continue
                X = states_by_language[lang_i]
                Y = states_by_language[lang_j]
                if metric == "cka":
                    val = self.compute_cka(X, Y)
                elif metric == "rsa":
                    val = self.compute_rsa(X, Y)
                elif metric == "procrustes_disparity":
                    _, val, _ = self.compute_procrustes(X, Y)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                matrix[i, j] = val

        return matrix, langs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _centre_kernel(self, K: Tensor) -> Tensor:
        """Double-centre a kernel (Gram) matrix."""
        n = K.shape[0]
        H = torch.eye(n, device=K.device, dtype=K.dtype) - (1.0 / n)
        return H @ K @ H

    def _rbf_kernel(self, X: Tensor) -> Tensor:
        """RBF (Gaussian) kernel matrix."""
        sq_dists = (
            (X.unsqueeze(1) - X.unsqueeze(0)).pow(2).sum(dim=-1)
        )
        return torch.exp(-sq_dists / (2.0 * self.rbf_sigma ** 2))
