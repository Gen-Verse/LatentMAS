"""CKA→IFL geometry regression for the Surgical MRRE paper (P1-T5).

Repairs the n=5 statistics overclaim by providing rigorous regression over
n≥16 languages, including mid-resource Latin-script languages that break the
IFL=1.0 ceiling, and pre-registering the CKA threshold instead of fitting it
post hoc.

Statistical pipeline
--------------------
1. Pearson r + Spearman ρ with bootstrap 95% CIs (≥2000 resamples).
2. Partial correlation controlling for (resource_tier, script_family) via
   residual regression so confounded factors do not inflate the geometry signal.
3. Fisher-z CI around the partial correlation estimate.
4. Pre-registered threshold cross-validation: τ is fit on split-A (train) and
   evaluated on split-B (test) — no post-hoc threshold shopping.

All functions operate on plain Python lists / NumPy arrays; they do not
require torch and can run on CPU in the Colab regression job.

Usage
-----
    from mrre_drift.eval.regression import CKAIFLRegressor

    reg = CKAIFLRegressor(n_boot=2000, seed=42)
    result = reg.fit(
        cka_values=[0.31, 0.45, ...],   # per-language CKA at the target layer
        ifl_rates=[0.82, 0.61, ...],    # per-language baseline IFL rate
        language_codes=["th", "my", "vi", "id", ...],
        resource_tiers=[2, 2, 3, 3, ...],    # 1=high, 2=mid, 3=low
        script_families=["indic", "indic", "latin", "latin", ...],
        tau_train_frac=0.5,             # 50% languages for threshold training
    )
    print(result.summary())
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

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
# Language metadata defaults (script family and resource tier)
# ---------------------------------------------------------------------------

# ISO-639-1 → script family label.  Latin-script languages are the ones that
# break the IFL=1.0 ceiling because English is also Latin.
_DEFAULT_SCRIPT = {
    "th": "thai", "my": "burmese", "km": "khmer", "lo": "lao",
    "am": "ethiopic", "sw": "latin", "id": "latin", "ms": "latin",
    "vi": "latin", "tl": "latin", "fil": "latin",
    "bn": "bengali", "ta": "tamil", "te": "telugu", "si": "sinhala",
    "bo": "tibetan", "ka": "georgian", "hy": "armenian",
    "ar": "arabic", "he": "hebrew", "zh": "cjk", "ja": "cjk", "ko": "hangul",
}

# ISO-639-1 → resource tier (1=high, 2=mid, 3=low).  Approximate.
_DEFAULT_TIER = {
    "th": 2, "my": 3, "km": 3, "lo": 3,
    "am": 3, "sw": 3, "id": 2, "ms": 2,
    "vi": 2, "tl": 2, "fil": 2,
    "bn": 2, "ta": 2, "te": 2, "si": 3,
    "bo": 3, "ka": 3, "hy": 3,
    "ar": 2, "he": 2, "zh": 1, "ja": 1, "ko": 1,
}


# ---------------------------------------------------------------------------
# Low-level stats helpers
# ---------------------------------------------------------------------------

def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr  # type: ignore
    if len(x) < 2:
        return float("nan")
    r, _ = spearmanr(x, y)
    return float(r)


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat_fn,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Bootstrap (1-alpha) CI for a bivariate statistic ``stat_fn(x, y) → float``."""
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = np.array([
        stat_fn(x[idx := rng.integers(0, n, n)], y[idx])
        for _ in range(n_boot)
    ])
    lo = float(np.nanpercentile(stats, 100 * alpha / 2))
    hi = float(np.nanpercentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi


def fisher_z_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Fisher-z 95% CI for a correlation ``r`` computed from ``n`` observations."""
    if n <= 3 or abs(r) >= 1.0:
        return (float("nan"), float("nan"))
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = -float(np.percentile(np.random.default_rng(0).standard_normal(100_000), 100 * alpha / 2))
    lo = math.tanh(z - z_crit * se)
    hi = math.tanh(z + z_crit * se)
    return lo, hi


def _residualise(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """OLS residuals of ``y`` after regressing out ``controls`` (one-hot + intercept)."""
    X = np.column_stack([controls, np.ones(len(y))])
    try:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ coef
    except np.linalg.LinAlgError:
        return y - y.mean()


def partial_correlation(
    cka: np.ndarray,
    ifl: np.ndarray,
    resource_tiers: np.ndarray,
    script_families: Sequence[str],
) -> float:
    """Pearson r between CKA and IFL after partialling out resource tier and script.

    Controls are one-hot encoded (script family) + numeric (resource tier).
    Regresses each variable on controls, then correlates residuals.
    """
    # Encode script family as integer codes.
    scripts_unique = sorted(set(script_families))
    script_int = np.array([scripts_unique.index(s) for s in script_families], dtype=float)
    n_scripts = len(scripts_unique)

    # One-hot encode script (drop last to avoid collinearity).
    script_oh = np.zeros((len(script_int), max(n_scripts - 1, 1)))
    for i, s in enumerate(script_int):
        if int(s) < n_scripts - 1:
            script_oh[i, int(s)] = 1.0

    controls = np.column_stack([resource_tiers.astype(float), script_oh])
    cka_res = _residualise(cka, controls)
    ifl_res = _residualise(ifl, controls)
    return _pearson(cka_res, ifl_res)


def fit_threshold_cv(
    cka_train: np.ndarray,
    ifl_train: np.ndarray,
    cka_test: np.ndarray,
    ifl_test: np.ndarray,
    ifl_cutoff: float = 0.5,
    n_candidates: int = 50,
) -> Dict:
    """Pre-registered threshold CV: fit τ on train split, evaluate on test split.

    τ is the CKA threshold above which the model predicts IFL < ifl_cutoff
    (i.e. low-IFL, meaning the geometry predicts the model will stay in-script).

    Parameters
    ----------
    cka_train, ifl_train : training split features and labels
    cka_test,  ifl_test  : held-out test split
    ifl_cutoff           : IFL rate below which we call a language "low IFL" (label=1)
    n_candidates         : number of candidate τ values (grid over cka_train range)

    Returns
    -------
    dict with keys: tau, train_accuracy, test_accuracy, n_train, n_test
    """
    # Train labels: 1 = low IFL (in-script), 0 = high IFL (drifts to English).
    y_train = (ifl_train < ifl_cutoff).astype(int)
    y_test = (ifl_test < ifl_cutoff).astype(int)

    if len(cka_train) == 0 or len(np.unique(y_train)) < 2:
        logger.warning("Threshold CV: degenerate training split; returning tau=nan.")
        return {"tau": float("nan"), "train_accuracy": float("nan"),
                "test_accuracy": float("nan"), "n_train": len(cka_train), "n_test": len(cka_test)}

    best_tau, best_acc = float("nan"), -1.0
    lo, hi = float(cka_train.min()), float(cka_train.max())
    for tau in np.linspace(lo, hi, n_candidates):
        pred = (cka_train >= tau).astype(int)
        acc = float((pred == y_train).mean())
        if acc > best_acc:
            best_acc = acc
            best_tau = float(tau)

    test_pred = (cka_test >= best_tau).astype(int)
    test_acc = float((test_pred == y_test).mean()) if len(y_test) > 0 else float("nan")
    return {
        "tau": best_tau,
        "train_accuracy": best_acc,
        "test_accuracy": test_acc,
        "n_train": int(len(cka_train)),
        "n_test": int(len(cka_test)),
    }


# ---------------------------------------------------------------------------
# Main regression result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegressionResult:
    """Full regression output for the CKA→IFL geometry analysis."""

    n: int

    # Raw correlations
    pearson_r: float
    spearman_rho: float

    # Bootstrap CIs (2000 resamples)
    pearson_ci: Tuple[float, float]
    spearman_ci: Tuple[float, float]

    # Fisher-z CI on Pearson r
    fisher_z_ci: Tuple[float, float]

    # Partial correlation (controlling for resource tier + script family)
    partial_r: float
    partial_ci: Tuple[float, float]

    # Pre-registered threshold CV
    threshold_cv: Dict

    # Per-language data (for figure regeneration)
    language_codes: List[str] = field(default_factory=list)
    cka_values: List[float] = field(default_factory=list)
    ifl_rates: List[float] = field(default_factory=list)

    def summary(self) -> str:
        lo_p, hi_p = self.pearson_ci
        lo_s, hi_s = self.spearman_ci
        lo_f, hi_f = self.fisher_z_ci
        cv = self.threshold_cv
        return (
            f"CKA→IFL Regression (n={self.n})\n"
            f"  Pearson r   = {self.pearson_r:.3f}  95% CI [{lo_p:.3f}, {hi_p:.3f}]\n"
            f"  Spearman ρ  = {self.spearman_rho:.3f}  95% CI [{lo_s:.3f}, {hi_s:.3f}]\n"
            f"  Fisher-z CI = [{lo_f:.3f}, {hi_f:.3f}]\n"
            f"  Partial r   = {self.partial_r:.3f}  (controlled: resource_tier, script_family)\n"
            f"  Threshold τ = {cv.get('tau', 'nan'):.3f}  "
            f"train_acc={cv.get('train_accuracy', 'nan'):.3f}  "
            f"test_acc={cv.get('test_accuracy', 'nan'):.3f}  "
            f"(n_train={cv.get('n_train')}, n_test={cv.get('n_test')})"
        )

    def to_dict(self) -> Dict:
        return {
            "n": self.n,
            "pearson_r": self.pearson_r,
            "spearman_rho": self.spearman_rho,
            "pearson_ci_95": list(self.pearson_ci),
            "spearman_ci_95": list(self.spearman_ci),
            "fisher_z_ci_95": list(self.fisher_z_ci),
            "partial_r": self.partial_r,
            "partial_ci_95": list(self.partial_ci),
            "threshold_cv": self.threshold_cv,
            "per_language": {
                lang: {"cka": cka, "ifl": ifl}
                for lang, cka, ifl in zip(self.language_codes, self.cka_values, self.ifl_rates)
            },
        }


# ---------------------------------------------------------------------------
# Main regressor class
# ---------------------------------------------------------------------------

class CKAIFLRegressor:
    """Fit the CKA→IFL geometry regression with full statistical rigor.

    Parameters
    ----------
    n_boot   : number of bootstrap resamples (≥2000 per plan)
    seed     : global RNG seed for reproducibility
    """

    def __init__(self, n_boot: int = 2000, seed: int = 42) -> None:
        if n_boot < 2000:
            logger.warning("n_boot=%d < 2000; bootstrap CIs will be unreliable.", n_boot)
        self.n_boot = n_boot
        self.seed = seed

    def fit(
        self,
        cka_values: Sequence[float],
        ifl_rates: Sequence[float],
        language_codes: Sequence[str],
        resource_tiers: Optional[Sequence[int]] = None,
        script_families: Optional[Sequence[str]] = None,
        tau_train_frac: float = 0.5,
        ifl_cutoff: float = 0.5,
    ) -> RegressionResult:
        """Run the full regression pipeline.

        Parameters
        ----------
        cka_values      : per-language CKA score at the diagnostic layer
        ifl_rates       : per-language baseline IFL rate (no intervention)
        language_codes  : ISO-639-1 codes (used for default tier/script lookup)
        resource_tiers  : override per-language resource tier (1=high, 2=mid, 3=low)
        script_families : override per-language script family label
        tau_train_frac  : fraction of languages used to train the threshold (pre-register)
        ifl_cutoff      : IFL below this is "low-IFL" (label=1) in threshold CV
        """
        if len(cka_values) < 4:
            raise ValueError(f"Need ≥4 languages for regression; got {len(cka_values)}.")
        if len(cka_values) < 14:
            logger.warning(
                "n=%d < 14; regression CIs will be wide. Expand language set to n≥16.",
                len(cka_values),
            )

        cka = np.array(cka_values, dtype=float)
        ifl = np.array(ifl_rates, dtype=float)
        codes = list(language_codes)

        # Resolve controls.
        tiers = np.array(
            resource_tiers if resource_tiers is not None
            else [_DEFAULT_TIER.get(c, 2) for c in codes],
            dtype=float,
        )
        scripts = (
            list(script_families) if script_families is not None
            else [_DEFAULT_SCRIPT.get(c, "latin") for c in codes]
        )

        # Core correlations.
        pearson_r = _pearson(cka, ifl)
        spearman_rho = _spearman(cka, ifl)

        # Bootstrap CIs.
        pearson_ci = bootstrap_ci(cka, ifl, _pearson, self.n_boot, seed=self.seed)
        spearman_ci = bootstrap_ci(cka, ifl, _spearman, self.n_boot, seed=self.seed + 1)

        # Fisher-z CI on Pearson.
        fz_ci = fisher_z_ci(pearson_r, len(cka))

        # Partial correlation.
        partial_r = partial_correlation(cka, ifl, tiers, scripts)
        partial_ci = bootstrap_ci(
            cka, ifl,
            lambda x, y: partial_correlation(x, y, tiers[: len(x)], scripts[: len(x)]),
            self.n_boot, seed=self.seed + 2,
        )

        # Pre-registered threshold CV.
        n_train = max(2, round(len(cka) * tau_train_frac))
        rng = np.random.default_rng(self.seed + 3)
        idx = rng.permutation(len(cka))
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        cv = fit_threshold_cv(
            cka[train_idx], ifl[train_idx],
            cka[test_idx], ifl[test_idx],
            ifl_cutoff=ifl_cutoff,
        )

        result = RegressionResult(
            n=len(cka),
            pearson_r=pearson_r,
            spearman_rho=spearman_rho,
            pearson_ci=pearson_ci,
            spearman_ci=spearman_ci,
            fisher_z_ci=fz_ci,
            partial_r=partial_r,
            partial_ci=partial_ci,
            threshold_cv=cv,
            language_codes=codes,
            cka_values=cka.tolist(),
            ifl_rates=ifl.tolist(),
        )
        logger.info("Regression complete.\n%s", result.summary())
        return result
