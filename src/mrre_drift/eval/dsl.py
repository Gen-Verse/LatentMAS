"""
DSL (Distribution-over-Sequence-Length) correction for IFL rates.

Short generations can trivially stay in-script (a one-word answer), so a raw IFL rate is
confounded by the model's output-length distribution. The DSL corrector stratifies samples
into length bins, computes the IFL rate **within** each bin, then recombines them under a
fixed reference (calibration) length distribution. This yields a length-debiased IFL that is
comparable across conditions even when the intervention changes output length.

This is a real, deterministic reweighting of measured per-sample SFR flags — no values are
invented.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class DSLReport:
    """Length-stratified IFL correction."""

    length_bins: List[int]
    bin_ifl_rates: Dict[str, float] = field(default_factory=dict)   # "lo-hi" -> ifl
    bin_counts: Dict[str, int] = field(default_factory=dict)
    reference_weights: Dict[str, float] = field(default_factory=dict)
    raw_ifl_rate: float = 0.0
    corrected_ifl_rate: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class DSLCorrector:
    """Length-binned IFL correction.

    Parameters
    ----------
    length_bins
        Bin edges in characters, e.g. ``[0, 10, 50, 200]`` produces bins
        ``[0,10), [10,50), [50,200), [200, inf)``.
    """

    def __init__(self, length_bins: Sequence[int]) -> None:
        if not length_bins or sorted(length_bins) != list(length_bins):
            raise ValueError("length_bins must be a non-empty, ascending list of edges.")
        self.length_bins = list(length_bins)

    def _bin_label(self, idx: int) -> str:
        edges = self.length_bins
        lo = edges[idx]
        hi = edges[idx + 1] if idx + 1 < len(edges) else None
        return f"{lo}-{hi}" if hi is not None else f"{lo}+"

    def _bin_index(self, length: int) -> int:
        idx = 0
        for i, edge in enumerate(self.length_bins):
            if length >= edge:
                idx = i
        return idx

    def correct(
        self,
        lengths: Sequence[int],
        ifl_flags: Sequence[float],
        reference_weights: Optional[Dict[str, float]] = None,
    ) -> DSLReport:
        """Compute raw and length-corrected IFL.

        Parameters
        ----------
        lengths
            Per-sample output lengths (characters).
        ifl_flags
            Per-sample IFL failure flags (1.0 = failed to stay in-script).
        reference_weights
            Optional fixed bin weights (must sum > 0). When omitted, the corrected rate
            uses the *empirical* bin distribution of this sample (i.e. equals the raw rate),
            so a caller can pass a baseline distribution to debias a steered condition.
        """
        if len(lengths) != len(ifl_flags):
            raise ValueError("lengths and ifl_flags must be the same length.")
        n = len(lengths)
        if n == 0:
            return DSLReport(length_bins=self.length_bins)

        n_bins = len(self.length_bins)
        bin_fail = [0.0] * n_bins
        bin_count = [0] * n_bins
        for length, flag in zip(lengths, ifl_flags):
            b = self._bin_index(int(length))
            bin_fail[b] += float(flag)
            bin_count[b] += 1

        bin_ifl = {
            self._bin_label(i): (bin_fail[i] / bin_count[i]) if bin_count[i] else 0.0
            for i in range(n_bins)
        }
        counts = {self._bin_label(i): bin_count[i] for i in range(n_bins)}

        raw = sum(ifl_flags) / n

        # Reference distribution: provided weights, else empirical bin frequencies.
        if reference_weights:
            weights = {k: float(v) for k, v in reference_weights.items()}
        else:
            weights = {self._bin_label(i): bin_count[i] / n for i in range(n_bins)}

        wsum = sum(weights.values())
        corrected = 0.0
        if wsum > 0:
            for label, w in weights.items():
                corrected += (w / wsum) * bin_ifl.get(label, 0.0)

        return DSLReport(
            length_bins=self.length_bins,
            bin_ifl_rates=bin_ifl,
            bin_counts=counts,
            reference_weights={k: v / wsum for k, v in weights.items()} if wsum else weights,
            raw_ifl_rate=raw,
            corrected_ifl_rate=corrected,
        )
