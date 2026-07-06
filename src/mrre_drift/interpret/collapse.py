"""
Language collapse detector.

Combines Logit Lens and CRAF signals into a unified CollapseProfile that:
  1. Identifies the layer at which English probability mass first exceeds a threshold
  2. Identifies the layer with peak English dominance
  3. Recommends safe intervention layers for Surgical MRRE
  4. Flags vision-fusion layers that must not be touched (set fraction=0.0 for text LLMs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

if TYPE_CHECKING:
    from mrre_drift.interpret.craf import CRAFProfile
    from mrre_drift.interpret.logit_lens import LogitLensScan



@dataclass
class CollapseProfile:
    """
    Unified language-collapse profile for a single model on a given language corpus.
    Derived by fusing Logit Lens and CRAF signals.
    """
    n_layers: int
    vision_fusion_cutoff: int            # layers < this index are vision-fusion layers
    collapse_onset_layer: Optional[int]  # first layer where english_mass > threshold
    peak_collapse_layer: int             # layer with maximum English dominance

    logit_lens_english_mass: Dict[int, float] = field(default_factory=dict)
    craf_alignment_delta: Dict[int, float] = field(default_factory=dict)

    safe_enhancement_layers: List[int] = field(default_factory=list)
    safe_anchoring_layers: List[int] = field(default_factory=list)

    def is_vision_fusion_layer(self, layer_id: int) -> bool:
        return layer_id < self.vision_fusion_cutoff

    def summary(self) -> str:
        lines = [
            f"Collapse Profile  (n_layers={self.n_layers})",
            f"  Vision fusion cutoff : layer {self.vision_fusion_cutoff}",
            f"  Collapse onset       : layer {self.collapse_onset_layer}",
            f"  Peak collapse layer  : layer {self.peak_collapse_layer}",
            f"  Enhancement layers   : {self.safe_enhancement_layers}",
            f"  Anchoring layers     : {self.safe_anchoring_layers}",
        ]
        return "\n".join(lines)


class CollapseDetector:
    """
    Fuses Logit Lens and CRAF signals to produce a CollapseProfile that guides
    Surgical MRRE layer selection.

    Parameters
    ----------
    n_layers                 : total transformer depth
    vision_fusion_fraction   : fraction of layers classified as vision-fusion
                               (set to 0.0 for text-only LLMs)
    collapse_threshold       : english_mass level that defines "collapse onset"
    anchoring_tail_fraction  : fraction of layers at the end targeted for Stage 2
    """

    def __init__(
        self,
        n_layers: int,
        vision_fusion_fraction: float = 0.30,
        collapse_threshold: float = 0.40,
        anchoring_tail_fraction: float = 0.25,
    ) -> None:
        self.n_layers = n_layers
        self.vision_fusion_fraction = vision_fusion_fraction
        self.collapse_threshold = collapse_threshold
        self.anchoring_tail_fraction = anchoring_tail_fraction

    @property
    def vision_fusion_cutoff(self) -> int:
        return int(self.n_layers * self.vision_fusion_fraction)

    @property
    def anchoring_start(self) -> int:
        return int(self.n_layers * (1.0 - self.anchoring_tail_fraction))

    def detect(
        self,
        logit_scan: "LogitLensScan",
        craf_profile: Optional["CRAFProfile"] = None,
    ) -> CollapseProfile:
        """
        Build a CollapseProfile from a Logit Lens scan and optional CRAF profile.

        The Logit Lens scan supplies per-layer english_mass values.  The CRAF
        profile (if provided) supplies alignment_delta values.  A layer is in
        collapse when either signal exceeds its threshold.
        """
        lens_en_mass = {r.layer_id: r.english_mass for r in logit_scan.layer_results}
        craf_deltas = craf_profile.alignment_deltas if craf_profile else {}

        cutoff = self.vision_fusion_cutoff
        onset: Optional[int] = None
        for lid in range(cutoff, self.n_layers):
            en_mass = lens_en_mass.get(lid, 0.0)
            delta = craf_deltas.get(lid, 0.0)
            if en_mass >= self.collapse_threshold or delta >= 0.05:
                onset = lid
                break

        candidate_layers = {
            lid: mass for lid, mass in lens_en_mass.items() if lid >= cutoff
        }
        peak = max(candidate_layers, key=candidate_layers.get) if candidate_layers else cutoff

        enh_start = onset if onset is not None else cutoff
        safe_enh = [
            lid for lid in range(enh_start, self.anchoring_start)
            if lid >= cutoff
        ]
        safe_anch = list(range(self.anchoring_start, self.n_layers))

        return CollapseProfile(
            n_layers=self.n_layers,
            vision_fusion_cutoff=cutoff,
            collapse_onset_layer=onset,
            peak_collapse_layer=peak,
            logit_lens_english_mass=lens_en_mass,
            craf_alignment_delta=craf_deltas,
            safe_enhancement_layers=safe_enh,
            safe_anchoring_layers=safe_anch,
        )

    def detect_from_scans(
        self,
        scans: Sequence["LogitLensScan"],
        craf_profile: Optional["CRAFProfile"] = None,
    ) -> CollapseProfile:
        """
        Detect collapse from multiple Logit Lens scans by averaging per-layer
        english_mass before running detection.
        """
        if not scans:
            raise ValueError("scans must not be empty")

        n_layers = len(scans[0].layer_results)
        averaged_mass: Dict[int, float] = {}
        for lid in range(n_layers):
            masses = [s.layer_results[lid].english_mass for s in scans if lid < len(s.layer_results)]
            averaged_mass[lid] = sum(masses) / len(masses) if masses else 0.0

        from mrre_drift.interpret.logit_lens import LayerLensResult, LogitLensScan as LLS
        avg_scan = LLS(text="[averaged]", layer_results=[
            LayerLensResult(
                layer_id=lid,
                top_tokens=[],
                english_mass=averaged_mass[lid],
                target_mass=0.0,
                entropy=0.0,
            )
            for lid in range(n_layers)
        ])
        return self.detect(avg_scan, craf_profile)
