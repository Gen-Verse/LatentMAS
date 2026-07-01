"""Tests for the CollapseDetector and CollapseProfile."""

import pytest
from mrre_drift.interpret.collapse import CollapseDetector, CollapseProfile
from mrre_drift.interpret.logit_lens import LayerLensResult, LogitLensScan

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"



def _make_scan(english_masses: list) -> LogitLensScan:
    results = [
        LayerLensResult(layer_id=i, top_tokens=[], english_mass=m, target_mass=1.0 - m, entropy=1.0)
        for i, m in enumerate(english_masses)
    ]
    return LogitLensScan(text="synthetic", layer_results=results)


class TestCollapseDetector:
    def test_vision_fusion_cutoff_fraction(self):
        det = CollapseDetector(n_layers=20, vision_fusion_fraction=0.30)
        assert det.vision_fusion_cutoff == 6

    def test_anchoring_start_fraction(self):
        det = CollapseDetector(n_layers=20, anchoring_tail_fraction=0.25)
        assert det.anchoring_start == 15

    def test_detect_returns_collapse_profile(self):
        masses = [0.1] * 6 + [0.5] * 8 + [0.6] * 6
        scan = _make_scan(masses)
        det = CollapseDetector(n_layers=20, vision_fusion_fraction=0.30, collapse_threshold=0.40)
        profile = det.detect(scan)
        assert isinstance(profile, CollapseProfile)

    def test_collapse_onset_detected_correctly(self):
        masses = [0.1] * 6 + [0.45] + [0.6] * 13
        scan = _make_scan(masses)
        det = CollapseDetector(n_layers=20, vision_fusion_fraction=0.30, collapse_threshold=0.40)
        profile = det.detect(scan)
        assert profile.collapse_onset_layer == 6

    def test_no_collapse_when_mass_always_low(self):
        masses = [0.1] * 20
        scan = _make_scan(masses)
        det = CollapseDetector(n_layers=20, collapse_threshold=0.50)
        profile = det.detect(scan)
        assert profile.collapse_onset_layer is None

    def test_peak_collapse_is_max_mass_outside_fusion(self):
        masses = [0.1] * 6 + [0.3, 0.9, 0.4, 0.5, 0.2] + [0.3] * 9
        scan = _make_scan(masses)
        det = CollapseDetector(n_layers=20, vision_fusion_fraction=0.30)
        profile = det.detect(scan)
        assert profile.peak_collapse_layer == 7

    def test_safe_enhancement_layers_not_in_fusion_zone(self):
        masses = [0.1] * 6 + [0.6] * 14
        scan = _make_scan(masses)
        det = CollapseDetector(n_layers=20, vision_fusion_fraction=0.30, collapse_threshold=0.40)
        profile = det.detect(scan)
        for lid in profile.safe_enhancement_layers:
            assert lid >= 6

    def test_safe_anchoring_layers_in_tail(self):
        masses = [0.1] * 20
        scan = _make_scan(masses)
        det = CollapseDetector(n_layers=20, anchoring_tail_fraction=0.25)
        profile = det.detect(scan)
        anchoring_start = int(20 * 0.75)
        for lid in profile.safe_anchoring_layers:
            assert lid >= anchoring_start

    def test_detect_from_scans_averages_correctly(self):
        scan1 = _make_scan([0.1] * 5 + [0.8] * 15)
        scan2 = _make_scan([0.1] * 5 + [0.2] * 15)
        det = CollapseDetector(n_layers=20, collapse_threshold=0.40)
        profile = det.detect_from_scans([scan1, scan2])
        assert profile.collapse_onset_layer is not None

    def test_detect_from_scans_empty_raises(self):
        det = CollapseDetector(n_layers=10)
        with pytest.raises(ValueError):
            det.detect_from_scans([])

    def test_is_vision_fusion_layer(self):
        masses = [0.1] * 10
        scan = _make_scan(masses)
        det = CollapseDetector(n_layers=10, vision_fusion_fraction=0.30)
        profile = det.detect(scan)
        assert profile.is_vision_fusion_layer(0)
        assert profile.is_vision_fusion_layer(2)
        assert not profile.is_vision_fusion_layer(5)

    def test_summary_contains_key_info(self):
        masses = [0.1] * 5 + [0.6] * 15
        scan = _make_scan(masses)
        det = CollapseDetector(n_layers=20)
        profile = det.detect(scan)
        summary = profile.summary()
        assert "Collapse Profile" in summary
        assert "Vision fusion cutoff" in summary
        assert "Enhancement layers" in summary
