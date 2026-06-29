"""Tests for Concept Recursive Activation Factorization (CRAF)."""

import pytest
pytest.importorskip("torch", reason="torch not installed")
import torch
from mrre_drift.interpret.craf import CRAF, CRAFProfile, LayerCRAFResult, ConceptDirection
from mrre_drift.models.layers import get_transformer_layers

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


class TestCRAF:
    def test_profile_returns_correct_n_layers(self, model, tokenizer, en_texts, tgt_texts):
        craf = CRAF(model, tokenizer, n_components=2)
        profile = craf.profile(en_texts, tgt_texts)
        n_layers = len(get_transformer_layers(model))
        assert len(profile.layer_results) == n_layers

    def test_layer_result_has_concept_directions(self, model, tokenizer, en_texts, tgt_texts):
        craf = CRAF(model, tokenizer, n_components=2)
        profile = craf.profile(en_texts, tgt_texts)
        for lid, result in profile.layer_results.items():
            assert len(result.concept_directions) >= 1

    def test_concept_directions_are_unit_vectors(self, model, tokenizer, en_texts, tgt_texts):
        craf = CRAF(model, tokenizer, n_components=1)
        profile = craf.profile(en_texts, tgt_texts)
        for lid, result in profile.layer_results.items():
            for cd in result.concept_directions:
                norm = cd.direction.norm().item()
                assert abs(norm - 1.0) < 1e-4

    def test_explained_variance_ratio_in_range(self, model, tokenizer, en_texts, tgt_texts):
        craf = CRAF(model, tokenizer, n_components=2)
        profile = craf.profile(en_texts, tgt_texts)
        for lid, result in profile.layer_results.items():
            for cd in result.concept_directions:
                assert 0.0 <= cd.explained_variance_ratio <= 1.0

    def test_alignment_values_bounded(self, model, tokenizer, en_texts, tgt_texts):
        craf = CRAF(model, tokenizer, n_components=1)
        profile = craf.profile(en_texts, tgt_texts)
        for lid, result in profile.layer_results.items():
            assert -1.0 <= result.english_alignment <= 1.0
            assert -1.0 <= result.target_alignment <= 1.0

    def test_alignment_delta_property(self, model, tokenizer, en_texts, tgt_texts):
        craf = CRAF(model, tokenizer, n_components=1)
        profile = craf.profile(en_texts, tgt_texts)
        for lid, result in profile.layer_results.items():
            expected = result.english_alignment - result.target_alignment
            assert abs(result.alignment_delta - expected) < 1e-6

    def test_empty_en_texts_raises(self, model, tokenizer, tgt_texts):
        craf = CRAF(model, tokenizer)
        with pytest.raises(ValueError):
            craf.profile([], tgt_texts)

    def test_empty_tgt_texts_raises(self, model, tokenizer, en_texts):
        craf = CRAF(model, tokenizer)
        with pytest.raises(ValueError):
            craf.profile(en_texts, [])

    def test_peak_drift_layer_is_valid(self, model, tokenizer, en_texts, tgt_texts):
        craf = CRAF(model, tokenizer, n_components=1)
        profile = craf.profile(en_texts, tgt_texts)
        n_layers = len(get_transformer_layers(model))
        assert 0 <= profile.peak_drift_layer() < n_layers

    def test_drift_onset_is_int_or_none(self, model, tokenizer, en_texts, tgt_texts):
        craf = CRAF(model, tokenizer, n_components=1)
        profile = craf.profile(en_texts, tgt_texts)
        onset = profile.drift_onset_layer(threshold=0.05)
        assert onset is None or isinstance(onset, int)

    def test_safe_intervention_layers_outside_fusion_zone(
        self, model, tokenizer, en_texts, tgt_texts
    ):
        craf = CRAF(model, tokenizer, n_components=1)
        profile = craf.profile(en_texts, tgt_texts)
        n_layers = len(get_transformer_layers(model))
        fusion_cutoff = int(n_layers * 0.30)
        safe = profile.safe_intervention_layers(vision_fusion_fraction=0.30)
        for lid in safe:
            assert lid >= fusion_cutoff
