"""Tests for Logit Lens hidden state mapping."""

import pytest
pytest.importorskip("torch", reason="torch not installed")
import torch
from mrre_drift.interpret.logit_lens import (
    LogitLens, LogitLensScan, LayerLensResult, get_lm_head_components,
)
from mrre_drift.models.layers import get_transformer_layers

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


class TestGetLMHeadComponents:
    def test_finds_lm_head(self, model):
        _, lm_head = get_lm_head_components(model)
        assert lm_head is not None

    def test_finds_final_norm(self, model):
        final_norm, _ = get_lm_head_components(model)
        assert final_norm is not None


class TestLogitLens:
    def test_scan_returns_correct_n_layers(self, model, tokenizer):
        ll = LogitLens(model, tokenizer, top_k=5)
        scan = ll.scan("hello world")
        n_layers = len(get_transformer_layers(model))
        assert len(scan.layer_results) == n_layers

    def test_layer_ids_sequential(self, model, tokenizer):
        ll = LogitLens(model, tokenizer)
        scan = ll.scan("test")
        ids = [r.layer_id for r in scan.layer_results]
        assert ids == list(range(len(ids)))

    def test_top_tokens_count(self, model, tokenizer):
        k = 5
        ll = LogitLens(model, tokenizer, top_k=k)
        scan = ll.scan("hello")
        for r in scan.layer_results:
            assert len(r.top_tokens) <= k

    def test_english_mass_in_range(self, model, tokenizer):
        ll = LogitLens(model, tokenizer)
        scan = ll.scan("hello world")
        for r in scan.layer_results:
            assert 0.0 <= r.english_mass <= 1.0

    def test_target_mass_zero_without_target_ids(self, model, tokenizer):
        ll = LogitLens(model, tokenizer)
        scan = ll.scan("test", target_token_ids=None)
        for r in scan.layer_results:
            assert r.target_mass == pytest.approx(0.0)

    def test_entropy_positive(self, model, tokenizer):
        ll = LogitLens(model, tokenizer)
        scan = ll.scan("hello world")
        for r in scan.layer_results:
            assert r.entropy > 0.0

    def test_collapse_onset_returns_int_or_none(self, model, tokenizer):
        ll = LogitLens(model, tokenizer)
        scan = ll.scan("test")
        onset = scan.collapse_onset_layer(threshold=0.5)
        assert onset is None or isinstance(onset, int)

    def test_peak_english_layer_is_valid(self, model, tokenizer):
        ll = LogitLens(model, tokenizer)
        scan = ll.scan("hello")
        n_layers = len(get_transformer_layers(model))
        assert 0 <= scan.peak_english_layer() < n_layers

    def test_mean_scan_averages_correctly(self, model, tokenizer, en_texts):
        ll = LogitLens(model, tokenizer)
        mean_scan = ll.mean_scan(en_texts)
        n_layers = len(get_transformer_layers(model))
        assert len(mean_scan.layer_results) == n_layers
        for r in mean_scan.layer_results:
            assert 0.0 <= r.english_mass <= 1.0

    def test_mean_scan_empty_raises(self, model, tokenizer):
        ll = LogitLens(model, tokenizer)
        with pytest.raises(ValueError):
            ll.mean_scan([])

    def test_scan_batch_length(self, model, tokenizer, en_texts):
        ll = LogitLens(model, tokenizer)
        scans = ll.scan_batch(en_texts)
        assert len(scans) == len(en_texts)

    def test_english_mass_by_layer_property(self, model, tokenizer):
        ll = LogitLens(model, tokenizer)
        scan = ll.scan("test")
        mapping = scan.english_mass_by_layer
        assert len(mapping) == len(scan.layer_results)
        for lid, mass in mapping.items():
            assert 0.0 <= mass <= 1.0
