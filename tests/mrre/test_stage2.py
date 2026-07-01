"""Tests for MRRE Stage 2 — Target-Language Output Anchoring Vectors."""

import pytest
pytest.importorskip("torch", reason="torch not installed")
import torch

from mrre_drift.mrre.stage2 import TargetLanguageAnchorer
from mrre_drift.mrre.anchoring import AnchoringVectors, compute_anchoring_vectors
from mrre_drift.mrre.stage1 import CrossLingualEnhancer
from mrre_drift.models.layers import get_transformer_layers, layer_ids_from_fractions
from mrre_drift.utils.capture import HiddenStateCapture

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"



def final_layer_ids(model):
    return layer_ids_from_fractions(model, [0.75, 0.875])


def mid_layer_ids(model):
    return layer_ids_from_fractions(model, [0.4, 0.6])


class TestAnchoringVectors:
    def test_vector_shapes(self, model, tokenizer, forcing_pairs):
        layer_ids = final_layer_ids(model)
        av = compute_anchoring_vectors(model, tokenizer, forcing_pairs, layer_ids)
        hidden_dim = model.config.hidden_size
        assert av.layer_ids == layer_ids
        for lid in layer_ids:
            assert lid in av.vectors
            assert av.vectors[lid].shape == (hidden_dim,)
            assert av.vectors[lid].dtype == torch.float32

    def test_vectors_are_nonzero(self, model, tokenizer, forcing_pairs):
        layer_ids = final_layer_ids(model)
        av = compute_anchoring_vectors(model, tokenizer, forcing_pairs, layer_ids)
        for lid, vec in av.vectors.items():
            assert vec.norm().item() > 0

    def test_save_load_roundtrip(self, model, tokenizer, forcing_pairs, tmp_path):
        layer_ids = final_layer_ids(model)
        av = compute_anchoring_vectors(model, tokenizer, forcing_pairs, layer_ids)
        path = tmp_path / "anchoring.pt"
        av.save(path)
        av2 = AnchoringVectors.load(path)
        assert av2.layer_ids == av.layer_ids
        for lid in layer_ids:
            assert torch.allclose(av.vectors[lid], av2.vectors[lid])


class TestTargetLanguageAnchorerAPI:
    def test_repr_before_fit(self, model, tokenizer):
        anchorer = TargetLanguageAnchorer(model, tokenizer, final_layer_ids(model))
        assert "fitted=False" in repr(anchorer)

    def test_repr_after_fit(self, model, tokenizer, forcing_pairs):
        anchorer = TargetLanguageAnchorer(model, tokenizer, final_layer_ids(model))
        anchorer.fit(forcing_pairs)
        assert "fitted=True" in repr(anchorer)

    def test_apply_raises_without_fit(self, model, tokenizer):
        anchorer = TargetLanguageAnchorer(model, tokenizer, final_layer_ids(model))
        with pytest.raises(RuntimeError, match="fit"):
            with anchorer.apply():
                pass

    def test_fit_empty_pairs_raises(self, model, tokenizer):
        anchorer = TargetLanguageAnchorer(model, tokenizer, final_layer_ids(model))
        with pytest.raises(ValueError):
            anchorer.fit([])

    def test_save_raises_without_fit(self, model, tokenizer, tmp_path):
        anchorer = TargetLanguageAnchorer(model, tokenizer, final_layer_ids(model))
        with pytest.raises(RuntimeError, match="fit"):
            anchorer.save(tmp_path / "a.pt")

    def test_load_roundtrip(self, model, tokenizer, forcing_pairs, tmp_path):
        layer_ids = final_layer_ids(model)
        a1 = TargetLanguageAnchorer(model, tokenizer, layer_ids)
        a1.fit(forcing_pairs)
        a1.save(tmp_path / "a.pt")
        a2 = TargetLanguageAnchorer(model, tokenizer, layer_ids)
        a2.load(tmp_path / "a.pt")
        for lid in layer_ids:
            assert torch.allclose(a1.vectors.vectors[lid], a2.vectors.vectors[lid])

    def test_vector_norms_nonempty_after_fit(self, model, tokenizer, forcing_pairs):
        anchorer = TargetLanguageAnchorer(model, tokenizer, final_layer_ids(model))
        anchorer.fit(forcing_pairs)
        norms = anchorer.vector_norms()
        assert len(norms) == len(final_layer_ids(model))
        for lid, norm_val in norms.items():
            assert norm_val > 0

    def test_vector_norms_empty_before_fit(self, model, tokenizer):
        anchorer = TargetLanguageAnchorer(model, tokenizer, final_layer_ids(model))
        assert anchorer.vector_norms() == {}


class TestAnchoringInjectionHooks:
    def test_apply_modifies_hidden_states(self, model, tokenizer, forcing_pairs):
        layer_ids = final_layer_ids(model)
        layers = get_transformer_layers(model)
        target_layers = [layers[i] for i in layer_ids]
        anchorer = TargetLanguageAnchorer(model, tokenizer, layer_ids, alpha=1.0)
        anchorer.fit(forcing_pairs)
        inputs = tokenizer("test input text", return_tensors="pt")
        with torch.no_grad():
            with HiddenStateCapture(target_layers) as baseline:
                model(**inputs)
        with torch.no_grad():
            with anchorer.apply():
                with HiddenStateCapture(target_layers) as injected:
                    model(**inputs)
        for pos in range(len(layer_ids)):
            assert not torch.allclose(baseline.states[pos], injected.states[pos])

    def test_hooks_removed_after_apply_context(self, model, tokenizer, forcing_pairs):
        layer_ids = final_layer_ids(model)
        layers = get_transformer_layers(model)
        before = [len(layer._forward_hooks) for layer in layers]
        anchorer = TargetLanguageAnchorer(model, tokenizer, layer_ids)
        anchorer.fit(forcing_pairs)
        with anchorer.apply():
            pass
        after = [len(layer._forward_hooks) for layer in layers]
        assert before == after

    def test_hooks_removed_on_exception(self, model, tokenizer, forcing_pairs):
        layer_ids = final_layer_ids(model)
        layers = get_transformer_layers(model)
        before = [len(layer._forward_hooks) for layer in layers]
        anchorer = TargetLanguageAnchorer(model, tokenizer, layer_ids)
        anchorer.fit(forcing_pairs)
        try:
            with anchorer.apply():
                raise RuntimeError("simulated")
        except RuntimeError:
            pass
        after = [len(layer._forward_hooks) for layer in layers]
        assert before == after

    def test_alpha_zero_is_identity(self, model, tokenizer, forcing_pairs):
        layer_ids = final_layer_ids(model)
        layers = get_transformer_layers(model)
        target_layers = [layers[i] for i in layer_ids]
        anchorer = TargetLanguageAnchorer(model, tokenizer, layer_ids, alpha=0.0)
        anchorer.fit(forcing_pairs)
        inputs = tokenizer("hello world", return_tensors="pt")
        with torch.no_grad():
            with HiddenStateCapture(target_layers) as baseline:
                model(**inputs)
            with anchorer.apply():
                with HiddenStateCapture(target_layers) as injected:
                    model(**inputs)
        for pos in range(len(layer_ids)):
            assert torch.allclose(baseline.states[pos], injected.states[pos], atol=1e-6)


class TestStage1Stage2Composition:
    def test_combined_stages_do_not_crash(
        self, model, tokenizer, prompt_pairs, forcing_pairs
    ):
        enhancer = CrossLingualEnhancer(model, tokenizer, mid_layer_ids(model), alpha=1.0)
        enhancer.fit(prompt_pairs)
        anchorer = TargetLanguageAnchorer(model, tokenizer, final_layer_ids(model), alpha=1.0)
        anchorer.fit(forcing_pairs)
        inputs = tokenizer("ทีละขั้นตอน: 144 รากที่สองคือเท่าไร?", return_tensors="pt")
        gen_kwargs = dict(max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        with torch.no_grad():
            with enhancer.apply():
                with anchorer.apply():
                    out = model.generate(**inputs, **gen_kwargs)
        assert out.shape[1] > inputs.input_ids.shape[1]

    def test_combined_stages_use_disjoint_layers(self, model):
        s1_ids = set(mid_layer_ids(model))
        s2_ids = set(final_layer_ids(model))
        assert s1_ids.isdisjoint(s2_ids)

    def test_combined_hooks_all_removed_after_context(
        self, model, tokenizer, prompt_pairs, forcing_pairs
    ):
        layers = get_transformer_layers(model)
        before = [len(layer._forward_hooks) for layer in layers]
        enhancer = CrossLingualEnhancer(model, tokenizer, mid_layer_ids(model))
        enhancer.fit(prompt_pairs)
        anchorer = TargetLanguageAnchorer(model, tokenizer, final_layer_ids(model))
        anchorer.fit(forcing_pairs)
        with enhancer.apply():
            with anchorer.apply():
                pass
        after = [len(layer._forward_hooks) for layer in layers]
        assert before == after
