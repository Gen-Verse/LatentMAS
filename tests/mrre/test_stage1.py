"""
Tests for MRRE Stage 1 — Cross-Lingual Reasoning Enhancement Vectors.

Test strategy:
  1. Mechanics  — hooks attach / detach cleanly, shapes are correct
  2. Vector math — enhancement vectors move hidden states in the right direction
  3. Public API  — CrossLingualEnhancer fit/apply/save/load contract
  4. Semantic    — injected states are closer (cosine sim) to English space
"""

import pytest
pytest.importorskip("torch", reason="torch not installed")
import torch
import torch.nn.functional as F

from mrre_drift.mrre.stage1 import CrossLingualEnhancer
from mrre_drift.mrre.vectors import EnhancementVectors, compute_enhancement_vectors
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



def mid_layer_ids(model):
    return layer_ids_from_fractions(model, [0.4, 0.6])


class TestHiddenStateCapture:
    def test_captures_correct_shape(self, model, tokenizer):
        layers = get_transformer_layers(model)
        inputs = tokenizer("hello world", return_tensors="pt")
        hidden_dim = model.config.hidden_size
        with HiddenStateCapture([layers[0]]) as cap:
            with torch.no_grad():
                model(**inputs)
        assert 0 in cap.states
        assert cap.states[0].shape == (1, hidden_dim)

    def test_multiple_layers_captured(self, model, tokenizer):
        layers = get_transformer_layers(model)
        inputs = tokenizer("test", return_tensors="pt")
        with HiddenStateCapture([layers[0], layers[-1]]) as cap:
            with torch.no_grad():
                model(**inputs)
        assert len(cap.states) == 2

    def test_hooks_removed_after_context(self, model):
        layers = get_transformer_layers(model)
        before = [len(layer._forward_hooks) for layer in layers]
        with HiddenStateCapture([layers[0], layers[1]]):
            pass
        after = [len(layer._forward_hooks) for layer in layers]
        assert before == after

    def test_hooks_removed_on_exception(self, model):
        layers = get_transformer_layers(model)
        before = [len(layer._forward_hooks) for layer in layers]
        try:
            with HiddenStateCapture([layers[0]]):
                raise RuntimeError("simulated error")
        except RuntimeError:
            pass
        after = [len(layer._forward_hooks) for layer in layers]
        assert before == after


class TestEnhancementVectors:
    def test_vector_shapes(self, model, tokenizer, prompt_pairs):
        layer_ids = mid_layer_ids(model)
        ev = compute_enhancement_vectors(model, tokenizer, prompt_pairs, layer_ids)
        hidden_dim = model.config.hidden_size
        assert ev.layer_ids == layer_ids
        for lid in layer_ids:
            assert lid in ev.vectors
            assert ev.vectors[lid].shape == (hidden_dim,)
            assert ev.vectors[lid].dtype == torch.float32

    def test_vectors_are_nonzero(self, model, tokenizer, prompt_pairs):
        layer_ids = mid_layer_ids(model)
        ev = compute_enhancement_vectors(model, tokenizer, prompt_pairs, layer_ids)
        for lid, vec in ev.vectors.items():
            assert vec.norm().item() > 0, f"Layer {lid}: enhancement vector is all zeros"

    def test_save_load_roundtrip(self, model, tokenizer, prompt_pairs, tmp_path):
        layer_ids = mid_layer_ids(model)
        ev = compute_enhancement_vectors(model, tokenizer, prompt_pairs, layer_ids)
        path = tmp_path / "vectors.pt"
        ev.save(path)
        ev2 = EnhancementVectors.load(path)
        assert ev2.layer_ids == ev.layer_ids
        for lid in layer_ids:
            assert torch.allclose(ev.vectors[lid], ev2.vectors[lid])

    def test_more_pairs_runs_without_error(self, model, tokenizer, prompt_pairs):
        layer_ids = mid_layer_ids(model)
        ev1 = compute_enhancement_vectors(model, tokenizer, prompt_pairs[:1], layer_ids)
        ev_full = compute_enhancement_vectors(model, tokenizer, prompt_pairs, layer_ids)
        assert isinstance(ev1.vectors[layer_ids[0]].norm().item(), float)
        assert isinstance(ev_full.vectors[layer_ids[0]].norm().item(), float)


class TestCrossLingualEnhancerAPI:
    def test_repr_before_fit(self, model, tokenizer):
        enhancer = CrossLingualEnhancer(model, tokenizer, mid_layer_ids(model))
        assert "fitted=False" in repr(enhancer)

    def test_repr_after_fit(self, model, tokenizer, prompt_pairs):
        enhancer = CrossLingualEnhancer(model, tokenizer, mid_layer_ids(model))
        enhancer.fit(prompt_pairs)
        assert "fitted=True" in repr(enhancer)

    def test_apply_raises_without_fit(self, model, tokenizer):
        enhancer = CrossLingualEnhancer(model, tokenizer, mid_layer_ids(model))
        with pytest.raises(RuntimeError, match="fit"):
            with enhancer.apply():
                pass

    def test_fit_empty_pairs_raises(self, model, tokenizer):
        enhancer = CrossLingualEnhancer(model, tokenizer, mid_layer_ids(model))
        with pytest.raises(ValueError):
            enhancer.fit([])

    def test_save_raises_without_fit(self, model, tokenizer, tmp_path):
        enhancer = CrossLingualEnhancer(model, tokenizer, mid_layer_ids(model))
        with pytest.raises(RuntimeError, match="fit"):
            enhancer.save(tmp_path / "v.pt")

    def test_load_roundtrip(self, model, tokenizer, prompt_pairs, tmp_path):
        layer_ids = mid_layer_ids(model)
        e1 = CrossLingualEnhancer(model, tokenizer, layer_ids)
        e1.fit(prompt_pairs)
        e1.save(tmp_path / "v.pt")
        e2 = CrossLingualEnhancer(model, tokenizer, layer_ids)
        e2.load(tmp_path / "v.pt")
        for lid in layer_ids:
            assert torch.allclose(e1.vectors.vectors[lid], e2.vectors.vectors[lid])

    def test_vector_norms_nonempty_after_fit(self, model, tokenizer, prompt_pairs):
        enhancer = CrossLingualEnhancer(model, tokenizer, mid_layer_ids(model))
        enhancer.fit(prompt_pairs)
        norms = enhancer.vector_norms()
        assert len(norms) == len(mid_layer_ids(model))
        for lid, norm_val in norms.items():
            assert norm_val > 0, f"Layer {lid}: norm is zero"


class TestInjectionHooks:
    def test_apply_modifies_hidden_states(self, model, tokenizer, prompt_pairs):
        layer_ids = mid_layer_ids(model)
        layers = get_transformer_layers(model)
        target_layers = [layers[i] for i in layer_ids]
        enhancer = CrossLingualEnhancer(model, tokenizer, layer_ids, alpha=1.0)
        enhancer.fit(prompt_pairs)
        inputs = tokenizer("test input text", return_tensors="pt")
        with torch.no_grad():
            with HiddenStateCapture(target_layers) as baseline:
                model(**inputs)
        with torch.no_grad():
            with enhancer.apply():
                with HiddenStateCapture(target_layers) as injected:
                    model(**inputs)
        for pos in range(len(layer_ids)):
            assert not torch.allclose(baseline.states[pos], injected.states[pos])

    def test_injection_delta_matches_vector(self, model, tokenizer, prompt_pairs):
        single_lid = [mid_layer_ids(model)[-1]]
        layers = get_transformer_layers(model)
        target_layers = [layers[single_lid[0]]]
        alpha = 0.5
        enhancer = CrossLingualEnhancer(model, tokenizer, single_lid, alpha=alpha)
        enhancer.fit(prompt_pairs)
        inputs = tokenizer("arithmetic test", return_tensors="pt")
        with torch.no_grad():
            with HiddenStateCapture(target_layers) as baseline:
                model(**inputs)
            with enhancer.apply():
                with HiddenStateCapture(target_layers) as injected:
                    model(**inputs)
        lid = single_lid[0]
        expected = alpha * enhancer.vectors.vectors[lid]
        actual = (injected.states[0] - baseline.states[0]).squeeze(0)
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_hooks_removed_after_apply_context(self, model, tokenizer, prompt_pairs):
        layer_ids = mid_layer_ids(model)
        layers = get_transformer_layers(model)
        before = [len(layer._forward_hooks) for layer in layers]
        enhancer = CrossLingualEnhancer(model, tokenizer, layer_ids)
        enhancer.fit(prompt_pairs)
        with enhancer.apply():
            pass
        after = [len(layer._forward_hooks) for layer in layers]
        assert before == after

    def test_hooks_removed_on_exception_in_apply(self, model, tokenizer, prompt_pairs):
        layer_ids = mid_layer_ids(model)
        layers = get_transformer_layers(model)
        before = [len(layer._forward_hooks) for layer in layers]
        enhancer = CrossLingualEnhancer(model, tokenizer, layer_ids)
        enhancer.fit(prompt_pairs)
        try:
            with enhancer.apply():
                raise RuntimeError("simulated")
        except RuntimeError:
            pass
        after = [len(layer._forward_hooks) for layer in layers]
        assert before == after

    def test_alpha_zero_is_identity(self, model, tokenizer, prompt_pairs):
        layer_ids = mid_layer_ids(model)
        layers = get_transformer_layers(model)
        target_layers = [layers[i] for i in layer_ids]
        enhancer = CrossLingualEnhancer(model, tokenizer, layer_ids, alpha=0.0)
        enhancer.fit(prompt_pairs)
        inputs = tokenizer("hello world", return_tensors="pt")
        with torch.no_grad():
            with HiddenStateCapture(target_layers) as baseline:
                model(**inputs)
            with enhancer.apply():
                with HiddenStateCapture(target_layers) as injected:
                    model(**inputs)
        for pos in range(len(layer_ids)):
            assert torch.allclose(baseline.states[pos], injected.states[pos], atol=1e-6)


class TestSemanticCorrectness:
    def test_injection_moves_states_toward_english_space(
        self, model, tokenizer, prompt_pairs
    ):
        layer_ids = mid_layer_ids(model)
        layers = get_transformer_layers(model)
        target_layers = [layers[i] for i in layer_ids]
        enhancer = CrossLingualEnhancer(model, tokenizer, layer_ids, alpha=1.0)
        enhancer.fit(prompt_pairs)
        en_text = "Calculate step by step: what is the square root of 144?"
        th_text = "คำนวณทีละขั้นตอน: รากที่สองของ 144 คือเท่าไร?"
        en_inputs = tokenizer(en_text, return_tensors="pt")
        th_inputs = tokenizer(th_text, return_tensors="pt")
        with torch.no_grad():
            with HiddenStateCapture(target_layers) as en_cap:
                model(**en_inputs)
            with HiddenStateCapture(target_layers) as th_cap:
                model(**th_inputs)
            with enhancer.apply():
                with HiddenStateCapture(target_layers) as th_enh_cap:
                    model(**th_inputs)
        improvements = []
        for pos, lid in enumerate(layer_ids):
            en_h = en_cap.states[pos].squeeze(0)
            th_h = th_cap.states[pos].squeeze(0)
            th_enh_h = th_enh_cap.states[pos].squeeze(0)
            sim_before = F.cosine_similarity(en_h.unsqueeze(0), th_h.unsqueeze(0)).item()
            sim_after = F.cosine_similarity(en_h.unsqueeze(0), th_enh_h.unsqueeze(0)).item()
            improvements.append(sim_after - sim_before)
        assert any(delta > 0 for delta in improvements)

    def test_injection_on_english_produces_valid_output(
        self, model, tokenizer, prompt_pairs
    ):
        layer_ids = mid_layer_ids(model)
        enhancer = CrossLingualEnhancer(model, tokenizer, layer_ids, alpha=1.0)
        enhancer.fit(prompt_pairs)
        inputs = tokenizer("The capital of Germany is", return_tensors="pt")
        gen_kwargs = dict(max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        with torch.no_grad():
            baseline_out = model.generate(**inputs, **gen_kwargs)
            with enhancer.apply():
                enhanced_out = model.generate(**inputs, **gen_kwargs)
        assert baseline_out.shape[1] > inputs.input_ids.shape[1]
        assert enhanced_out.shape[1] > inputs.input_ids.shape[1]
