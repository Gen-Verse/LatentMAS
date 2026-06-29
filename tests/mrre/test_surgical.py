"""Tests for SurgicalMRRE — Phase 2 mechanistic intervention."""

import pytest
pytest.importorskip("torch", reason="torch not installed")
import torch

from mrre_drift.mrre.surgical import SurgicalMRRE, SurgicalMRREConfig
from mrre_drift.interpret.collapse import CollapseDetector, CollapseProfile
from mrre_drift.interpret.logit_lens import LayerLensResult, LogitLensScan
from mrre_drift.models.layers import get_transformer_layers
from mrre_drift.utils.capture import HiddenStateCapture

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


def _make_collapse(model, n_layers, onset_layer=4):
    fusion_cutoff = int(n_layers * 0.30)
    anchoring_start = int(n_layers * 0.75)
    return CollapseProfile(
        n_layers=n_layers,
        vision_fusion_cutoff=fusion_cutoff,
        collapse_onset_layer=onset_layer,
        peak_collapse_layer=onset_layer + 1,
        safe_enhancement_layers=list(range(max(onset_layer, fusion_cutoff), anchoring_start)),
        safe_anchoring_layers=list(range(anchoring_start, n_layers)),
    )


def _default_config():
    return SurgicalMRREConfig(
        vision_fusion_fraction=0.0,
        enhancement_fractions=[0.4, 0.6],
        anchoring_fractions=[0.75, 0.875],
    )


class TestLayerSelection:
    def test_no_fusion_layers_when_fraction_zero(self, model, tokenizer):
        surgical = SurgicalMRRE(model, tokenizer, config=_default_config())
        n_layers = len(get_transformer_layers(model))
        for lid in surgical.enhancement_layer_ids:
            assert 0 <= lid < n_layers

    def test_fusion_layers_excluded(self, model, tokenizer):
        n_layers = len(get_transformer_layers(model))
        config = SurgicalMRREConfig(
            vision_fusion_fraction=0.5,
            enhancement_fractions=[0.2, 0.4, 0.6],
            anchoring_fractions=[0.75, 0.875],
        )
        surgical = SurgicalMRRE(model, tokenizer, config=config)
        fusion_cutoff = int(n_layers * 0.5)
        for lid in surgical.enhancement_layer_ids:
            assert lid >= fusion_cutoff

    def test_collapse_profile_overrides_fractions(self, model, tokenizer):
        n_layers = len(get_transformer_layers(model))
        collapse = _make_collapse(model, n_layers, onset_layer=3)
        surgical = SurgicalMRRE(model, tokenizer, collapse=collapse, config=_default_config())
        assert surgical.enhancement_layer_ids == collapse.safe_enhancement_layers
        assert surgical.anchoring_layer_ids == collapse.safe_anchoring_layers

    def test_enhancement_and_anchoring_disjoint_by_default(self, model, tokenizer):
        surgical = SurgicalMRRE(model, tokenizer, config=_default_config())
        enh = set(surgical.enhancement_layer_ids)
        anch = set(surgical.anchoring_layer_ids)
        assert enh.isdisjoint(anch)


class TestSurgicalMRREAPI:
    def test_repr_before_fit(self, model, tokenizer):
        surgical = SurgicalMRRE(model, tokenizer, config=_default_config())
        assert "fitted=False" in repr(surgical)

    def test_repr_after_fit(self, model, tokenizer, prompt_pairs, forcing_pairs):
        surgical = SurgicalMRRE(model, tokenizer, config=_default_config())
        surgical.fit(prompt_pairs, forcing_pairs)
        assert "fitted=True" in repr(surgical)

    def test_apply_raises_without_fit(self, model, tokenizer):
        surgical = SurgicalMRRE(model, tokenizer, config=_default_config())
        with pytest.raises(RuntimeError, match="fit"):
            with surgical.apply():
                pass

    def test_fit_empty_prompt_pairs_raises(self, model, tokenizer, forcing_pairs):
        surgical = SurgicalMRRE(model, tokenizer, config=_default_config())
        with pytest.raises(ValueError):
            surgical.fit([], forcing_pairs)

    def test_fit_empty_forcing_pairs_raises(self, model, tokenizer, prompt_pairs):
        surgical = SurgicalMRRE(model, tokenizer, config=_default_config())
        with pytest.raises(ValueError):
            surgical.fit(prompt_pairs, [])

    def test_save_load_roundtrip(self, model, tokenizer, prompt_pairs, forcing_pairs, tmp_path):
        s1 = SurgicalMRRE(model, tokenizer, config=_default_config())
        s1.fit(prompt_pairs, forcing_pairs)
        s1.save(tmp_path / "surgical")
        s2 = SurgicalMRRE(model, tokenizer, config=_default_config())
        s2.load(tmp_path / "surgical")
        for lid in s1.enhancement_layer_ids:
            assert torch.allclose(
                s1._enhancement_vectors.vectors[lid],
                s2._enhancement_vectors.vectors[lid],
            )

    def test_save_raises_without_fit(self, model, tokenizer, tmp_path):
        surgical = SurgicalMRRE(model, tokenizer, config=_default_config())
        with pytest.raises(RuntimeError, match="fit"):
            surgical.save(tmp_path / "v")

    def test_enhancement_norms_empty_before_fit(self, model, tokenizer):
        surgical = SurgicalMRRE(model, tokenizer, config=_default_config())
        assert surgical.enhancement_norms() == {}

    def test_anchoring_norms_nonempty_after_fit(
        self, model, tokenizer, prompt_pairs, forcing_pairs
    ):
        surgical = SurgicalMRRE(model, tokenizer, config=_default_config())
        surgical.fit(prompt_pairs, forcing_pairs)
        norms = surgical.anchoring_norms()
        assert len(norms) > 0
        for lid, norm in norms.items():
            assert norm > 0


class TestDispatchRoutesSurgical:
    """Regression suite for the Stage-2 dispatch bug.

    Historically, Stage-2 anchoring defaulted to CLAS (uniform alpha, no ramp),
    making ``mrre_two_stage`` byte-identical to a single-stage uniform baseline.
    These tests assert that the surgical path is taken and the anchoring ramp
    is applied.
    """

    def test_anchoring_ramp_on_by_default(self, model, tokenizer):
        """SurgicalMRREConfig should enable the anchoring ramp by default."""
        cfg = SurgicalMRREConfig(
            vision_fusion_fraction=0.0,
            enhancement_fractions=[0.4, 0.6],
            anchoring_fractions=[0.75, 0.875],
        )
        assert cfg.anchoring_ramp is True

    def test_ramp_produces_increasing_alphas(self, model, tokenizer, prompt_pairs, forcing_pairs):
        """With anchoring_ramp=True, per-layer alphas must be monotonically increasing."""
        from mrre_drift.mrre.hooks import make_linear_alpha_ramp
        cfg = SurgicalMRREConfig(
            vision_fusion_fraction=0.0,
            enhancement_fractions=[0.4, 0.6],
            anchoring_fractions=[0.75, 0.875],
            anchoring_ramp=True,
            alpha_anchoring_min=0.4,
            alpha_anchoring_max=0.8,
        )
        surgical = SurgicalMRRE(model, tokenizer, config=cfg)
        surgical.fit(prompt_pairs, forcing_pairs)
        alpha_map = surgical._anchoring_alpha()
        assert isinstance(alpha_map, dict), "Ramp must return a per-layer dict"
        alphas = [alpha_map[lid] for lid in sorted(alpha_map)]
        assert alphas == sorted(alphas), "Anchoring alphas must be non-decreasing across layers"
        assert alphas[0] < alphas[-1], "First anchoring layer must have a lower alpha than the last"

    def test_ramp_off_returns_scalar(self, model, tokenizer, prompt_pairs, forcing_pairs):
        """With anchoring_ramp=False (CLAS-like ablation), _anchoring_alpha returns a scalar."""
        cfg = SurgicalMRREConfig(
            vision_fusion_fraction=0.0,
            enhancement_fractions=[0.4, 0.6],
            anchoring_fractions=[0.75, 0.875],
            anchoring_ramp=False,
            alpha_anchoring=0.6,
        )
        surgical = SurgicalMRRE(model, tokenizer, config=cfg)
        surgical.fit(prompt_pairs, forcing_pairs)
        alpha_val = surgical._anchoring_alpha()
        assert isinstance(alpha_val, float), "Ramp-off path must return a scalar alpha"
        assert alpha_val == 0.6

    def test_surgical_differs_from_uniform_anchor(
        self, model, tokenizer, prompt_pairs, forcing_pairs
    ):
        """Surgical (ramped) and uniform-anchor (CLAS-like) must apply different alphas."""
        layers = get_transformer_layers(model)
        n_layers = len(layers)

        surgical_cfg = SurgicalMRREConfig(
            vision_fusion_fraction=0.0,
            enhancement_fractions=[0.4],
            anchoring_fractions=[0.75, 0.875],
            anchoring_ramp=True,
            alpha_anchoring_min=0.2,
            alpha_anchoring_max=0.9,
        )
        clas_cfg = SurgicalMRREConfig(
            vision_fusion_fraction=0.0,
            enhancement_fractions=[0.4],
            anchoring_fractions=[0.75, 0.875],
            anchoring_ramp=False,
            alpha_anchoring=0.55,
        )

        surgical = SurgicalMRRE(model, tokenizer, config=surgical_cfg)
        surgical.fit(prompt_pairs, forcing_pairs)
        clas = SurgicalMRRE(model, tokenizer, config=clas_cfg)
        clas.fit(prompt_pairs, forcing_pairs)

        surg_alpha = surgical._anchoring_alpha()
        clas_alpha = clas._anchoring_alpha()
        assert surg_alpha != clas_alpha, (
            "Surgical ramped alpha must differ from CLAS uniform alpha to confirm "
            "the dispatch bug is fixed and the two paths are not byte-identical."
        )

    def test_anchoring_vectors_from_forcing_pairs(
        self, model, tokenizer, prompt_pairs, forcing_pairs
    ):
        """Stage-2 anchoring must be computed from forcing pairs (tgt - en direction)."""
        from mrre_drift.mrre.anchoring import compute_anchoring_vectors
        from mrre_drift.models.layers import layer_ids_from_fractions

        surgical = SurgicalMRRE(
            model, tokenizer,
            config=SurgicalMRREConfig(
                vision_fusion_fraction=0.0,
                enhancement_fractions=[0.4],
                anchoring_fractions=[0.75, 0.875],
            ),
        )
        surgical.fit(prompt_pairs, forcing_pairs)

        # Re-derive anchoring vectors independently and check they match the stored ones.
        ref_vecs = compute_anchoring_vectors(
            model, tokenizer, forcing_pairs, surgical.anchoring_layer_ids, device="cpu"
        )
        for lid in surgical.anchoring_layer_ids:
            assert torch.allclose(
                surgical._anchoring_vectors.vectors[lid],
                ref_vecs.vectors[lid],
                atol=1e-5,
            ), f"Anchoring vector at layer {lid} does not match forcing-pair derivation"


class TestSurgicalMRREHooks:
    def test_apply_modifies_hidden_states(
        self, model, tokenizer, prompt_pairs, forcing_pairs
    ):
        surgical = SurgicalMRRE(model, tokenizer, config=_default_config())
        surgical.fit(prompt_pairs, forcing_pairs)
        all_target_ids = surgical.enhancement_layer_ids + surgical.anchoring_layer_ids
        layers = get_transformer_layers(model)
        target_layers = [layers[i] for i in all_target_ids]
        inputs = tokenizer("test surgical injection", return_tensors="pt")
        with torch.no_grad():
            with HiddenStateCapture(target_layers) as baseline:
                model(**inputs)
            with surgical.apply():
                with HiddenStateCapture(target_layers) as injected:
                    model(**inputs)
        changed = any(
            not torch.allclose(baseline.states[pos], injected.states[pos])
            for pos in range(len(all_target_ids))
        )
        assert changed

    def test_all_hooks_removed_after_apply(
        self, model, tokenizer, prompt_pairs, forcing_pairs
    ):
        surgical = SurgicalMRRE(model, tokenizer, config=_default_config())
        surgical.fit(prompt_pairs, forcing_pairs)
        layers = get_transformer_layers(model)
        before = [len(layer._forward_hooks) for layer in layers]
        with surgical.apply():
            pass
        after = [len(layer._forward_hooks) for layer in layers]
        assert before == after

    def test_hooks_removed_on_exception(
        self, model, tokenizer, prompt_pairs, forcing_pairs
    ):
        surgical = SurgicalMRRE(model, tokenizer, config=_default_config())
        surgical.fit(prompt_pairs, forcing_pairs)
        layers = get_transformer_layers(model)
        before = [len(layer._forward_hooks) for layer in layers]
        try:
            with surgical.apply():
                raise RuntimeError("simulated failure")
        except RuntimeError:
            pass
        after = [len(layer._forward_hooks) for layer in layers]
        assert before == after

    def test_generate_does_not_crash(
        self, model, tokenizer, prompt_pairs, forcing_pairs
    ):
        surgical = SurgicalMRRE(model, tokenizer, config=_default_config())
        surgical.fit(prompt_pairs, forcing_pairs)
        inputs = tokenizer("ทีละขั้นตอน:", return_tensors="pt")
        with torch.no_grad():
            with surgical.apply():
                out = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
        assert out.shape[1] > inputs.input_ids.shape[1]
