"""Regression: mechanistic steered baselines must run with distinct configs.

The ``mrre_two_stage`` baseline was byte-identical to ``standard_clas`` because
``run_steered`` treated the baseline name as a label only, the sole differentiating
knob (subspace projection) was a no-op (steerer built with ``decomposer=None``), and
the Gaussian schedule was applied to every config equally. These tests pin the fix:
each steered baseline carries a distinct configuration, the projection path is
wired, and a guard warns if any two baselines still collapse to identical metrics.
"""

import itertools
import logging
import tempfile

import torch

from mechanistic_disentangle.eval.steering_benchmark import BenchmarkRunner
from mechanistic_disentangle.steering.gaussian_scheduler import GaussianDepthScheduler
from mechanistic_disentangle.steering.latent_steerer import LatentSteerer, _SteeringHook
from mechanistic_disentangle.steering.magnitude_norm import MagnitudeNormalizer

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


# ---- config table is structurally distinct ---------------------------------

def test_steering_configs_pairwise_distinct():
    cfgs = BenchmarkRunner.STEERING_CONFIGS
    for a, b in itertools.combinations(cfgs, 2):
        assert cfgs[a] != cfgs[b], f"{a} and {b} share an identical steering config"


def test_mrre_two_stage_differs_from_standard_clas():
    cfgs = BenchmarkRunner.STEERING_CONFIGS
    # the exact knobs that the bug collapsed
    assert cfgs["mrre_two_stage"]["subspace_proj"] is True
    assert cfgs["standard_clas"]["subspace_proj"] is False
    assert cfgs["mrre_two_stage"]["use_schedule"] != cfgs["standard_clas"]["use_schedule"]


# ---- dispatch threads a distinct config to each baseline --------------------

def test_run_suite_dispatches_distinct_configs(monkeypatch):
    runner = BenchmarkRunner(output_dir=tempfile.mkdtemp())
    captured = {}

    def fake_run_steered(model, tokenizer, samples, steerer, config_name,
                         steering_vectors, layer_ids, eta,
                         apply_subspace_projection=False, use_schedule=True):
        captured[config_name] = (round(eta, 6), apply_subspace_projection, use_schedule)
        # Return metrics that depend on the config so distinct configs -> distinct rows.
        return {"accuracy": eta + (0.1 if apply_subspace_projection else 0.0)
                + (0.01 if use_schedule else 0.0)}

    def fake_run_baseline(model, tokenizer, samples, config_name):
        return {"accuracy": 0.0}

    monkeypatch.setattr(runner, "run_steered", fake_run_steered)
    monkeypatch.setattr(runner, "run_baseline", fake_run_baseline)

    report = runner.run_suite(
        model=object(), tokenizer=object(), samples=[object()],
        steerer=object(), steering_vectors={}, layer_ids=[15, 24],
        eta=0.5, apply_subspace_proj=True,
    )

    # standard_clas: flat weights, no projection.
    assert captured["standard_clas"] == (0.5, False, False)
    # gaussian_scheduled: scheduled, no projection.
    assert captured["gaussian_scheduled"] == (0.5, False, True)
    # mrre_two_stage: scheduled + projection (the real two-stage method).
    assert captured["mrre_two_stage"] == (0.5, True, True)
    # aggressive_oversteering: scheduled, 3x eta.
    assert captured["aggressive_oversteering"] == (1.5, False, True)

    # No two steered baselines collapse to the same metrics.
    rows = [tuple(sorted(report.results[n].items()))
            for n in report.results if n != "no_intervention"]
    assert len(rows) == len(set(rows))


def test_apply_subspace_proj_false_disables_projection(monkeypatch):
    """When the suite disables projection, mrre_two_stage must not request it."""
    runner = BenchmarkRunner(output_dir=tempfile.mkdtemp())
    captured = {}

    def fake_run_steered(model, tokenizer, samples, steerer, config_name,
                         steering_vectors, layer_ids, eta,
                         apply_subspace_projection=False, use_schedule=True):
        captured[config_name] = apply_subspace_projection
        return {"accuracy": 1.0 if use_schedule else 0.0}

    monkeypatch.setattr(runner, "run_steered", fake_run_steered)
    monkeypatch.setattr(runner, "run_baseline",
                        lambda *a, **k: {"accuracy": 0.0})
    runner.run_suite(model=object(), tokenizer=object(), samples=[object()],
                     steerer=object(), steering_vectors={}, layer_ids=[15],
                     eta=0.5, apply_subspace_proj=False,
                     baselines=["mrre_two_stage", "gaussian_scheduled"])
    assert captured["mrre_two_stage"] is False  # suite veto respected


# ---- the identical-baselines guard ------------------------------------------

def test_guard_warns_on_identical_baselines(caplog):
    results = {
        "no_intervention": {"accuracy": 0.1},
        "standard_clas": {"accuracy": 0.30, "ifl_rate": 0.5},
        "mrre_two_stage": {"accuracy": 0.30, "ifl_rate": 0.5},  # identical -> bug
    }
    with caplog.at_level(logging.WARNING):
        BenchmarkRunner._warn_on_identical_baselines(results)
    assert any("IDENTICAL" in r.message for r in caplog.records)


def test_guard_silent_when_distinct(caplog):
    results = {
        "standard_clas": {"accuracy": 0.30},
        "mrre_two_stage": {"accuracy": 0.42},
    }
    with caplog.at_level(logging.WARNING):
        BenchmarkRunner._warn_on_identical_baselines(results)
    assert not any("IDENTICAL" in r.message for r in caplog.records)


# ---- steerer: schedule + per-layer decomposer -------------------------------

def _steerer():
    sched = GaussianDepthScheduler(alpha_0=1.2, mu_s=18.0, sigma_s=5.0, n_layers=32)
    return LatentSteerer(sched, MagnitudeNormalizer())


def test_decomposer_for_handles_map_and_single():
    s = _steerer()
    s.decomposer = {15: "dec15", 24: "dec24"}
    assert s._decomposer_for(15) == "dec15"
    assert s._decomposer_for(99) is None
    s.decomposer = "single"
    assert s._decomposer_for(15) == "single"
    assert s._decomposer_for(99) == "single"


# ---- Stage F must inject at the layers where vectors actually exist ----------

def test_steering_vector_layers_unions_per_model_layers():
    """Stage F derives injection layers from the vectors, not a hardcoded default.

    The bug: on --resume Stage B's early return skipped setting self._layer_ids, so
    Stage F fell back to [15,24]. For Gemma (vectors at [21,31]) / SeaLLMs ([14,21])
    that matched nothing -> "0 layers active" -> steering silently no-op'd.
    """
    from mechanistic_disentangle.pipeline.mechanistic_pipeline import MechanisticPipeline

    class _V:
        def __init__(self, layers):
            self.vectors = {l: torch.ones(4) for l in layers}

    # Gemma-like: vectors live at 21 and 31, not the old [15,24] default.
    svd = {
        "th": {"mean_diff": _V([21, 31]), "subspace_projected": {}},
        "my": {"mean_diff": _V([21, 31])},
    }
    assert MechanisticPipeline._steering_vector_layers(svd) == [21, 31]
    # No vectors -> empty (caller falls back to depth-relative layers).
    assert MechanisticPipeline._steering_vector_layers({}) == []
    assert MechanisticPipeline._steering_vector_layers(
        {"th": {"subspace_projected": {}}}
    ) == []


def test_hook_invokes_decomposer_projection():
    """A wired decomposer must actually run project_to_reasoning (was a no-op)."""
    class _Dec:
        called = False
        def project_to_reasoning(self, h):
            type(self).called = True
            return h

    hook = _SteeringHook(
        layer_id=15,
        steering_vector=torch.ones(8),
        weight=1.0,
        eta=0.5,
        normalizer=MagnitudeNormalizer(),
        decomposer=_Dec(),
        log_injections=False,
    )
    out = hook(module=None, inputs=(), output=torch.ones(1, 4, 8))
    assert _Dec.called
    assert out.shape == (1, 4, 8)
