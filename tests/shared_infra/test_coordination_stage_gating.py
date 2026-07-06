"""CoordinationPipeline.run(stages=...) actually gating which stages execute.

Regression test for a real bug: run() used to take a `stages` argument and ignore it
completely, always executing all 7 internal stages regardless of what --stages asked
for -- meaning a "just run the benchmark eval" invocation would also always pay for
CVAE training, adapter pretraining, etc. Uses monkeypatched stage methods (no real
models/GPUs) to prove only the requested stages actually run.
"""

import tempfile

import pytest

from latent_coordination.pipeline.coordination_pipeline import CoordinationPipeline

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"


def _make_pipeline(tmp_path, calls):
    pipeline = CoordinationPipeline({"project": {"output_dir": str(tmp_path)}})
    cm = pipeline.checkpoint_manager

    def stage_a():
        calls.append("A")
        result = ("router", "uls")
        cm.save(result, "stage_a")
        return result

    def stage_b():
        calls.append("B")
        cm.save("cvae", "stage_b")
        return "cvae"

    def stage_c(uls, router=None):
        calls.append("C")
        cm.save(True, "stage_c")

    def stage_d(router):
        calls.append("D")
        cm.save(True, "stage_d")

    def stage_e(router, uls, cvae_prior=None):
        calls.append("E")
        result = {"results_by_mode": {}}
        cm.save(result, "stage_f")
        return result

    pipeline._run_stage_a = stage_a
    pipeline._run_stage_b = stage_b
    pipeline._run_stage_c = stage_c
    pipeline._run_stage_d = stage_d
    pipeline._run_stage_e = stage_e
    pipeline._run_stage_f = lambda router, uls, report: calls.append("F")
    pipeline._run_stage_g = lambda report: (calls.append("G"), {"status": "completed"})[1]
    return pipeline


def test_default_runs_all_stages_in_order():
    calls = []
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _make_pipeline(tmp, calls)
        pipeline.run()
    assert calls == ["A", "B", "C", "D", "E", "F", "G"]


def test_requesting_only_e_without_checkpoints_raises():
    calls = []
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _make_pipeline(tmp, calls)
        with pytest.raises(RuntimeError, match="Stage A was not requested"):
            pipeline.run(stages=["E"])
    # Nothing should have run -- A is the first missing dependency.
    assert calls == []


def test_requesting_only_e_after_a_full_run_loads_checkpoints_not_recompute():
    calls = []
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _make_pipeline(tmp, calls)
        pipeline.run()  # populates checkpoints for A-E
        calls.clear()
        pipeline.run(stages=["E"])
    # Only E actually recomputes; A-D are loaded from checkpoint (not appended to calls).
    assert calls == ["E"]


def test_unknown_stage_letter_rejected():
    calls = []
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _make_pipeline(tmp, calls)
        with pytest.raises(ValueError, match="Unknown stage letter"):
            pipeline.run(stages=["Z"])


def test_g_not_requested_returns_raw_benchmark_report():
    calls = []
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _make_pipeline(tmp, calls)
        result = pipeline.run(stages=["A", "B", "C", "D", "E"])
    assert "G" not in calls
    assert result == {"results_by_mode": {}}
