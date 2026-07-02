"""Tests for the (model x baseline x benchmark x language x metric) validator."""

from shared.combinations import (
    BASELINES, BENCHMARKS, MODELS, METRICS,
    enumerate_valid_combinations, validate_combination,
)

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"


def test_encoder_only_model_rejected():
    ok, reason = validate_combination(
        "deepset/xlm-roberta-large-squad2", "LatentMASBaseline", "mgsm", "en", "exact_match",
    )
    assert not ok
    assert "causal-LM" in reason


def test_valid_combination_accepted():
    ok, _ = validate_combination(
        "Qwen/Qwen2.5-7B-Instruct", "LatentMASBaseline", "mgsm", "th", "exact_match",
    )
    assert ok


def test_unsupported_language_rejected():
    # MGSM (juletxara/mgsm) has no Khmer data upstream.
    ok, reason = validate_combination(
        "Qwen/Qwen2.5-7B-Instruct", "LatentMASBaseline", "mgsm", "km", "exact_match",
    )
    assert not ok
    assert "km" in reason


def test_baseline_benchmark_mismatch_rejected():
    # single_agent_baseline is a coordination-pipeline comm mode; it doesn't run mgsm.
    ok, reason = validate_combination(
        "Qwen/Qwen2.5-7B-Instruct", "single_agent_baseline", "mgsm", "en", "exact_match",
    )
    assert not ok
    assert "does not support" in reason


def test_non_runnable_baseline_rejected():
    ok, reason = validate_combination(
        "Qwen/Qwen2.5-7B-Instruct", "CacheToCacheBaseline", "mgsm", "en", "exact_match",
    )
    assert not ok
    assert "no benchmark integration" in reason


def test_metric_task_type_mismatch_rejected():
    # safety_pass_rate only applies to SAFETY task type; mgsm is MATH_REASONING.
    ok, reason = validate_combination(
        "Qwen/Qwen2.5-7B-Instruct", "LatentMASBaseline", "mgsm", "en", "safety_pass_rate",
    )
    assert not ok
    assert "does not apply" in reason


def test_english_only_benchmark_rejects_language():
    ok, reason = validate_combination(
        "Qwen/Qwen2.5-7B-Instruct", "LatentMASBaseline", "gsm8k", "th", "exact_match",
    )
    # gsm8k isn't in any baseline's supported_benchmarks either, but if it were,
    # passing a language should still be rejected -- check via BENCHMARKS directly.
    assert not ok
    from shared.combinations import BenchmarkSpec
    assert BENCHMARKS["gsm8k"].languages is None


def test_all_recommended_models_and_metrics_now_wired():
    """As of this session, every model/metric previously flagged 'recommended' in
    dev_doc.md (Sailor2-8B-Chat, aya-expanse-8b, Llama-3.1-8B-Instruct, xcomet,
    cometkiwi) has been wired into configs/latent_coordination_heterogeneous.yaml and
    benchmark_runner.py respectively, and promoted to 'in_use'/'implemented'. If this
    fails, either something regressed or a genuinely new recommendation was added --
    update this test to match, don't just relax the assertion."""
    recommended_models = [m for m, spec in MODELS.items() if spec.status == "recommended"]
    assert recommended_models == [], f"still unwired: {recommended_models}"
    recommended_metrics = [m for m, spec in METRICS.items() if spec.status == "recommended"]
    assert recommended_metrics == [], f"still unwired: {recommended_metrics}"


def test_excluded_models_never_returned_regardless_of_include_recommended():
    # deepset/xlm-roberta-large-squad2 is encoder-only (causal_lm=False) -- must stay
    # excluded even when include_recommended=True, since that flag only affects
    # status filtering, not the hard causal_lm/runnable/language/task-type checks.
    combos = enumerate_valid_combinations(
        models=["deepset/xlm-roberta-large-squad2"], include_recommended=True,
    )
    assert combos == []


def test_every_benchmark_language_pair_actually_validates():
    """Every (benchmark, language) pair for a benchmark that's actually wired into at
    least one baseline's supported_benchmarks must pass validate_combination for some
    runnable model+baseline+metric combo. Benchmarks with real loaders but no baseline
    CLI wired to them yet (e.g. laobench, global_mmlu) are a known, separate gap --
    tracked explicitly in test_unwired_benchmarks_are_a_known_documented_gap below,
    not silently swept into this assertion."""
    wired_benchmarks = {
        bench_id for baseline in BASELINES.values()
        for bench_id in (baseline.supported_benchmarks or frozenset())
    }
    for bench_id, spec in BENCHMARKS.items():
        if spec.languages is None or bench_id not in wired_benchmarks:
            continue
        for lang in spec.languages:
            combos = enumerate_valid_combinations(benchmarks=[bench_id], languages=[lang])
            assert combos, f"no valid combination found for benchmark={bench_id} language={lang}"


def test_unwired_benchmarks_are_a_known_documented_gap():
    """Loaders that exist in data.py/dataset_loader.py but have no baseline CLI wired
    to them yet. This is a real, current gap (not a validator bug) -- if you wire one
    up (like mgsm_pro was this session), move it out of this set."""
    wired_benchmarks = {
        bench_id for baseline in BASELINES.values()
        for bench_id in (baseline.supported_benchmarks or frozenset())
    }
    unwired = {b for b in BENCHMARKS if b not in wired_benchmarks}
    expected_unwired = {
        "laobench", "mathmist", "sea_helm", "banglamath", "global_mmlu", "xquad", "mlqa",
        "gsm8k", "aime2024", "aime2025", "gpqa_diamond", "arc_easy", "arc_challenge",
        "winogrande", "mbppplus", "humanevalplus", "medqa", "seabench", "multichallenge",
        "multilingual_reasoning_gym", "sea_vl",
    }
    assert unwired == expected_unwired, (
        f"unwired-benchmark set changed: added={unwired - expected_unwired}, "
        f"removed={expected_unwired - unwired}. Update this test to match reality, and "
        "if something was newly wired, that's good news -- narrow expected_unwired."
    )


def test_registry_internal_consistency():
    # Every baseline's supported_benchmarks (if not None) must reference real benchmarks.
    for baseline in BASELINES.values():
        if baseline.supported_benchmarks:
            for bench_id in baseline.supported_benchmarks:
                assert bench_id in BENCHMARKS, f"{baseline.baseline_id} references unknown benchmark {bench_id}"
    # Every model marked causal_lm=False must be the documented exclusion, not silently wrong.
    excluded = [m for m in MODELS.values() if not m.causal_lm]
    assert all(m.status == "excluded" for m in excluded)
