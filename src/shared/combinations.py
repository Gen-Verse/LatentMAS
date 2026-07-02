"""Registry + validator for (model x baseline x benchmark x language x metric) combinations.

Not every combination in the cartesian product is meaningful or runnable: a baseline's
CLI only wires up specific benchmarks, a benchmark only has data for specific languages,
and a metric only applies to specific task types. This module is the single source of
truth for those constraints so callers don't have to re-derive them by hand.

Every fact recorded here (language coverage, gating status, VRAM estimates, CLI wiring)
was verified against the actual HF dataset/model metadata or the actual code, not
assumed from documentation -- see the inline citations. When a new benchmark/model is
added to the project, register it here too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from typing import Dict, FrozenSet, List, Optional, Tuple

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


# ---------------------------------------------------------------------------
# Task types (gate which metrics apply to which benchmarks)
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    MATH_REASONING = "math_reasoning"
    READING_COMPREHENSION = "reading_comprehension"
    KNOWLEDGE_MCQA = "knowledge_mcqa"
    CODE = "code"
    COMMONSENSE = "commonsense"
    TRANSLATION = "translation"
    SAFETY = "safety"
    OPEN_GENERATION = "open_generation"


# A language axis of None means the benchmark has no per-language parameter at all
# (English-only or single fixed language).
ANY_LANGUAGE = None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    params_b: float
    architecture: str          # HF `model_type`, e.g. "qwen2", "llama", "cohere"
    causal_lm: bool
    gated: bool
    license: str
    status: str                # "in_use" | "recommended" (verified real, not yet wired into configs)
    notes: str = ""

    def fits_v100_16gb(self, quant: str = "8bit") -> bool:
        """Rough VRAM feasibility check for a single 16GB V100 (fp16-only, cc 7.0)."""
        bytes_per_param = {"fp16": 2.0, "8bit": 1.0, "4bit": 0.5}.get(quant)
        if bytes_per_param is None:
            raise ValueError(f"Unknown quant '{quant}'; expected fp16|8bit|4bit")
        # +~10-15% overhead for KV cache / activations at short context.
        return self.params_b * 1e9 * bytes_per_param * 1.15 / 1e9 < 16.0


MODELS: Dict[str, ModelSpec] = {
    # --- in active use (configs/*.yaml, dev_doc.md) ---
    "Qwen/Qwen2.5-7B-Instruct": ModelSpec(
        "Qwen/Qwen2.5-7B-Instruct", 7.6, "qwen2", True, False, "apache-2.0", "in_use",
    ),
    "aisingapore/Llama-SEA-LION-v3-8B-IT": ModelSpec(
        "aisingapore/Llama-SEA-LION-v3-8B-IT", 8.0, "llama", True, False, "llama3",
        "in_use", "Default agent model across configs/*.yaml.",
    ),
    "aisingapore/sea-lion-7b-instruct": ModelSpec(
        "aisingapore/sea-lion-7b-instruct", 7.0, "mpt", True, False, "apache-2.0", "in_use",
    ),
    "SeaLLMs/SeaLLMs-v3-7B-Chat": ModelSpec(
        "SeaLLMs/SeaLLMs-v3-7B-Chat", 7.6, "qwen2", True, False, "other", "in_use",
    ),
    "llava-hf/llava-1.5-7b-hf": ModelSpec(
        "llava-hf/llava-1.5-7b-hf", 7.0, "llava", True, False, "llama2",
        "in_use", "Multimodal; only meaningful for SEA-Vision's image-QA path.",
    ),
    # --- wired into configs/latent_coordination_heterogeneous.yaml (2026-07-02) ---
    "sail/Sailor2-8B-Chat": ModelSpec(
        "sail/Sailor2-8B-Chat", 8.5, "qwen2", True, False, "apache-2.0", "in_use",
        "Successor to SeaLLMs; covers 12 SEA languages incl. lo/my/km/jv/su/fil -- "
        "broader single-model SEA coverage than anything else configured. Translation "
        "agent in configs/latent_coordination_heterogeneous.yaml. Downloaded (17GB).",
    ),
    "CohereLabs/aya-expanse-8b": ModelSpec(
        "CohereLabs/aya-expanse-8b", 8.0, "cohere", True, True, "cc-by-nc-4.0", "in_use",
        "Only genuinely different architecture family (cohere, not qwen2/llama/gemma) "
        "of any model here -- needed for the heterogeneous cross-architecture ablation "
        "(strategy.md 7.2's 'LLaMA safety agent + Qwen reasoning agent' claim needs an "
        "actually-different arch pair to be true, not just a different HF repo). "
        "cc-by-nc-4.0: research use only, not commercial. Gated (auto-approved). Safety "
        "agent in configs/latent_coordination_heterogeneous.yaml. Downloaded (15GB).",
    ),
    "meta-llama/Llama-3.1-8B-Instruct": ModelSpec(
        "meta-llama/Llama-3.1-8B-Instruct", 8.0, "llama", True, True, "llama3.1", "in_use",
        "Standard llama-arch comparison point alongside SEA-LION. Reasoning agent in "
        "configs/latent_coordination_heterogeneous.yaml. Gated, access requested but not "
        "yet approved as of 2026-07-02 -- not downloaded yet; wired in code regardless.",
    ),
    # --- explicitly excluded ---
    "deepset/xlm-roberta-large-squad2": ModelSpec(
        "deepset/xlm-roberta-large-squad2", 0.56, "xlm-roberta", False, False, "cc-by-4.0",
        "excluded", "Encoder-only extractive-QA model; incompatible with every baseline/"
        "pipeline here (all require AutoModelForCausalLM).",
    ),
}


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaselineSpec:
    baseline_id: str
    kind: str                              # "comm_mode" | "standalone_class"
    runnable: bool                         # has an actual CLI/benchmark integration today
    supported_benchmarks: Optional[FrozenSet[str]]  # None = any benchmark the pipeline supports
    notes: str = ""


BASELINES: Dict[str, BaselineSpec] = {
    # Built into the main coordination pipeline (benchmark_runner.py); benchmark is
    # whatever the pipeline config points at (flores_plus / sea_vision / sea_safeguardbench).
    "single_agent_baseline": BaselineSpec(
        "single_agent_baseline", "comm_mode", True,
        frozenset({"flores_plus", "sea_vision", "sea_safeguardbench"}),
    ),
    "token_based_mas": BaselineSpec(
        "token_based_mas", "comm_mode", True,
        frozenset({"flores_plus", "sea_vision", "sea_safeguardbench"}),
    ),
    "latent_based_mas_ours": BaselineSpec(
        "latent_based_mas_ours", "comm_mode", True,
        frozenset({"flores_plus", "sea_vision", "sea_safeguardbench"}),
        "Requires HFBackend (hidden-state transfer) -- never VLLMBackend.",
    ),
    # Standalone baseline classes with a runnable CLI (run_latentmas.py / run_thoughtcomm.py).
    "LatentMASBaseline": BaselineSpec(
        "LatentMASBaseline", "standalone_class", True, frozenset({"mgsm", "mgsm_pro", "belebele"}),
    ),
    "ThoughtCommBaseline": BaselineSpec(
        "ThoughtCommBaseline", "standalone_class", True, frozenset({"mgsm", "mgsm_pro", "belebele"}),
    ),
    # Standalone classes that exist but have no benchmark-runner CLI yet.
    "CacheToCacheBaseline": BaselineSpec(
        "CacheToCacheBaseline", "standalone_class", False, frozenset(),
        "Class implemented; no run_*.py CLI wired to any benchmark yet.",
    ),
    "GDesignerBaseline": BaselineSpec(
        "GDesignerBaseline", "standalone_class", False, frozenset(),
        "Class implemented; no run_*.py CLI wired to any benchmark yet.",
    ),
    "MasRouterBaseline": BaselineSpec(
        "MasRouterBaseline", "standalone_class", False, frozenset(),
        "Class implemented; no run_*.py CLI wired to any benchmark yet.",
    ),
    "VisionWormholeBaseline": BaselineSpec(
        "VisionWormholeBaseline", "standalone_class", False, frozenset(),
        "Class implemented; no run_*.py CLI wired to any benchmark yet.",
    ),
    "BlackboardMASBaseline": BaselineSpec(
        "BlackboardMASBaseline", "standalone_class", False, frozenset(),
        "Class implemented; no run_*.py CLI wired to any benchmark yet.",
    ),
}


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    task_type: TaskType
    languages: Optional[FrozenSet[str]]  # None = no language axis (single/English-only)
    loader: str                          # "module.function" for the real loader
    gated: bool = False
    status: str = "verified"             # "verified" | "verified_gated_untested"
    notes: str = ""


BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "mgsm": BenchmarkSpec(
        "mgsm", TaskType.MATH_REASONING,
        frozenset({"bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"}),
        "data.py::load_mgsm / eval.correctness::load_mgsm_tasks",
        notes="juletxara/mgsm. No lo/km/my/am upstream -- confirmed via "
        "datasets.get_dataset_config_names, not fixable in code.",
    ),
    "mgsm_pro": BenchmarkSpec(
        "mgsm_pro", TaskType.MATH_REASONING,
        frozenset({"am", "en", "fr", "ig", "ja", "sw", "tw", "yo", "zh"}),
        "data.py::load_mgsm_pro",
        notes="McGill-NLP/mgsm-pro. Languages are HF *splits*, not a config param -- "
        "loader was broken (passed lang as config) until fixed this session. "
        "Notably includes Amharic, which base MGSM lacks.",
    ),
    "belebele": BenchmarkSpec(
        "belebele", TaskType.READING_COMPREHENSION,
        frozenset({"th", "my", "km", "lo", "jv", "su", "ceb", "vi", "id", "ms", "fil", "am", "sw"}),
        "data.py::load_belebele / dataset_loader.py::DatasetLoader.load_belebele",
        notes="facebook/belebele has ~122 languages total; only the ones this repo's "
        "BELEBELE_LANG_MAP maps are enumerated here.",
    ),
    "multilingual_reasoning_gym": BenchmarkSpec(
        "multilingual_reasoning_gym", TaskType.MATH_REASONING,
        frozenset({"de", "en", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt", "ru", "uk", "zh"}),
        "data.py::load_multilingual_reasoning_gym",
        notes="MauroPello/multilingual-reasoning-gym-sft. NO SEA languages at all -- "
        "cannot be used for th/lo/km/my/am/sw regardless of config.",
    ),
    "laobench": BenchmarkSpec(
        "laobench", TaskType.KNOWLEDGE_MCQA, frozenset({"lo"}),
        "data.py::load_laobench",
        notes="BAAI/LaoBench, apache-2.0, ungated. Added this session to close the "
        "Lao reasoning-benchmark gap (MGSM/MRG have none). MCQA subset only "
        "(K12 Foundational Education + Knowledge Application); excludes the "
        "Bilingual-Translation rows by default.",
    ),
    "mathmist": BenchmarkSpec(
        "mathmist", TaskType.MATH_REASONING, frozenset({"am", "sw"}),
        "data.py::load_mathmist",
        gated=True, status="verified",
        notes="mahbubhimel/MathMist. Robustness-appendix only per strategy.md, not a "
        "core benchmark. Access granted + schema verified 2026-07-02: single 'default' "
        "config, no 'test' split -- each language is its own split ('amharic', "
        "'swahili', ...), columns are capitalized ('Question'/'Exact Answer', not "
        "lowercase). Loader signature changed to load_mathmist(lang=...) accordingly. "
        "Does NOT cover th/lo/km/my despite the paper's 13-language claim; only am/sw "
        "of this project's target set.",
    ),
    "sea_helm": BenchmarkSpec(
        "sea_helm", TaskType.KNOWLEDGE_MCQA, frozenset({"th", "my", "km", "lo"}),
        "data.py::load_sea_helm",
        gated=True, status="verified",
        notes="Real components live at aisingapore/<ComponentName> (e.g. "
        "aisingapore/NLU-Belebele-MCQA, aisingapore/NLR-Causal-Reasoning) -- the "
        "loader's old 'aisingapore/sea-helm-{subset}' ID pattern 404s and was fixed. "
        "Access granted + schema verified 2026-07-02 for NLU-Belebele-MCQA: "
        "language-configured (load_dataset(repo, lang)), splits are 'eval'/'examples' "
        "not 'test', MCQA content nested under a single-element 'prompts' list "
        "(choice1..4/question/text) with a top-level letter 'label'. Loader signature "
        "changed to load_sea_helm(lang=..., subset=..., split='eval') accordingly. "
        "Other components' schemas are unverified -- don't assume this shape "
        "generalizes without checking.",
    ),
    "banglamath": BenchmarkSpec(
        "banglamath", TaskType.MATH_REASONING, frozenset({"bn"}),
        "data.py::load_banglamath",
        notes="Optional/narrow-scope per strategy.md -- Bengali only, not a target "
        "SEA language for this project.",
    ),
    "flores_plus": BenchmarkSpec(
        "flores_plus", TaskType.TRANSLATION, None,  # ~200 languages; treat as "any"
        "data.py::load_flores_plus; primary corpus for coordination_pipeline.py",
        notes="facebook/flores, ~200 languages. Pass any FLORES+ language code "
        "(e.g. tha_Thai, lao_Laoo, khm_Khmr, mya_Mymr, amh_Ethi, swh_Latn).",
    ),
    "sea_vision": BenchmarkSpec(
        "sea_vision", TaskType.OPEN_GENERATION,
        frozenset({"th", "my", "km", "lo", "vi", "id", "ms", "fil", "en", "zh"}),
        "dataset_loader.py::DatasetLoader.load_sea_vision",
    ),
    "sea_vl": BenchmarkSpec(
        "sea_vl", TaskType.OPEN_GENERATION,
        frozenset({"th", "my", "km", "lo", "vi", "id", "ms", "fil", "ta", "zh", "en", "jv", "su", "ceb"}),
        "dataset_loader.py::DatasetLoader.load_sea_vl",
    ),
    "sea_safeguardbench": BenchmarkSpec(
        "sea_safeguardbench", TaskType.SAFETY, None,  # depends on configured repo_id
        "dataset_loader.py::DatasetLoader.load_sea_safeguardbench",
        notes="Language set depends on whichever repo_id is configured -- no hardcoded set.",
    ),
    "global_mmlu": BenchmarkSpec(
        "global_mmlu", TaskType.KNOWLEDGE_MCQA,
        frozenset({"ar", "bn", "de", "es", "fr", "hi", "id", "it", "ja", "ko",
                   "pt", "ru", "sw", "te", "th", "yo", "zh"}),
        "dataset_loader.py::DatasetLoader.load_global_mmlu",
    ),
    "xquad": BenchmarkSpec(
        "xquad", TaskType.READING_COMPREHENSION,
        frozenset({"ar", "de", "el", "en", "es", "hi", "ro", "ru", "th", "tr", "vi", "zh"}),
        "data.py::load_xquad",
    ),
    "mlqa": BenchmarkSpec(
        "mlqa", TaskType.READING_COMPREHENSION,
        frozenset({"ar", "de", "en", "es", "hi", "vi", "zh"}),
        "data.py::load_mlqa", notes="No SEA target languages.",
    ),
    # English-only / single-language, no language axis.
    "gsm8k": BenchmarkSpec("gsm8k", TaskType.MATH_REASONING, None, "data.py::load_gsm8k"),
    "aime2024": BenchmarkSpec("aime2024", TaskType.MATH_REASONING, None, "data.py::load_aime2024"),
    "aime2025": BenchmarkSpec("aime2025", TaskType.MATH_REASONING, None, "data.py::load_aime2025"),
    "gpqa_diamond": BenchmarkSpec("gpqa_diamond", TaskType.KNOWLEDGE_MCQA, None, "data.py::load_gpqa_diamond"),
    "arc_easy": BenchmarkSpec("arc_easy", TaskType.COMMONSENSE, None, "data.py::load_arc_easy"),
    "arc_challenge": BenchmarkSpec("arc_challenge", TaskType.COMMONSENSE, None, "data.py::load_arc_challenge"),
    "winogrande": BenchmarkSpec("winogrande", TaskType.COMMONSENSE, None, "data.py::load_winogrande"),
    "mbppplus": BenchmarkSpec("mbppplus", TaskType.CODE, None, "data.py::load_mbppplus"),
    "humanevalplus": BenchmarkSpec("humanevalplus", TaskType.CODE, None, "data.py::load_humanevalplus"),
    "medqa": BenchmarkSpec("medqa", TaskType.KNOWLEDGE_MCQA, None, "data.py::load_medqa"),
    "seabench": BenchmarkSpec("seabench", TaskType.OPEN_GENERATION, None, "data.py::load_seabench"),
    "multichallenge": BenchmarkSpec("multichallenge", TaskType.OPEN_GENERATION, None, "data.py::load_multichallenge"),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    applicable_task_types: FrozenSet[TaskType]
    package: str
    status: str          # "implemented" | "recommended"
    notes: str = ""


METRICS: Dict[str, MetricSpec] = {
    "exact_match": MetricSpec(
        "exact_match",
        frozenset({TaskType.MATH_REASONING, TaskType.KNOWLEDGE_MCQA, TaskType.COMMONSENSE, TaskType.CODE}),
        "eval.correctness (in-repo)", "implemented",
    ),
    "bleu": MetricSpec(
        "bleu", frozenset({TaskType.TRANSLATION, TaskType.OPEN_GENERATION}),
        "sacrebleu", "implemented",
    ),
    "chrf": MetricSpec(
        "chrf", frozenset({TaskType.TRANSLATION, TaskType.OPEN_GENERATION}),
        "sacrebleu", "implemented",
    ),
    "sfr_ifl": MetricSpec(
        "sfr_ifl", frozenset({TaskType.TRANSLATION, TaskType.OPEN_GENERATION}),
        "eval.script_fidelity (in-repo)", "implemented",
        "Script Fidelity Rate / Involuntary Fidelity Loss. Unicode-range based -- blind "
        "for same-script language pairs (e.g. sw vs en, both Latin-script).",
    ),
    "language_consistency": MetricSpec(
        "language_consistency", frozenset({TaskType.TRANSLATION, TaskType.OPEN_GENERATION}),
        "langid", "implemented",
        "Added this session (eval.script_fidelity::LanguageConsistencyEvaluator) "
        "specifically to cover SFR's Latin-script blind spot. Unsupported for Burmese "
        "(langid has no 'my' class).",
    ),
    "cka_procrustes_rsa": MetricSpec(
        "cka_procrustes_rsa", frozenset({TaskType.TRANSLATION, TaskType.OPEN_GENERATION}),
        "geometry.isomorphism (in-repo)", "implemented",
        "Geometric isomorphism probes; mechanistic/geometry pipeline only, not a "
        "generic per-sample text metric.",
    ),
    "safety_pass_rate": MetricSpec(
        "safety_pass_rate", frozenset({TaskType.SAFETY}),
        "eval.benchmark_runner (in-repo)", "implemented",
    ),
    "efficiency_latency_tokens": MetricSpec(
        "efficiency_latency_tokens",
        frozenset(TaskType),  # applies regardless of task type
        "eval.efficiency_metrics / eval.cost (in-repo)", "implemented",
    ),
    "adversarial_drift": MetricSpec(
        "adversarial_drift", frozenset({TaskType.TRANSLATION, TaskType.OPEN_GENERATION}),
        "eval.adversarial (in-repo)", "implemented",
    ),
    "information_theoretic": MetricSpec(
        "information_theoretic", frozenset(TaskType),
        "eval.information_theory (in-repo)", "implemented",
    ),
    # --- wired into MultiAgentBenchmarkRunner._compute_translation_quality (flores_plus
    #     only; opt-in via configs/*.yaml benchmarks.flores_plus.translation_metrics,
    #     since both load multi-GB gated checkpoints) ---
    "xcomet": MetricSpec(
        "xcomet", frozenset({TaskType.TRANSLATION}),
        "unbabel-comet (checkpoint: Unbabel/XCOMET-XL)", "implemented",
        "Reference-based, best correlation with human judgment as of 2025/2026; "
        "supersedes the plain COMET checkpoint. Gives fine-grained error spans too. "
        "shared.metrics::compute_xcomet.",
    ),
    "cometkiwi": MetricSpec(
        "cometkiwi", frozenset({TaskType.TRANSLATION}),
        "unbabel-comet (checkpoint: Unbabel/wmt23-cometkiwi-da-xl)", "implemented",
        "Reference-FREE quality estimation -- useful where there's no gold "
        "translation (e.g. many FLORES+ target-language generations). "
        "shared.metrics::compute_cometkiwi.",
    ),
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class Combination:
    model: str
    baseline: str
    benchmark: str
    language: Optional[str]
    metric: str


def validate_combination(
    model: str, baseline: str, benchmark: str, language: Optional[str], metric: str,
) -> Tuple[bool, str]:
    """Return (is_valid, reason). `reason` explains why when invalid, else 'ok'."""
    if model not in MODELS:
        return False, f"unknown model '{model}'"
    if baseline not in BASELINES:
        return False, f"unknown baseline '{baseline}'"
    if benchmark not in BENCHMARKS:
        return False, f"unknown benchmark '{benchmark}'"
    if metric not in METRICS:
        return False, f"unknown metric '{metric}'"

    m, b, bm, met = MODELS[model], BASELINES[baseline], BENCHMARKS[benchmark], METRICS[metric]

    if not m.causal_lm:
        return False, f"model '{model}' is not causal-LM compatible ({m.notes or 'excluded'})"
    if not b.runnable:
        return False, f"baseline '{baseline}' has no benchmark integration yet ({b.notes})"
    if b.supported_benchmarks is not None and benchmark not in b.supported_benchmarks:
        return False, (
            f"baseline '{baseline}' does not support benchmark '{benchmark}'; "
            f"supported: {sorted(b.supported_benchmarks) or '(none)'}"
        )
    if bm.languages is not None:
        if language is None:
            return False, f"benchmark '{benchmark}' requires a language argument"
        if language not in bm.languages:
            return False, (
                f"benchmark '{benchmark}' has no data for language '{language}'; "
                f"supported: {sorted(bm.languages)}"
            )
    else:
        if language is not None and benchmark not in ("flores_plus", "sea_safeguardbench"):
            return False, f"benchmark '{benchmark}' has no language axis; pass language=None"
    if bm.task_type not in met.applicable_task_types:
        return False, (
            f"metric '{metric}' does not apply to task type '{bm.task_type.value}' "
            f"(benchmark '{benchmark}')"
        )
    return True, "ok"


def enumerate_valid_combinations(
    models: Optional[List[str]] = None,
    baselines: Optional[List[str]] = None,
    benchmarks: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    include_recommended: bool = False,
) -> List[Combination]:
    """Enumerate every valid (model, baseline, benchmark, language, metric) tuple.

    Filters default to every registered entry; pass explicit lists to narrow the search.
    `include_recommended=False` (default) restricts to models/metrics with
    status == "implemented"/"in_use" so the result is only combinations you can run today.
    """
    def _status_ok(spec) -> bool:
        return include_recommended or spec.status in ("in_use", "implemented", "verified")

    # Explicit filter lists are still gated by include_recommended -- passing
    # models=["some-recommended-model"] should not bypass the "runnable today" default.
    model_pool = [m for m in (models or MODELS) if m in MODELS and _status_ok(MODELS[m])]
    baseline_pool = baselines or list(BASELINES)
    benchmark_pool = [b for b in (benchmarks or BENCHMARKS) if b in BENCHMARKS and _status_ok(BENCHMARKS[b])]
    metric_pool = [m for m in (metrics or METRICS) if m in METRICS and _status_ok(METRICS[m])]

    results: List[Combination] = []
    for model, baseline, benchmark, metric in product(model_pool, baseline_pool, benchmark_pool, metric_pool):
        bm = BENCHMARKS.get(benchmark)
        lang_pool = (
            [l for l in (languages or []) if bm and bm.languages and l in bm.languages]
            if languages else (sorted(bm.languages) if bm and bm.languages else [None])
        )
        for language in lang_pool or [None]:
            ok, _ = validate_combination(model, baseline, benchmark, language, metric)
            if ok:
                results.append(Combination(model, baseline, benchmark, language, metric))
    return results
