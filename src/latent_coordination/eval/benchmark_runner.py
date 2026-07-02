"""Benchmark runner for the multi-agent coordination system.

Executes tasks on real agents loaded with Qwen3.5-9B and measures
latency, token cost, and task accuracy across three communication modes:
  - single_agent_baseline: one agent handles the full task
  - token_based_mas: agents communicate via decoded text strings
  - latent_based_mas_ours: agents communicate via latent state transfers
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

__author__ = "Himon Thakur"
__copyright__ = "Copyright [2026], Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


logger = logging.getLogger(__name__)


@dataclass
class MultiAgentBenchmarkReport:
    """Contains results of multi-agent evaluations against standard baselines."""
    timestamp: str
    results_by_mode: Dict[str, Dict[str, float]] = field(default_factory=dict)
    task_details: Dict[str, List[Dict]] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        # task_details embed AgentResponse.latent_state tensors → make JSON-safe.
        from shared.serialization import to_json_safe
        return to_json_safe(asdict(self))

    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate a comparison table formatting metrics for paper submission."""
        rows = []
        for mode, metrics in self.results_by_mode.items():
            rows.append({
                "System Setup / Communication Mode": mode.replace("_", " ").title(),
                "Task Accuracy": metrics.get("accuracy", 0.0),
                "Communication Latency (ms)": metrics.get("latency_ms", 999.0),
                "Overhead Token Cost": metrics.get("token_cost", 0.0),
                "Safety Pass Rate": metrics.get("safety_rate", 0.0),
            })
        return pd.DataFrame(rows)

    def save_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("MultiAgentBenchmarkReport saved to %s", path)


class MultiAgentBenchmarkRunner:
    """Orchestrates benchmark evaluation of the multi-agent coordination system.

    Runs the same set of tasks under three setups and measures real latency,
    token overhead, and task accuracy from agent responses.
    """

    def __init__(
        self,
        output_dir: Optional[Path | str] = "results/coordination",
        max_samples_per_language: Optional[int] = None,
        languages: Optional[List[str]] = None,
        translation_metrics: Optional[Dict[str, bool]] = None,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("results/coordination")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Optional target-language subset (ISO-639-1). None = the full FLORES+ benchmark set.
        self.languages = list(languages) if languages else None
        # Per-language cap on FLORES+ tasks. ``None`` means use the full devtest
        # split (1012/language). Honors ``benchmarks.flores_plus.n_samples_per_language``.
        if max_samples_per_language is not None and max_samples_per_language <= 0:
            raise ValueError(
                f"max_samples_per_language must be a positive int or None, got {max_samples_per_language}"
            )
        self.max_samples_per_language = max_samples_per_language
        # Real translation-quality scoring against the FLORES+ gold reference (task.context)
        # and source (task.query). chrF is cheap and on by default; xcomet/cometkiwi load
        # multi-GB checkpoints so they're opt-in (see dev_doc.md "Recommended upgrade path
        # for COMET"). Keys: "chrf" | "xcomet" | "cometkiwi".
        self.translation_metrics = {"chrf": True, "xcomet": False, "cometkiwi": False}
        if translation_metrics:
            self.translation_metrics.update(translation_metrics)

    def _compute_translation_quality(
        self, answers: List[str], scored_tasks: List,
    ) -> Dict[str, float]:
        """Score `answers` against each task's FLORES+ gold reference (task.context) and
        source (task.query). Only meaningful for FLORES+-sourced tasks -- callers must
        only invoke this with (answer, task) pairs that actually came from FLORES+.
        """
        if not answers or not scored_tasks:
            return {}
        references = [t.context for t in scored_tasks]
        sources = [t.query for t in scored_tasks]
        metrics: Dict[str, float] = {}
        if self.translation_metrics.get("chrf"):
            from shared.metrics import compute_chrf
            metrics["chrf"] = compute_chrf(answers, references)
        if self.translation_metrics.get("xcomet"):
            from shared.metrics import compute_xcomet
            metrics["xcomet"] = compute_xcomet(answers, references, sources)
        if self.translation_metrics.get("cometkiwi"):
            from shared.metrics import compute_cometkiwi
            metrics["cometkiwi"] = compute_cometkiwi(answers, sources)
        return metrics

    @staticmethod
    def _device_hint() -> str:
        """A representative CUDA device for the vLLM capability probe (else cpu)."""
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _compute_accuracy(self, responses: List, tasks: List) -> float:
        """Completeness proxy: fraction of substantive answers that are non-empty.

        ``responses`` must already be the *substantive* answers (see
        :func:`latent_coordination.eval.scoring.select_answer`), never raw safety
        verdicts — a ``[SAFE]``/``[UNSAFE]`` verdict starts with ``[`` and would be
        counted as an error here. This is a completeness proxy, not translation
        correctness; callers needing correctness should score against references.
        """
        if not responses or not tasks:
            return 0.0
        correct = sum(
            1 for resp in responses
            if resp.output_text and not resp.output_text.startswith("[")
        )
        return correct / len(responses)

    def _compute_safety_rate(self, responses: List) -> float:
        """Compute safety pass rate from SafetyAgent metadata in responses."""
        safety_verdicts = [
            r.metadata.get("safety_verdict", {}).get("is_safe", True)
            for r in responses
            if "safety_verdict" in r.metadata
        ]
        if not safety_verdicts:
            return 1.0  # no safety agent = assume safe
        return sum(safety_verdicts) / len(safety_verdicts)

    # Communication modes. The first two are token-only (consume agent output_text and are
    # vLLM-eligible); the last transfers hidden states and therefore requires the HF backend.
    TOKEN_ONLY_MODES = ("single_agent_baseline", "token_based_mas")
    LATENT_MODES = ("latent_based_mas_ours",)
    ALL_MODES = TOKEN_ONLY_MODES + LATENT_MODES

    # ------------------------------------------------------------------
    # Per-mode evaluators (each returns (metrics_dict, responses_list))
    # ------------------------------------------------------------------

    def _eval_single_agent(self, router, tasks):
        t0 = time.perf_counter()
        responses = []
        scored_tasks = []
        for task in tasks:
            plan = router.route(task)
            if plan.selected_agents:
                responses.append(router.agents[plan.selected_agents[0]].process(task))
                scored_tasks.append(task)
        latency_ms = (time.perf_counter() - t0) / max(len(tasks), 1) * 1000
        metrics = {
            "accuracy": self._compute_accuracy(responses, tasks),
            "latency_ms": latency_ms,
            "token_cost": float(sum(len(r.output_text.split()) for r in responses)) / max(len(tasks), 1),
            "safety_rate": self._compute_safety_rate(responses),
        }
        metrics.update(self._compute_translation_quality(
            [r.output_text for r in responses], scored_tasks,
        ))
        return metrics, responses

    def _eval_token_based(self, router, tasks):
        from latent_coordination.agents.base_agent import AgentTask
        from latent_coordination.eval.scoring import is_safety_response, select_answer
        t0 = time.perf_counter()
        answers, safety, scored_tasks = [], [], []
        total_token_cost = 0.0
        for task in tasks:
            plan = router.route(task)
            context = task.context or ""
            step_responses = []
            for aid in plan.execution_order:
                agent = router.agents[aid]
                text_task = AgentTask(
                    task_id=f"{task.task_id}_token_{aid}",
                    query=task.query,
                    context=context,
                    latent_state=None,   # token mode: text only, no latent transfer
                    target_language=task.target_language,
                )
                resp = agent.process(text_task)
                context = resp.output_text
                total_token_cost += len(resp.output_text.split())
                step_responses.append(resp)
            # Score the substantive answer (last non-safety step), not the safety verdict.
            answer = select_answer(step_responses)
            if answer is not None:
                answers.append(answer)
                scored_tasks.append(task)
            safety.extend(r for r in step_responses if is_safety_response(r))
        latency_ms = (time.perf_counter() - t0) / max(len(tasks), 1) * 1000
        metrics = {
            "accuracy": self._compute_accuracy(answers, tasks),
            "latency_ms": latency_ms,
            "token_cost": total_token_cost / max(len(tasks), 1),
            "safety_rate": self._compute_safety_rate(safety),
        }
        metrics.update(self._compute_translation_quality(
            [a.output_text for a in answers], scored_tasks,
        ))
        return metrics, answers

    def _eval_latent(self, router, tasks, universal_space):
        from latent_coordination.eval.scoring import is_safety_response, select_answer
        t0 = time.perf_counter()
        answers, safety, scored_tasks = [], [], []
        for task in tasks:
            orch_result = router.execute(task, router.route(task), universal_space)
            chain = orch_result.agent_responses
            if chain:
                # Score the substantive answer (last non-safety agent), not the safety verdict.
                answer = select_answer(chain)
                if answer is not None:
                    answers.append(answer)
                    scored_tasks.append(task)
                safety.extend(r for r in chain if is_safety_response(r))
        latency_ms = (time.perf_counter() - t0) / max(len(tasks), 1) * 1000
        metrics = {
            "accuracy": self._compute_accuracy(answers, tasks),
            "latency_ms": latency_ms,
            "token_cost": 0.0,   # no token-level communication overhead
            "safety_rate": self._compute_safety_rate(safety),
        }
        metrics.update(self._compute_translation_quality(
            [a.output_text for a in answers], scored_tasks,
        ))
        return metrics, answers

    def run_eval(
        self,
        router,
        tasks,
        universal_space,
        modes=None,
        backend_name: str = "auto",
        checkpoint_manager=None,
        cache_prefix: Optional[str] = None,
    ) -> MultiAgentBenchmarkReport:
        """Run the selected communication modes and measure real performance.

        Parameters
        ----------
        modes : list[str] or None
            Subset of ``ALL_MODES`` to evaluate. Defaults to all three.
        backend_name : {"auto", "hf", "vllm"}
            Backend for token-only modes. vLLM is gated to Ampere+ (see
            ``shared.generation_backend``); on V100 it transparently falls back to HF.
        checkpoint_manager, cache_prefix :
            If both given, each mode's result is cached under
            ``f"{cache_prefix}::mode::{mode}"`` and reused on a later run — so changing the
            requested ``modes`` (or recovering from a crash) never recomputes a finished mode.

        Returns
        -------
        MultiAgentBenchmarkReport (only the requested/cached modes populated).
        """
        if tasks is None:
            tasks = self._load_real_tasks()
        if not tasks:
            raise RuntimeError(
                "No tasks provided and FLORES+ task loading failed. "
                "Provide real AgentTask objects to run_eval()."
            )

        modes = list(modes) if modes else list(self.ALL_MODES)
        invalid = [m for m in modes if m not in self.ALL_MODES]
        if invalid:
            raise ValueError(f"Unknown comm-mode(s): {invalid}. Valid: {self.ALL_MODES}")

        # Resolve the token-mode backend once (logs HF fallback on V100). The latent mode
        # always uses the HF-hooked agents regardless. Run token-only modes before the latent
        # mode so a vLLM engine (when active) is torn down before HF hooks need the GPU.
        from shared.generation_backend import vllm_supported
        token_backend = "vllm" if (backend_name in ("auto", "vllm") and vllm_supported(self._device_hint())) else "hf"
        logger.info(
            "Executing Multi-Agent Benchmark on %d tasks | modes=%s | token-backend=%s",
            len(tasks), modes, token_backend,
        )

        ordered = [m for m in self.ALL_MODES if m in modes]  # token-only first, latent last
        results: Dict[str, Dict[str, float]] = {}
        task_details: Dict[str, List[Dict]] = {}

        for mode in ordered:
            cache_key = f"{cache_prefix}::mode::{mode}" if cache_prefix else None
            if checkpoint_manager is not None and cache_key and checkpoint_manager.has_result(cache_key):
                cached = checkpoint_manager.get_result(cache_key)
                results[mode] = cached["metrics"]
                task_details[mode] = cached["task_details"]
                logger.info("Mode '%s' loaded from cache.", mode)
                continue

            logger.info("Evaluating Mode: %s", mode)
            if mode == "single_agent_baseline":
                metrics, responses = self._eval_single_agent(router, tasks)
            elif mode == "token_based_mas":
                metrics, responses = self._eval_token_based(router, tasks)
            else:
                metrics, responses = self._eval_latent(router, tasks, universal_space)

            details = [asdict(r) for r in responses]
            results[mode] = metrics
            task_details[mode] = details
            if checkpoint_manager is not None and cache_key:
                checkpoint_manager.cache_result(
                    cache_key, {"metrics": metrics, "task_details": details}
                )
            logger.info("Mode '%s' complete | accuracy=%.3f", mode, metrics["accuracy"])

        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report = MultiAgentBenchmarkReport(
            timestamp=ts,
            results_by_mode=results,
            task_details=task_details,
            metadata={"n_tasks": len(tasks), "modes": modes, "token_backend": token_backend},
        )
        out_path = self.output_dir / f"multiagent_benchmark_{ts}.json"
        report.save_json(out_path)
        return report

    def _load_real_tasks(self) -> List:
        """Load real evaluation tasks from FLORES+ via Hugging Face datasets.

        Returns
        -------
        List[AgentTask]
            Tasks sourced from FLORES+ devtest split for Thai, Burmese, and Khmer.
        """
        from latent_coordination.agents.base_agent import AgentTask

        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The 'datasets' library is required. Install with: pip install datasets"
            ) from exc

        tasks = []
        lang_pairs = [
            ("tha_Thai", "th"),
            ("mya_Mymr", "my"),
            ("khm_Khmr", "km"),
            ("lao_Laoo", "lo"),
            ("amh_Ethi", "am"),
            ("swh_Latn", "sw"),
        ]
        # Honor an optional language subset (--languages); default = all.
        if self.languages:
            wanted = set(self.languages)
            lang_pairs = [(f, i) for (f, i) in lang_pairs if i in wanted]
            if not lang_pairs:
                raise ValueError(
                    f"None of the requested languages {sorted(wanted)} are in the FLORES+ "
                    f"benchmark set (th, my, km, lo, am, sw)."
                )

        for flores_code, iso_code in lang_pairs:
            try:
                en_ds = load_dataset(
                    "openlanguagedata/flores_plus", name="eng_Latn",
                    split="devtest"
                )
                tgt_ds = load_dataset(
                    "openlanguagedata/flores_plus", name=flores_code,
                    split="devtest"
                )
                # Honor the configured per-language cap (full devtest if None).
                n_avail = min(len(en_ds), len(tgt_ds))
                n_take = n_avail if self.max_samples_per_language is None else min(
                    n_avail, self.max_samples_per_language
                )
                for i in range(n_take):
                    en_text = en_ds[i]["text"]
                    tgt_text = tgt_ds[i]["text"]
                    if en_text and tgt_text:
                        tasks.append(AgentTask(
                            task_id=f"flores_plus_{iso_code}_{i}",
                            query=en_text,
                            context=tgt_text,
                            target_language=iso_code,
                        ))
                logger.info(
                    "Loaded %d/%d FLORES+ tasks for '%s' (cap=%s).",
                    n_take, n_avail, iso_code,
                    self.max_samples_per_language if self.max_samples_per_language is not None else "all",
                )
            except Exception as exc:
                logger.error("Failed to load FLORES+ for '%s': %s", iso_code, exc)

        logger.info("Total benchmark tasks loaded: %d", len(tasks))
        return tasks
