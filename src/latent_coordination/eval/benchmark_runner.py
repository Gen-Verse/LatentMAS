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


def _code_regime() -> Dict[str, str]:
    """Fingerprint of the code that produced a result, stamped into every mode
    cache and report. The bench-suite driver retries a crashed run with whatever
    is on disk, so without this stamp, pre-fix (uniform-router) and post-fix
    (prototype-seeded adaptive router) results are indistinguishable once cached.
    """
    import subprocess
    repo_root = Path(__file__).resolve().parents[3]
    regime: Dict[str, str] = {}
    try:
        regime["git_sha"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        regime["git_dirty"] = "yes" if dirty else "no"
    except Exception:
        regime["git_sha"] = "unknown"
    try:
        import latent_coordination.orchestration.router as _router_mod
        regime["router"] = (
            "prototype-seeded" if hasattr(_router_mod, "ROLE_KEY_PROTOTYPES")
            else "uniform-random-keys"
        )
    except Exception:
        regime["router"] = "unknown"
    return regime


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
        benchmarks: Optional[Dict] = None,
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
        # Real translation-quality scoring against the FLORES+ gold reference (task.reference)
        # and source (task.query). chrF is cheap and on by default; xcomet/cometkiwi load
        # multi-GB checkpoints so they're opt-in (see dev_doc.md "Recommended upgrade path
        # for COMET"). Keys: "chrf" | "xcomet" | "cometkiwi".
        self.translation_metrics = {"chrf": True, "xcomet": False, "cometkiwi": False}
        if translation_metrics:
            self.translation_metrics.update(translation_metrics)
        # Raw configs/*.yaml ``benchmarks`` section. Drives multi-benchmark task
        # loading (flores_plus + mgsm + belebele + sea_vision +
        # sea_safeguardbench) and correctness scoring. None → FLORES+ only,
        # matching the historical behaviour.
        self.benchmarks = dict(benchmarks) if benchmarks else {}

    def _compute_translation_quality(
        self, answers: List[str], scored_tasks: List,
    ) -> Dict[str, float]:
        """Score `answers` against each task's FLORES+ gold reference (task.reference)
        and source (task.query). Only meaningful for FLORES+-sourced tasks -- callers
        must only invoke this with (answer, task) pairs that actually came from FLORES+.
        """
        if not answers or not scored_tasks:
            return {}
        references = [t.reference for t in scored_tasks]
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

    def _compute_accuracy(self, responses: List, tasks: List) -> float:
        """Completeness proxy: fraction of *tasks* that produced a substantive answer.

        ``responses`` must already be the *substantive* answers (see
        :func:`latent_coordination.eval.scoring.select_answer`), never raw safety
        verdicts — a ``[SAFE]``/``[UNSAFE]`` verdict starts with ``[`` and would be
        counted as an error here. This is a completeness proxy, not translation
        correctness; callers needing correctness should score against references.

        The denominator is ``len(tasks)``, NOT ``len(responses)``: a task that
        produced no answer at all (routing selected no agent, or every step was a
        safety verdict) is a failure, and dividing by the number of surviving
        answers silently inflated the score by dropping exactly those failures.
        """
        if not responses or not tasks:
            return 0.0
        correct = sum(
            1 for resp in responses
            if resp.output_text and not resp.output_text.startswith("[")
        )
        return correct / len(tasks)

    # ------------------------------------------------------------------
    # Real correctness (dev_doc.md §9 gap 1: the completeness proxy above is
    # NOT accuracy — gold-carrying benchmarks are scored for real here)
    # ------------------------------------------------------------------

    @staticmethod
    def _task_benchmark(task) -> str:
        """Benchmark a task belongs to (metadata-driven; FLORES by default)."""
        return (getattr(task, "metadata", None) or {}).get("benchmark", "flores_plus")

    @staticmethod
    def _pick_scoring_agent(router):
        """The substantive agent whose model scores log-likelihood MCQA.

        Prefers a LOADED non-safety agent (after an eval pass the executing
        agents are loaded); falls back to any non-safety agent.
        """
        candidates = [
            a for a in router.agents.values()
            if getattr(getattr(a, "config", None), "role", None) != "safety"
        ]
        for agent in candidates:
            if getattr(agent, "_model", None) is not None:
                return agent
        return candidates[0] if candidates else None

    _OPTION_PATTERN = None  # compiled lazily

    @classmethod
    def _parse_option_number(cls, text: str) -> Optional[int]:
        """Extract a 1-based MCQA option number from generated text."""
        import re
        if cls._OPTION_PATTERN is None:
            cls._OPTION_PATTERN = re.compile(r"\b([1-4])\b")
        m = cls._OPTION_PATTERN.search(text or "")
        return int(m.group(1)) if m else None

    def _compute_correctness(
        self,
        answers: List,
        scored_tasks: List,
        all_tasks: List,
        router,
        safety_responses: Optional[List] = None,
    ) -> Dict[str, float]:
        """Per-benchmark REAL accuracy over gold-carrying tasks.

        Denominators are per-benchmark counts over ``all_tasks`` — a task that
        produced no substantive answer is a failure, never silently dropped
        (same rule as the completeness proxy).

        Returns ``{"accuracy_<benchmark>": float, ...}`` plus the aggregate
        ``accuracy`` / ``accuracy_kind`` keys: when any gold-carrying tasks
        exist, ``accuracy`` is the task-count-weighted mean of the real
        per-benchmark accuracies (``accuracy_kind='correctness'``); otherwise
        it falls back to the completeness proxy
        (``accuracy_kind='completeness_proxy'``).
        """
        totals: Dict[str, int] = {}
        for t in all_tasks:
            b = self._task_benchmark(t)
            totals[b] = totals.get(b, 0) + 1

        by_bench: Dict[str, List] = {}
        for ans, task in zip(answers, scored_tasks):
            by_bench.setdefault(self._task_benchmark(task), []).append((ans, task))

        metrics: Dict[str, float] = {}
        correct_by_bench: Dict[str, int] = {}

        # --- MGSM: exact-match on the extracted final number -------------
        if totals.get("mgsm"):
            from latent_coordination.eval.correctness import score_mgsm
            n_correct = sum(
                1 for ans, task in by_bench.get("mgsm", [])
                if score_mgsm(
                    getattr(ans, "output_text", None) or str(ans),
                    float(task.metadata["gold_answer"]),
                ).is_correct
            )
            correct_by_bench["mgsm"] = n_correct
            metrics["accuracy_mgsm"] = n_correct / totals["mgsm"]

        # --- AfriMGSM: exact-match on the extracted final number ---------
        if totals.get("afrimgsm"):
            from latent_coordination.eval.correctness import score_mgsm
            n_correct = sum(
                1 for ans, task in by_bench.get("afrimgsm", [])
                if score_mgsm(
                    getattr(ans, "output_text", None) or str(ans),
                    float(task.metadata["gold_answer"]),
                ).is_correct
            )
            correct_by_bench["afrimgsm"] = n_correct
            metrics["accuracy_afrimgsm"] = n_correct / totals["afrimgsm"]

        # --- Belebele: 4-choice MCQA -------------------------------------
        if totals.get("belebele"):
            scoring = (
                self.benchmarks.get("belebele", {}).get("scoring", "loglikelihood")
            )
            if scoring == "loglikelihood":
                # Teacher-forced log-likelihood over the 4 choices with the
                # substantive agent's already-loaded model (the protocol the
                # gap explicitly names). This probes the model directly and is
                # answer-independent, so ALL belebele tasks are scored.
                from latent_coordination.eval.correctness import CorrectnessScorer
                agent = self._pick_scoring_agent(router)
                if agent is None or getattr(agent, "_model", None) is None:
                    raise RuntimeError(
                        "Belebele log-likelihood scoring needs a loaded non-safety "
                        "agent model; none is available. Use benchmarks.belebele."
                        "scoring='generative' for stub/offline runs."
                    )
                scorer = CorrectnessScorer(
                    agent._model, agent._tokenizer, device=str(agent._device)
                )
                bele_tasks = [t for t in all_tasks if self._task_benchmark(t) == "belebele"]
                report = scorer.score_multiple_choice_batch(
                    prompts=[t.metadata["prompt_stem"] for t in bele_tasks],
                    choices_list=[t.metadata["choices"] for t in bele_tasks],
                    gold_indices=[t.metadata["correct_idx"] for t in bele_tasks],
                    benchmark="belebele",
                )
                correct_by_bench["belebele"] = report.n_correct
                metrics["accuracy_belebele"] = report.accuracy
                # The log-likelihood probe above is answer-independent — it scores
                # the scoring agent's *model*, not what this comm-mode generated,
                # so it cannot separate comm-modes. Also score the mode's actual
                # generated answers so mode-vs-mode Belebele deltas are real.
                n_gen_correct = sum(
                    1 for ans, task in by_bench.get("belebele", [])
                    if self._parse_option_number(getattr(ans, "output_text", None) or str(ans))
                    == task.metadata["correct_idx"] + 1
                )
                metrics["accuracy_belebele_generative"] = n_gen_correct / totals["belebele"]
            elif scoring == "generative":
                n_correct = sum(
                    1 for ans, task in by_bench.get("belebele", [])
                    if self._parse_option_number(getattr(ans, "output_text", None) or str(ans))
                    == task.metadata["correct_idx"] + 1
                )
                correct_by_bench["belebele"] = n_correct
                metrics["accuracy_belebele"] = n_correct / totals["belebele"]
            else:
                raise ValueError(
                    f"Unknown benchmarks.belebele.scoring '{scoring}' "
                    "(valid: loglikelihood, generative)."
                )

        # --- SEA-Vision QA: normalized reference containment --------------
        if totals.get("sea_vision"):
            def _norm(s: str) -> str:
                return " ".join((s or "").lower().split())
            n_correct = 0
            for ans, task in by_bench.get("sea_vision", []):
                text = _norm(getattr(ans, "output_text", None) or str(ans))
                gold = _norm(task.reference or "")
                if gold and (gold == text or gold in text):
                    n_correct += 1
            correct_by_bench["sea_vision"] = n_correct
            metrics["accuracy_sea_vision"] = n_correct / totals["sea_vision"]

        # --- SEA safety benchmark: verdict agreement ----------------------
        if totals.get("sea_safeguardbench"):
            n_correct = 0
            safety_responses = safety_responses or []
            for task in all_tasks:
                if self._task_benchmark(task) != "sea_safeguardbench":
                    continue
                expected_safe = task.metadata.get("expected_verdict") == "safe"
                resp = next(
                    (r for r in safety_responses
                     if r.task_id == task.task_id
                     or r.task_id.startswith(f"{task.task_id}_")),
                    None,
                )
                if resp is None:
                    continue  # no safety verdict for this task = failure
                verdict = resp.metadata.get("safety_verdict", {})
                if bool(verdict.get("is_safe", True)) == expected_safe:
                    n_correct += 1
            correct_by_bench["sea_safeguardbench"] = n_correct
            metrics["accuracy_sea_safeguardbench"] = n_correct / totals["sea_safeguardbench"]

        # --- Aggregate -----------------------------------------------------
        gold_total = sum(totals[b] for b in correct_by_bench)
        if gold_total:
            metrics["accuracy"] = sum(correct_by_bench.values()) / gold_total
            metrics["accuracy_kind"] = "correctness"
        return metrics

    def _assemble_metrics(
        self,
        answers: List,
        scored_tasks: List,
        tasks: List,
        router,
        safety_responses: Optional[List] = None,
    ) -> Dict[str, float]:
        """Completeness proxy + real per-benchmark correctness + translation quality."""
        completeness = self._compute_accuracy(answers, tasks)
        metrics: Dict[str, float] = {
            "completeness": completeness,
            # Overwritten by real correctness below when gold-carrying tasks exist.
            "accuracy": completeness,
            "accuracy_kind": "completeness_proxy",
        }
        metrics.update(self._compute_correctness(
            answers, scored_tasks, tasks, router, safety_responses,
        ))
        # Translation quality is only meaningful for FLORES+-sourced pairs.
        flores_pairs = [
            (a, t) for a, t in zip(answers, scored_tasks)
            if self._task_benchmark(t) == "flores_plus" and t.reference
        ]
        if flores_pairs:
            metrics.update(self._compute_translation_quality(
                [getattr(a, "output_text", None) or str(a) for a, _ in flores_pairs],
                [t for _, t in flores_pairs],
            ))
        return metrics

    @staticmethod
    def _count_tokens(text: str, agent) -> int:
        """Real token count of ``text`` using the producing agent's tokenizer.

        The agent's tokenizer is loaded lazily with the model, so after a real
        ``process()`` call it is always present. Whitespace splitting is only a
        last-resort fallback (unit-test stubs without a tokenizer) and is wrong
        for unsegmented scripts — Thai/Burmese/Khmer/Lao text contains no spaces,
        so ``split()`` counts an entire sentence as ~1 "token" and silently
        underreports token cost for exactly the languages this project targets.
        """
        if not text:
            return 0
        tokenizer = getattr(agent, "_tokenizer", None)
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:  # noqa: BLE001 — fall through to whitespace
                pass
        return len(text.split())

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

    def _pick_single_agent(self, router, plan, task=None) -> Optional[str]:
        """Pick the substantive executor for the single-agent baseline.

        A safety agent's output is a ``[SAFE]/[UNSAFE]`` verdict, not a task
        answer — if the router happens to rank it first, using it as the sole
        executor makes the baseline score 0 by construction. Prefer the first
        selected agent whose role is not 'safety'; fall back to the first.
        Exception: for safety-benchmark tasks (sea_safeguardbench) the verdict
        IS the answer, so the safety agent is preferred instead.
        """
        want_safety = task is not None and self._task_benchmark(task) == "sea_safeguardbench"
        for aid in plan.selected_agents:
            agent = router.agents.get(aid)
            role = getattr(getattr(agent, "config", None), "role", None)
            if (role == "safety") == want_safety:
                return aid
        # A safety task with no safety agent routed: fall back to ANY safety
        # agent before giving up (the verdict is the only scoreable output).
        if want_safety:
            for aid, agent in router.agents.items():
                if getattr(getattr(agent, "config", None), "role", None) == "safety":
                    return aid
        return plan.selected_agents[0] if plan.selected_agents else None

    # ------------------------------------------------------------------
    # Chunked/checkpointed execution: a mode's tasks are grouped into
    # contiguous (benchmark, language) chunks, and progress is persisted
    # after every chunk under f"{cache_key}::partial" (when a checkpoint_manager
    # + cache_key are given). A crash mid-mode therefore only loses the
    # in-flight chunk (<= one language's worth of tasks) instead of the whole
    # mode -- previously only a fully-finished mode was ever cached, so a kill
    # after e.g. 7/9 languages of token_based_mas discarded ~28h of compute
    # (observed live 2026-07-05 on the bench_suite belebele_sg runs).
    # ------------------------------------------------------------------

    def _task_chunk_key(self, task) -> str:
        return f"{self._task_benchmark(task)}::{getattr(task, 'target_language', None)}"

    def _iter_task_chunks(self, tasks):
        """Group *tasks* into contiguous runs sharing the same chunk key."""
        chunks = []
        cur_key, cur = None, []
        for t in tasks:
            k = self._task_chunk_key(t)
            if k != cur_key and cur:
                chunks.append((cur_key, cur))
                cur = []
            cur_key, cur = k, cur + [t]
        if cur:
            chunks.append((cur_key, cur))
        return chunks

    def _run_mode_chunked(self, mode, process_fn, tasks, checkpoint_manager, cache_key):
        """Run ``process_fn(task) -> (answer_or_None, safety_responses, token_cost)``
        over *tasks* in per-(benchmark, language) chunks, checkpointing the
        accumulated state after each chunk so a resume skips completed chunks.
        """
        partial_key = f"{cache_key}::partial" if cache_key else None
        state = {
            "done_chunks": [], "answers": [], "scored_tasks": [],
            "safety": [], "token_cost": 0.0, "elapsed_s": 0.0,
        }
        if checkpoint_manager is not None and partial_key and checkpoint_manager.has_result(partial_key):
            state = checkpoint_manager.get_result(partial_key)
            logger.info(
                "Mode '%s' resuming from partial checkpoint | chunks_done=%d answers=%d",
                mode, len(state["done_chunks"]), len(state["answers"]),
            )
        done = set(state["done_chunks"])
        chunks = self._iter_task_chunks(tasks)
        for chunk_key, chunk_tasks in chunks:
            if chunk_key in done:
                continue
            t0 = time.perf_counter()
            for task in chunk_tasks:
                answer, safety, cost = process_fn(task)
                if answer is not None:
                    state["answers"].append(answer)
                    state["scored_tasks"].append(task)
                state["safety"].extend(safety)
                state["token_cost"] += cost
            state["elapsed_s"] += time.perf_counter() - t0
            done.add(chunk_key)
            state["done_chunks"] = list(done)
            if checkpoint_manager is not None and partial_key:
                checkpoint_manager.cache_result(partial_key, state)
            logger.info(
                "Mode '%s' chunk complete | chunk=%s (%d/%d chunks done)",
                mode, chunk_key, len(done), len(chunks),
            )
        return state

    def _finalize_mode_metrics(self, mode, state, tasks, router, safety_rate_source: str):
        latency_ms = state["elapsed_s"] / max(len(tasks), 1) * 1000
        metrics = self._assemble_metrics(
            state["answers"], state["scored_tasks"], tasks, router,
            safety_responses=state["safety"],
        )
        safety_pool = state["answers"] if safety_rate_source == "answers" else state["safety"]
        metrics.update({
            "latency_ms": latency_ms,
            "token_cost": state["token_cost"] / max(len(tasks), 1),
            "safety_rate": self._compute_safety_rate(safety_pool),
        })
        return metrics

    def _process_task_single_agent(self, router, task):
        from latent_coordination.eval.scoring import is_safety_response
        plan = router.route(task)
        aid = self._pick_single_agent(router, plan, task)
        if aid is None:
            return None, [], 0.0
        agent = router.agents[aid]
        resp = agent.process(task)
        cost = self._count_tokens(resp.output_text, agent)
        safety = [resp] if is_safety_response(resp) else []
        return resp, safety, cost

    def _eval_single_agent(self, router, tasks, checkpoint_manager=None, cache_key=None):
        state = self._run_mode_chunked(
            "single_agent_baseline",
            lambda task: self._process_task_single_agent(router, task),
            tasks, checkpoint_manager, cache_key,
        )
        metrics = self._finalize_mode_metrics(
            "single_agent_baseline", state, tasks, router, safety_rate_source="answers",
        )
        return metrics, state["answers"]

    def _process_task_token_based(self, router, task):
        from latent_coordination.agents.base_agent import AgentTask
        from latent_coordination.eval.scoring import is_safety_response, select_answer
        plan = router.route(task)
        context = task.context or ""
        step_responses = []
        token_cost = 0.0
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
            token_cost += self._count_tokens(resp.output_text, agent)
            step_responses.append(resp)
        # Score the substantive answer (last non-safety step), not the safety verdict.
        answer = select_answer(step_responses)
        safety = [r for r in step_responses if is_safety_response(r)]
        return answer, safety, token_cost

    def _eval_token_based(self, router, tasks, checkpoint_manager=None, cache_key=None):
        state = self._run_mode_chunked(
            "token_based_mas",
            lambda task: self._process_task_token_based(router, task),
            tasks, checkpoint_manager, cache_key,
        )
        metrics = self._finalize_mode_metrics(
            "token_based_mas", state, tasks, router, safety_rate_source="safety",
        )
        return metrics, state["answers"]

    def _process_task_latent(self, router, task, universal_space):
        from latent_coordination.eval.scoring import is_safety_response, select_answer
        orch_result = router.execute(task, router.route(task), universal_space)
        chain = orch_result.agent_responses
        if not chain:
            return None, [], 0.0
        # Score the substantive answer (last non-safety agent), not the safety verdict.
        answer = select_answer(chain)
        safety = [r for r in chain if is_safety_response(r)]
        return answer, safety, 0.0

    def _eval_latent(self, router, tasks, universal_space, checkpoint_manager=None, cache_key=None):
        state = self._run_mode_chunked(
            "latent_based_mas_ours",
            lambda task: self._process_task_latent(router, task, universal_space),
            tasks, checkpoint_manager, cache_key,
        )
        metrics = self._finalize_mode_metrics(
            "latent_based_mas_ours", state, tasks, router, safety_rate_source="safety",
        )
        return metrics, state["answers"]

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

        # Backend honesty: every mode currently runs through the agents' own
        # HF-hooked models (BaseAgent.process). A vLLM engine is NOT wired into the
        # agent path yet, so previously "auto"-probing vllm_supported() and recording
        # "vllm" in the report metadata claimed a backend that never generated a
        # single token. Record the backend actually used, and fail loudly on an
        # explicit --backend vllm request instead of silently running HF under a
        # vLLM label.
        if backend_name == "vllm":
            raise NotImplementedError(
                "backend='vllm' was requested, but the token-only comm-modes still "
                "execute through each agent's HF model (BaseAgent.process); a vLLM "
                "engine is not wired into the agent path. Use backend='hf'/'auto', "
                "or wire VLLMBackend into the token-only agents first."
            )
        token_backend = "hf"
        logger.info(
            "Executing Multi-Agent Benchmark on %d tasks | modes=%s | token-backend=%s "
            "(agent-native HF models)",
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
                metrics, responses = self._eval_single_agent(
                    router, tasks, checkpoint_manager=checkpoint_manager, cache_key=cache_key,
                )
            elif mode == "token_based_mas":
                metrics, responses = self._eval_token_based(
                    router, tasks, checkpoint_manager=checkpoint_manager, cache_key=cache_key,
                )
            else:
                metrics, responses = self._eval_latent(
                    router, tasks, universal_space,
                    checkpoint_manager=checkpoint_manager, cache_key=cache_key,
                )

            details = [asdict(r) for r in responses]
            results[mode] = metrics
            task_details[mode] = details
            if checkpoint_manager is not None and cache_key:
                checkpoint_manager.cache_result(
                    cache_key,
                    {"metrics": metrics, "task_details": details,
                     "code_regime": _code_regime()},
                )
                # The per-chunk partial state has been fully superseded by the
                # completed-mode cache above; drop it so a future run doesn't
                # keep stale in-progress state around.
                checkpoint_manager.delete_result(f"{cache_key}::partial")
            logger.info("Mode '%s' complete | accuracy=%.3f", mode, metrics["accuracy"])

        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report = MultiAgentBenchmarkReport(
            timestamp=ts,
            results_by_mode=results,
            task_details=task_details,
            metadata={"n_tasks": len(tasks), "modes": modes, "token_backend": token_backend,
                      "code_regime": _code_regime()},
        )
        out_path = self.output_dir / f"multiagent_benchmark_{ts}.json"
        report.save_json(out_path)
        return report

    def _load_real_tasks(self) -> List:
        """Load real evaluation tasks for every enabled benchmark.

        FLORES+ (translation) is on by default, matching the historical
        behaviour; MGSM (math EM), Belebele (MCQA), SEA-Vision (QA) and
        SEA-SafeguardBench (safety) activate via the ``benchmarks`` config
        section (dev_doc.md §9 gaps 1 + 6 — the latter three used to be
        silently ignored config).
        """
        tasks: List = []
        if self.benchmarks.get("flores_plus", {}).get("enabled", True):
            tasks += self._load_flores_tasks()
        if self.benchmarks.get("mgsm", {}).get("enabled"):
            tasks += self._load_mgsm_agent_tasks(self.benchmarks["mgsm"])
        if self.benchmarks.get("afrimgsm", {}).get("enabled"):
            tasks += self._load_afrimgsm_agent_tasks(self.benchmarks["afrimgsm"])
        if self.benchmarks.get("belebele", {}).get("enabled"):
            tasks += self._load_belebele_agent_tasks(self.benchmarks["belebele"])
        if self.benchmarks.get("sea_vision", {}).get("enabled"):
            tasks += self._load_sea_vision_agent_tasks(self.benchmarks["sea_vision"])
        if self.benchmarks.get("sea_safeguardbench", {}).get("enabled"):
            tasks += self._load_sea_safeguard_agent_tasks(
                self.benchmarks["sea_safeguardbench"]
            )
        logger.info("Total benchmark tasks loaded (all enabled benchmarks): %d", len(tasks))
        return tasks

    # ISO-639-1 → Belebele/FLORES-style config codes for the tracked languages.
    _BELEBELE_LANG_MAP = {
        "th": "tha_Thai", "my": "mya_Mymr", "km": "khm_Khmr", "lo": "lao_Laoo",
        "am": "amh_Ethi", "sw": "swh_Latn", "bn": "ben_Beng", "te": "tel_Telu",
        "en": "eng_Latn",
    }

    def _benchmark_languages(self, cfg: Dict, supported: Optional[set] = None) -> List[str]:
        """Languages for a benchmark: explicit config list, else the runner's
        target languages filtered to what the benchmark actually covers."""
        langs = cfg.get("languages")
        if langs:
            return list(langs)
        langs = list(self.languages or ["en"])
        if supported is not None:
            covered = [l for l in langs if l in supported]
            if not covered:
                raise ValueError(
                    f"None of the target languages {langs} are covered by this "
                    f"benchmark (supported: {sorted(supported)}); set an explicit "
                    "benchmarks.<name>.languages list."
                )
            return covered
        return langs

    def _load_mgsm_agent_tasks(self, cfg: Dict) -> List:
        """MGSM math tasks with the gold numeric answer for exact-match scoring."""
        from latent_coordination.agents.base_agent import AgentTask
        from latent_coordination.eval.correctness import (
            MGSM_SUPPORTED_LANGUAGES,
            load_mgsm_tasks,
        )

        tasks = []
        n = cfg.get("n_samples")
        for lang in self._benchmark_languages(cfg, set(MGSM_SUPPORTED_LANGUAGES)):
            items = load_mgsm_tasks(language=lang, n=n)
            for i, item in enumerate(items):
                tasks.append(AgentTask(
                    task_id=f"mgsm_{lang}_{i}",
                    query=item["question"],
                    target_language=lang,
                    metadata={
                        "benchmark": "mgsm",
                        "gold_answer": float(item["answer"]),
                    },
                ))
            logger.info("Loaded %d MGSM tasks for '%s'.", len(items), lang)
        return tasks

    def _load_afrimgsm_agent_tasks(self, cfg: Dict) -> List:
        """AfriMGSM math tasks (translated GSM8k, African languages) with gold answer."""
        from latent_coordination.agents.base_agent import AgentTask
        from latent_coordination.eval.correctness import (
            AFRIMGSM_SUPPORTED_LANGUAGES,
            load_afrimgsm_tasks,
        )

        tasks = []
        n = cfg.get("n_samples")
        for lang in self._benchmark_languages(cfg, set(AFRIMGSM_SUPPORTED_LANGUAGES)):
            items = load_afrimgsm_tasks(language=lang, n=n)
            for i, item in enumerate(items):
                tasks.append(AgentTask(
                    task_id=f"afrimgsm_{lang}_{i}",
                    query=item["question"],
                    target_language=lang,
                    metadata={
                        "benchmark": "afrimgsm",
                        "gold_answer": float(item["answer"]),
                    },
                ))
            logger.info("Loaded %d AfriMGSM tasks for '%s'.", len(items), lang)
        return tasks

    def _load_belebele_agent_tasks(self, cfg: Dict) -> List:
        """Belebele reading-comprehension MCQA with choices + gold index."""
        from latent_coordination.agents.base_agent import AgentTask
        from latent_coordination.eval.correctness import load_belebele_tasks

        tasks = []
        n = cfg.get("n_samples")
        for lang in self._benchmark_languages(cfg, set(self._BELEBELE_LANG_MAP)):
            items = load_belebele_tasks(language=self._BELEBELE_LANG_MAP[lang], n=n)
            for i, item in enumerate(items):
                options = "\n".join(
                    f"{k + 1}. {c}" for k, c in enumerate(item["choices"])
                )
                prompt_stem = (
                    f"Passage: {item['passage']}\n\nQuestion: {item['question']}\n"
                    f"Options:\n{options}\n\n"
                    "Identify the correct option number (1, 2, 3, or 4). Answer:"
                )
                tasks.append(AgentTask(
                    task_id=f"belebele_{lang}_{i}",
                    query=prompt_stem,
                    target_language=lang,
                    metadata={
                        "benchmark": "belebele",
                        "prompt_stem": prompt_stem,
                        "choices": item["choices"],
                        "correct_idx": item["correct_idx"],
                    },
                ))
            logger.info("Loaded %d Belebele tasks for '%s'.", len(items), lang)
        return tasks

    def _load_sea_vision_agent_tasks(self, cfg: Dict) -> List:
        """SEA-Vision text-QA tasks (reference-scored via containment EM)."""
        from latent_coordination.agents.base_agent import AgentTask
        from shared.data.dataset_loader import DatasetLoader

        loader = DatasetLoader()
        samples = loader.load_sea_vision(
            languages=self._benchmark_languages(cfg),
            max_per_language=cfg.get("n_samples_per_language"),
            local_dir=cfg.get("local_dir"),
        )
        tasks = [
            AgentTask(
                task_id=f"seavision_{s.language}_{i}",
                query=s.text,
                target_language=s.language,
                reference=s.reference_answer,
                metadata={"benchmark": "sea_vision"},
            )
            for i, s in enumerate(samples)
        ]
        logger.info("Loaded %d SEA-Vision tasks.", len(tasks))
        return tasks

    def _load_sea_safeguard_agent_tasks(self, cfg: Dict) -> List:
        """SEA safety tasks scored by SafetyAgent-verdict agreement."""
        from latent_coordination.agents.base_agent import AgentTask
        from shared.data.dataset_loader import DatasetLoader

        loader = DatasetLoader()
        # SAMPLING SEMANTICS: ``n_samples`` here is a TOTAL cap split across the
        # configured languages (200 with th+my -> 100/lang), unlike Belebele's
        # ``n_samples`` which is per-language. ``n_samples_per_language`` makes
        # per-language intent explicit; changing an existing config between the
        # two invalidates comparability with its earlier runs.
        n_per_lang = cfg.get("n_samples_per_language")
        if n_per_lang is not None:
            max_samples = int(n_per_lang) * len(cfg.get("languages") or [])
        else:
            max_samples = cfg.get("n_samples")
        samples = loader.load_sea_safeguardbench(
            languages=cfg.get("languages"),
            max_samples=max_samples,
            repo_id=cfg.get("repo_id"),  # loader raises without one — no default repo exists
        )
        per_lang_counts: Dict[str, int] = {}
        for s in samples:
            per_lang_counts[s.language] = per_lang_counts.get(s.language, 0) + 1
        logger.info(
            "SEA-SafeguardBench sampling: total cap=%s -> per-language n=%s "
            "(report these n, not the config value)", max_samples, per_lang_counts,
        )
        # "n" / "non-toxic": SEA-HELM Safety-Toxicity-Detection labels rows
        # Y (toxic) / N (non-toxic); its Burmese config labels in Burmese script
        # instead — သန့်ရှင်း ("clean") vs မုန်းတီးမှု ("hate") — verified against
        # the real dataset rows 2026-07-03.
        safe_labels = {
            "safe", "benign", "harmless", "no", "n", "non-toxic", "nontoxic",
            "0", "false", "သန့်ရှင်း",
        }
        tasks = []
        for i, s in enumerate(samples):
            expected = "safe" if str(s.reference_answer).strip().lower() in safe_labels else "unsafe"
            tasks.append(AgentTask(
                task_id=f"safeguard_{s.language}_{i}",
                query=s.text,
                target_language=s.language,
                metadata={
                    "benchmark": "sea_safeguardbench",
                    "expected_verdict": expected,
                },
            ))
        logger.info("Loaded %d SEA-SafeguardBench tasks.", len(tasks))
        return tasks

    def _load_flores_tasks(self) -> List:
        """Load real translation tasks from FLORES+ via Hugging Face datasets.

        Returns
        -------
        List[AgentTask]
            Tasks sourced from the FLORES+ devtest split for the tracked
            SEA/low-resource language pairs.
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

        # The English source split is shared by every language pair — load it once,
        # not once per language (6 redundant loads of the same 1012-row split).
        en_ds = None
        for flores_code, iso_code in lang_pairs:
            try:
                if en_ds is None:
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
                            # NOT context=tgt_text: context is fed verbatim into every
                            # specialized agent's prompt (see AgentTask.context's
                            # docstring) -- putting the gold translation there leaked
                            # the answer into the first agent's own input on every
                            # comm-mode. The gold translation belongs in `reference`,
                            # read only by _compute_translation_quality for scoring.
                            reference=tgt_text,
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
