"""Reference-based correctness scoring for multi-agent benchmark evaluation.

Replaces the completeness proxy (non-empty output heuristic) with real accuracy
for three benchmark workloads:

  MGSM        — multi-step math (exact-match on final numeric answer)
  MMLU-ProX   — 10-choice multilingual QA (teacher-forced log-likelihood)
  Belebele    — reading comprehension (4-choice log-likelihood)

Usage
-----
    scorer = CorrectnessScorer(model, tokenizer, device="cuda:0")
    result = scorer.score_mgsm(response_text, gold_answer="42")
    result = scorer.score_multiple_choice(response_text, choices, gold_idx=2)

The module is designed to be called after agents produce output_text so it
integrates with the existing AgentResponse dataclass without modifying agents.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CorrectnessResult:
    """Score for a single (prediction, reference) pair."""
    benchmark: str          # "mgsm" | "mmlu_prox" | "belebele"
    is_correct: bool
    predicted: Any          # extracted answer (number, choice index, or text)
    gold: Any               # reference answer
    score: float            # 1.0 correct, 0.0 incorrect
    details: Dict = field(default_factory=dict)


@dataclass
class BenchmarkCorrectnessReport:
    """Aggregated correctness results across a benchmark split."""
    benchmark: str
    n_total: int
    n_correct: int
    accuracy: float
    results: List[CorrectnessResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "benchmark": self.benchmark,
            "n_total": self.n_total,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
        }


# ---------------------------------------------------------------------------
# MGSM exact-match helpers
# ---------------------------------------------------------------------------

# Patterns to extract the final numeric answer from a chain-of-thought response.
# Priority: explicit "The answer is X" > last number in the text.
_MGSM_ANSWER_PATTERNS = [
    re.compile(r"(?:the\s+)?answer\s+is[:\s]+(-?[\d,\.]+)", re.IGNORECASE),
    re.compile(r"(?:答案|答え|답|câu trả lời|उत्तर|الإجابة)[^:：]*[：:]\s*(-?[\d,\.]+)", re.UNICODE),
    re.compile(r"=\s*(-?[\d,\.]+)\s*$", re.MULTILINE),
]
_LAST_NUMBER_PATTERN = re.compile(r"(-?[\d,\.]+)(?:\s*$|\s+[^\d])", re.MULTILINE)


def extract_mgsm_answer(text: str) -> Optional[float]:
    """Extract the final numeric answer from a MGSM chain-of-thought response.

    Returns the number as a float, or None if no number found.
    Strips commas used as thousand separators before parsing.
    """
    for pat in _MGSM_ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                continue
    # Fallback: last number appearing in the text.
    numbers = _LAST_NUMBER_PATTERN.findall(text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def score_mgsm(predicted_text: str, gold_answer: float, tolerance: float = 1e-3) -> CorrectnessResult:
    """Exact-match score for MGSM: correct iff extracted number == gold.

    Parameters
    ----------
    predicted_text : model's free-form generation (may include chain-of-thought)
    gold_answer    : the reference numeric answer
    tolerance      : absolute tolerance for float comparison (handles rounding)
    """
    pred = extract_mgsm_answer(predicted_text)
    is_correct = pred is not None and abs(pred - gold_answer) <= tolerance
    return CorrectnessResult(
        benchmark="mgsm",
        is_correct=is_correct,
        predicted=pred,
        gold=gold_answer,
        score=1.0 if is_correct else 0.0,
        details={"raw_text_snippet": predicted_text[:200]},
    )


# ---------------------------------------------------------------------------
# Multiple-choice log-likelihood scoring (MMLU-ProX / Belebele)
# ---------------------------------------------------------------------------

def _log_likelihood(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    continuation: str,
    device: str = "cpu",
) -> float:
    """Teacher-forced log-likelihood of ``continuation`` conditioned on ``prompt``."""
    full = prompt + continuation
    enc_full = tokenizer(full, return_tensors="pt", truncation=True, max_length=1024)
    enc_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    n_prompt_tokens = enc_prompt["input_ids"].shape[1]

    input_ids = enc_full["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        # Manually compute per-token NLL for continuation tokens only.
        logits = outputs.logits  # (1, T, V)
    shift_logits = logits[0, :-1]  # (T-1, V)
    shift_labels = input_ids[0, 1:]  # (T-1,)
    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    token_lls = log_probs[range(len(shift_labels)), shift_labels]
    # Sum over continuation tokens only.
    cont_lls = token_lls[n_prompt_tokens - 1:]
    return float(cont_lls.sum().item())


class CorrectnessScorer:
    """Reference-based accuracy scorer for MGSM, MMLU-ProX, and Belebele.

    Parameters
    ----------
    model, tokenizer
        A loaded causal LM and tokenizer (eval mode). Required for log-likelihood
        scoring (MMLU-ProX, Belebele). Not required for MGSM exact-match.
    device
        Device string for forward passes.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        tokenizer=None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # ------------------------------------------------------------------
    # MGSM
    # ------------------------------------------------------------------

    def score_mgsm(self, predicted_text: str, gold_answer: float) -> CorrectnessResult:
        """Score a single MGSM example via exact-match on the extracted number."""
        return score_mgsm(predicted_text, gold_answer)

    def score_mgsm_batch(
        self,
        predictions: Sequence[str],
        gold_answers: Sequence[float],
    ) -> BenchmarkCorrectnessReport:
        """Score a full MGSM split and return a report."""
        results = [
            score_mgsm(pred, gold)
            for pred, gold in zip(predictions, gold_answers)
        ]
        n_correct = sum(r.is_correct for r in results)
        return BenchmarkCorrectnessReport(
            benchmark="mgsm",
            n_total=len(results),
            n_correct=n_correct,
            accuracy=n_correct / max(len(results), 1),
            results=results,
        )

    # ------------------------------------------------------------------
    # Multiple-choice log-likelihood (MMLU-ProX and Belebele)
    # ------------------------------------------------------------------

    def score_multiple_choice(
        self,
        prompt: str,
        choices: Sequence[str],
        gold_idx: int,
        benchmark: str = "mmlu_prox",
    ) -> CorrectnessResult:
        """Score a multiple-choice question by teacher-forced log-likelihood.

        The choice with the highest log-likelihood conditioned on the prompt is
        selected as the prediction. ``gold_idx`` is the 0-based index of the
        correct choice.

        Parameters
        ----------
        prompt    : the question stem (e.g. "Question: ... Answer:")
        choices   : list of choice strings (e.g. ["A. Paris", "B. London", ...])
        gold_idx  : 0-based index of the correct choice
        benchmark : "mmlu_prox" (10 choices) or "belebele" (4 choices)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "CorrectnessScorer requires a model and tokenizer for multiple-choice scoring."
            )
        self.model.eval()
        lls = [
            _log_likelihood(self.model, self.tokenizer, prompt, choice, self.device)
            for choice in choices
        ]
        pred_idx = int(max(range(len(lls)), key=lambda i: lls[i]))
        is_correct = pred_idx == gold_idx
        return CorrectnessResult(
            benchmark=benchmark,
            is_correct=is_correct,
            predicted=pred_idx,
            gold=gold_idx,
            score=1.0 if is_correct else 0.0,
            details={"log_likelihoods": lls, "choices": list(choices)},
        )

    def score_multiple_choice_batch(
        self,
        prompts: Sequence[str],
        choices_list: Sequence[Sequence[str]],
        gold_indices: Sequence[int],
        benchmark: str = "mmlu_prox",
    ) -> BenchmarkCorrectnessReport:
        """Score a full MMLU-ProX or Belebele split and return a report."""
        results = []
        for prompt, choices, gold_idx in zip(prompts, choices_list, gold_indices):
            try:
                r = self.score_multiple_choice(prompt, choices, gold_idx, benchmark)
            except Exception as exc:
                logger.warning("Scoring failed for one example: %s", exc)
                r = CorrectnessResult(
                    benchmark=benchmark,
                    is_correct=False,
                    predicted=None,
                    gold=gold_idx,
                    score=0.0,
                    details={"error": str(exc)},
                )
            results.append(r)
        n_correct = sum(r.is_correct for r in results)
        return BenchmarkCorrectnessReport(
            benchmark=benchmark,
            n_total=len(results),
            n_correct=n_correct,
            accuracy=n_correct / max(len(results), 1),
            results=results,
        )

    # ------------------------------------------------------------------
    # Aggregate from AgentResponse lists (pipeline integration)
    # ------------------------------------------------------------------

    def score_agent_responses_mgsm(
        self,
        responses: Sequence[Any],
        gold_answers: Sequence[float],
    ) -> BenchmarkCorrectnessReport:
        """Score a list of AgentResponse objects on MGSM.

        Extracts ``output_text`` from each response. Responses must be
        substantive answers (run through
        :func:`~latent_coordination.eval.scoring.select_answer` first).
        """
        predictions = [
            getattr(r, "output_text", "") or "" for r in responses
        ]
        return self.score_mgsm_batch(predictions, gold_answers)


# ---------------------------------------------------------------------------
# Dataset loaders (thin wrappers around HF datasets for pipeline use)
# ---------------------------------------------------------------------------

# The upstream juletxara/mgsm release only ships these 11 configs -- it has no
# Lao/Khmer/Burmese/Amharic data at all (verified via
# datasets.get_dataset_config_names("juletxara/mgsm")). Validate up front so
# callers get one clear, actionable error instead of the underlying library's
# opaque "BuilderConfig 'km' not found" trace.
MGSM_SUPPORTED_LANGUAGES = frozenset({"bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"})


def load_mgsm_tasks(language: str = "en", split: str = "test", n: Optional[int] = None):
    """Load MGSM tasks from the Hugging Face datasets hub.

    Returns a list of dicts with keys: ``question``, ``answer`` (int).

    Raises:
        ValueError: if ``language`` is outside MGSM_SUPPORTED_LANGUAGES. MGSM has no
            Lao/Khmer/Burmese/Amharic data upstream; use Belebele/FLORES+/SEA-Vision
            for those languages instead of MGSM.
    """
    if language not in MGSM_SUPPORTED_LANGUAGES:
        raise ValueError(
            f"MGSM does not have data for language '{language}'. juletxara/mgsm only "
            f"covers {sorted(MGSM_SUPPORTED_LANGUAGES)}. This is an upstream dataset "
            "limitation (no Lao/Khmer/Burmese/Amharic release exists), not a config "
            "error -- use Belebele, FLORES+, or SEA-Vision for those languages instead."
        )
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError("datasets library required: pip install datasets") from exc
    ds = load_dataset("juletxara/mgsm", language, split=split)
    items = [{"question": row["question"], "answer": int(row["answer_number"])} for row in ds]
    return items[:n] if n is not None else items


# McGill-NLP/mgsm-pro keys language by HF *split* name (not config -- the two configs,
# "ic" and "symbolic", are instantiation categories). Coverage does NOT match base
# MGSM: it has Amharic/Igbo/Twi/Yoruba but not Bengali/German/Russian/Telugu/Thai.
MGSM_PRO_SUPPORTED_LANGUAGES = frozenset({"am", "en", "fr", "ig", "ja", "sw", "tw", "yo", "zh"})
_MGSM_PRO_LANG_TO_SPLIT = {
    "am": "amharic", "zh": "chinese", "en": "english", "fr": "french",
    "ig": "igbo", "ja": "japanese", "sw": "swahili", "tw": "twi", "yo": "yoruba",
}


def load_mgsm_pro_tasks(
    language: str = "en", config: str = "symbolic", n: Optional[int] = None,
):
    """Load MGSM-Pro tasks (memorization-resistant symbolic/name/context instantiations).

    Same return schema as :func:`load_mgsm_tasks` ({"question", "answer"}) so it's a
    drop-in benchmark option for the same MGSM-shaped baseline runners.

    Raises:
        ValueError: if ``language`` is outside MGSM_PRO_SUPPORTED_LANGUAGES.
    """
    if language not in MGSM_PRO_SUPPORTED_LANGUAGES:
        raise ValueError(
            f"MGSM-Pro does not have data for language '{language}'. It only covers "
            f"{sorted(MGSM_PRO_SUPPORTED_LANGUAGES)}."
        )
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError("datasets library required: pip install datasets") from exc
    split = _MGSM_PRO_LANG_TO_SPLIT[language]
    ds = load_dataset("McGill-NLP/mgsm-pro", config, split=split)
    items = [{"question": row["question"], "answer": int(row["answer"])} for row in ds]
    return items[:n] if n is not None else items


def load_belebele_tasks(language: str = "eng_Latn", split: str = "test", n: Optional[int] = None):
    """Load Belebele reading-comprehension tasks from HF datasets.

    Returns a list of dicts with keys: ``passage``, ``question``, ``choices``
    (list of 4 strings), ``correct_idx`` (0-based int).
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError("datasets library required: pip install datasets") from exc
    ds = load_dataset("facebook/belebele", language, split=split)
    items = []
    for row in ds:
        choices = [
            row["mc_answer1"], row["mc_answer2"],
            row["mc_answer3"], row["mc_answer4"],
        ]
        correct_idx = int(row["correct_answer_num"]) - 1  # 1-based → 0-based
        items.append({
            "passage": row["flores_passage"],
            "question": row["question"],
            "choices": choices,
            "correct_idx": correct_idx,
        })
    return items[:n] if n is not None else items
