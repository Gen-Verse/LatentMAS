"""
Specialised agent implementations for the Latent Coordination multi-agent system.

Provides three concrete BaseAgent subclasses:

    TranslationAgent  : Steers hidden states toward a target language using
                        representation engineering (LatentSteerer / TRIAD-TS).
    ReasoningAgent    : Amplifies reasoning subspace via SVD projection;
                        handles chain-of-thought delimiters.
    SafetyAgent       : Evaluates outputs against safety criteria and returns
                        a structured SafetyVerdict.
"""


import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from latent_coordination.agents.base_agent import AgentConfig, AgentResponse, AgentTask, BaseAgent

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
# Safety verdict dataclass
# ---------------------------------------------------------------------------

@dataclass
class SafetyVerdict:
    """Result of a SafetyAgent evaluation.

    Attributes:
        is_safe: True if the content passes all safety criteria.
        risk_score: Scalar risk estimate in [0, 1]; higher = riskier.
        risk_categories: List of detected risk category labels.
        explanation: Human-readable explanation of the verdict.
        raw_response: Raw model output that led to this verdict.
    """

    is_safe: bool
    risk_score: float
    risk_categories: List[str]
    explanation: str
    raw_response: str = ""


# ---------------------------------------------------------------------------
# Soft imports for latent_coordination (gracefully degrade if not available)
# ---------------------------------------------------------------------------

def _try_import_steerer():
    """Attempt to import LatentSteerer from the sibling Mechanistic Disentanglement package."""
    try:
        from latent_coordination.steering.latent_steerer import LatentSteerer  # type: ignore
        return LatentSteerer
    except ImportError:
        logger.warning(
            "latent_coordination.steering.LatentSteerer not found. "
            "TranslationAgent will fall back to standard generation."
        )
        return None


def _try_import_svd_decomposer():
    """Attempt to import SVDSubspaceDecomposer from the sibling Mechanistic Disentanglement package."""
    try:
        from latent_coordination.geometry.svd_decomposer import SVDSubspaceDecomposer  # type: ignore
        return SVDSubspaceDecomposer
    except ImportError:
        logger.warning(
            "latent_coordination.geometry.SVDSubspaceDecomposer not found. "
            "ReasoningAgent will fall back to standard generation."
        )
        return None


# ---------------------------------------------------------------------------
# Translation Agent
# ---------------------------------------------------------------------------

class TranslationAgent(BaseAgent):
    """Specialised agent for low-resource language translation.

    Steers the model's hidden representations toward a target language
    using LatentSteerer from the Mechanistic Disentanglement representation-engineering package.
    Falls back to standard prompted generation if the steerer is unavailable.

    Key features:
        - SFR-based quality gating: outputs below ``sfr_threshold`` are
          regenerated with a fallback prompt.
        - Latent injection: accepts ``task.latent_state`` from the orchestrator
          and injects it at the configured layer before generation.
        - Extracts final hidden states for downstream agents.

    Args:
        config: AgentConfig with role='translation'.
        steer_layer: Transformer layer index to apply steering at.
        sfr_threshold: Minimum acceptable SFR-like confidence score.
        target_language_map: Mapping from ISO codes to full language names
            used in prompts.
    """

    def __init__(
        self,
        config: AgentConfig,
        steer_layer: int = -1,
        sfr_threshold: float = 0.3,
        target_language_map: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(config)
        self.steer_layer = steer_layer
        self.sfr_threshold = sfr_threshold
        self.target_language_map: Dict[str, str] = target_language_map or {
            "id": "Indonesian",
            "ms": "Malay",
            "th": "Thai",
            "vi": "Vietnamese",
            "tl": "Filipino",
            "jv": "Javanese",
            "su": "Sundanese",
            "km": "Khmer",
            "lo": "Lao",
            "my": "Burmese",
            "en": "English",
            "zh": "Chinese",
        }
        # Lazily load LatentSteerer
        self._LatentSteerer = _try_import_steerer()
        self._steerer = None
        logger.info(
            "TranslationAgent '%s' ready (sfr_threshold=%.2f)",
            config.agent_id,
            sfr_threshold,
        )

    def _get_language_name(self, code: Optional[str]) -> str:
        if code is None:
            return "the target language"
        return self.target_language_map.get(code.lower(), code)

    def _build_translation_prompt(self, task: AgentTask) -> str:
        lang_name = self._get_language_name(task.target_language)
        context_block = f"\nContext: {task.context}" if task.context else ""
        return (
            f"You are an expert translator. Translate the following text into "
            f"{lang_name}. Produce only the translation with no commentary."
            f"{context_block}\n\nText: {task.query}\n\nTranslation:"
        )

    def _estimate_sfr(self, generated_text: str, source_text: str) -> float:
        """Lightweight SFR-like quality estimate.

        Uses length ratio and non-ASCII character density as a proxy for
        translation quality when a full SFR model is unavailable.

        Args:
            generated_text: Candidate translation.
            source_text: Original source text.

        Returns:
            Float quality score in [0, 1].
        """
        if not generated_text.strip():
            return 0.0
        # Length ratio heuristic (good translations are roughly similar in length)
        len_ratio = min(len(generated_text), len(source_text)) / max(
            max(len(generated_text), len(source_text)), 1
        )
        # Non-ASCII density (good for non-Latin scripts)
        non_ascii = sum(1 for c in generated_text if ord(c) > 127)
        script_score = min(non_ascii / max(len(generated_text), 1), 1.0)
        # Penalise empty / repetitive output
        unique_ratio = len(set(generated_text.split())) / max(
            len(generated_text.split()), 1
        )
        sfr = 0.4 * len_ratio + 0.3 * script_score + 0.3 * unique_ratio
        return float(min(sfr, 1.0))

    def process(self, task: AgentTask) -> AgentResponse:
        """Translate the task query into the target language.

        Steps:
            1. If a latent_state is provided, inject it before generation.
            2. Generate a translation using the language-steered prompt.
            3. Compute SFR quality estimate.
            4. If SFR < threshold, re-generate with a simplified fallback prompt.
            5. Extract final hidden states for downstream agents.

        Args:
            task: AgentTask with query and target_language.

        Returns:
            AgentResponse with translated text and hidden states.
        """
        t0 = self._start_timer()
        self._ensure_model_loaded()
        prompt = self._build_translation_prompt(task)

        # --- Attempt latent-steered generation ---
        generated_text = ""
        try:
            if task.latent_state is not None:
                # Inject the orchestrator-provided latent state at the target layer
                generated_text = self.inject_latent_and_generate(
                    task.latent_state,
                    prompt,
                    injection_layer=self.steer_layer,
                    max_new_tokens=self.config.max_new_tokens,
                )
            else:
                # Standard prompted generation
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
                with torch.no_grad():
                    out_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=0.3,
                        do_sample=True,
                    )
                generated_ids = out_ids[0, inputs["input_ids"].shape[1]:]
                generated_text = self._tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
        except Exception as exc:
            logger.error("TranslationAgent generation error: %s", exc, exc_info=True)
            generated_text = f"[Translation failed: {exc}]"

        # --- Quality gating ---
        sfr_score = self._estimate_sfr(generated_text, task.query)
        if sfr_score < self.sfr_threshold:
            logger.warning(
                "SFR=%.3f < threshold=%.3f; re-generating with fallback prompt.",
                sfr_score,
                self.sfr_threshold,
            )
            fallback_prompt = (
                f"Translate to {self._get_language_name(task.target_language)}: "
                f"{task.query}"
            )
            try:
                inputs = self._tokenizer(fallback_prompt, return_tensors="pt").to(self._device)
                with torch.no_grad():
                    out_ids = self._model.generate(
                        **inputs, max_new_tokens=self.config.max_new_tokens
                    )
                generated_ids = out_ids[0, inputs["input_ids"].shape[1]:]
                generated_text = self._tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                sfr_score = self._estimate_sfr(generated_text, task.query)
            except Exception as exc:
                logger.error("Fallback generation failed: %s", exc)

        # --- Extract hidden states for downstream ---
        hidden_states_dict = self.extract_hidden_states(
            generated_text[:512] if generated_text else prompt[:512],
            layer_ids=[-1],
        )
        latent = hidden_states_dict.get(-1)  # (1, L, D)

        elapsed_ms = self._stop_timer(t0)
        return self._build_response(
            task,
            output_text=generated_text.strip(),
            latent_state=latent,
            elapsed_ms=elapsed_ms,
            confidence=sfr_score,
            extra_meta={
                "target_language": task.target_language,
                "sfr_score": sfr_score,
                "used_latent_injection": task.latent_state is not None,
            },
        )


# ---------------------------------------------------------------------------
# Reasoning Agent
# ---------------------------------------------------------------------------

class ReasoningAgent(BaseAgent):
    """Specialised agent for multi-step reasoning tasks.

    Extracts and amplifies the reasoning subspace from the model's hidden
    representations using SVD-based subspace projection (when available).
    Supports explicit chain-of-thought via ``<think>`` / ``</think>``
    delimiters and captures intermediate reasoning states.

    Args:
        config: AgentConfig with role='reasoning'.
        reasoning_layer: Layer to extract and project the reasoning subspace.
        n_reasoning_components: Number of top SVD components to keep for the
            reasoning subspace projection.
        cot_delimiter_start: Opening tag for chain-of-thought.
        cot_delimiter_end: Closing tag for chain-of-thought.
    """

    _COT_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    def __init__(
        self,
        config: AgentConfig,
        reasoning_layer: int = -2,
        n_reasoning_components: int = 16,
        cot_delimiter_start: str = "<think>",
        cot_delimiter_end: str = "</think>",
    ) -> None:
        super().__init__(config)
        self.reasoning_layer = reasoning_layer
        self.n_reasoning_components = n_reasoning_components
        self.cot_delimiter_start = cot_delimiter_start
        self.cot_delimiter_end = cot_delimiter_end
        self._SVDDecomposer = _try_import_svd_decomposer()
        self._decomposer = None
        logger.info("ReasoningAgent '%s' ready", config.agent_id)

    def _build_cot_prompt(self, task: AgentTask) -> str:
        context_block = f"\nContext: {task.context}" if task.context else ""
        return (
            f"You are an expert reasoning engine. Think step by step before answering."
            f"{context_block}\n\nQuestion: {task.query}\n\n"
            f"{self.cot_delimiter_start}"
        )

    def _extract_cot_segments(self, text: str) -> Tuple[List[str], str]:
        """Parse chain-of-thought thinking and final answer from text.

        Args:
            text: Raw model output potentially containing <think>...</think> blocks.

        Returns:
            Tuple of (list of thinking steps, final answer text).
        """
        thinking_steps: List[str] = []
        matches = self._COT_THINK_PATTERN.findall(text)
        for m in matches:
            steps = [s.strip() for s in m.split("\n") if s.strip()]
            thinking_steps.extend(steps)

        # Final answer is text after the last </think>
        answer = self._COT_THINK_PATTERN.sub("", text).strip()
        return thinking_steps, answer

    def _amplify_reasoning_subspace(self, hidden_states: Tensor) -> Tensor:
        """Project hidden states onto the top reasoning SVD components.

        If SVDSubspaceDecomposer is unavailable, returns the input unchanged.

        Args:
            hidden_states: Tensor of shape (1, seq_len, hidden_dim).

        Returns:
            Amplified tensor of same shape.
        """
        if self._SVDDecomposer is None or self._decomposer is None:
            return hidden_states
        try:
            return self._decomposer.project_to_reasoning(hidden_states)
        except Exception as exc:
            logger.warning("SVD projection failed: %s", exc)
            return hidden_states

    def process(self, task: AgentTask) -> AgentResponse:
        """Process a reasoning task with chain-of-thought and subspace amplification.

        Steps:
            1. Build a CoT prompt with <think> delimiter.
            2. Generate output (with injected latent if provided).
            3. Parse <think> blocks and extract intermediate reasoning states.
            4. Amplify the reasoning subspace in the extracted hidden states.
            5. Return the final answer with reasoning metadata.

        Args:
            task: AgentTask with the reasoning query.

        Returns:
            AgentResponse with structured reasoning metadata.
        """
        t0 = self._start_timer()
        self._ensure_model_loaded()
        prompt = self._build_cot_prompt(task)

        generated_text = ""
        try:
            if task.latent_state is not None:
                # Inject provided latent state from orchestrator
                generated_text = self.inject_latent_and_generate(
                    task.latent_state,
                    prompt,
                    injection_layer=self.reasoning_layer,
                    max_new_tokens=self.config.max_new_tokens,
                )
            else:
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
                with torch.no_grad():
                    out_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=0.6,
                        do_sample=True,
                    )
                gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
                generated_text = self._tokenizer.decode(
                    gen_ids, skip_special_tokens=True
                )
        except Exception as exc:
            logger.error("ReasoningAgent generation error: %s", exc, exc_info=True)
            generated_text = f"[Reasoning failed: {exc}]"

        # --- Parse CoT structure ---
        full_text = self.cot_delimiter_start + generated_text
        thinking_steps, final_answer = self._extract_cot_segments(full_text)

        # --- Extract and amplify reasoning hidden states ---
        hs_dict = self.extract_hidden_states(
            (final_answer or generated_text)[:512], layer_ids=[self.reasoning_layer]
        )
        raw_states = hs_dict.get(self.reasoning_layer)
        amplified_states = self._amplify_reasoning_subspace(raw_states) if raw_states is not None else raw_states

        elapsed_ms = self._stop_timer(t0)

        # Confidence heuristic: longer reasoning = more confident
        n_steps = len(thinking_steps)
        confidence = min(0.5 + 0.05 * n_steps, 0.99)

        return self._build_response(
            task,
            output_text=final_answer or generated_text,
            latent_state=amplified_states,
            elapsed_ms=elapsed_ms,
            confidence=confidence,
            extra_meta={
                "thinking_steps": thinking_steps,
                "n_reasoning_steps": n_steps,
                "full_cot": generated_text,
            },
        )


# ---------------------------------------------------------------------------
# Safety Agent
# ---------------------------------------------------------------------------

# Risk category definitions (prompt-based, no external API)
_RISK_CATEGORIES = [
    "violence",
    "hate_speech",
    "sexual_content",
    "self_harm",
    "misinformation",
    "illegal_activities",
    "personal_data_exposure",
]

_SAFETY_SYSTEM_PROMPT = """You are a content safety classifier. Given the following text, determine if it contains harmful content.

For each category below, answer YES or NO:
- violence: does the text describe or encourage physical harm?
- hate_speech: does the text target or demean people based on identity?
- sexual_content: does the text contain explicit sexual material?
- self_harm: does the text encourage self-harm or suicide?
- misinformation: does the text contain verifiably false harmful claims?
- illegal_activities: does the text instruct on illegal acts?
- personal_data_exposure: does the text expose private personal information?

Then give an overall verdict: SAFE or UNSAFE.
Provide a brief explanation.

Text to evaluate:
---
{text}
---

Response format:
violence: YES/NO
hate_speech: YES/NO
sexual_content: YES/NO
self_harm: YES/NO
misinformation: YES/NO
illegal_activities: YES/NO
personal_data_exposure: YES/NO
VERDICT: SAFE/UNSAFE
Explanation: <1-2 sentences>
"""

_YES_PATTERN = re.compile(r"(\w+):\s*(YES|NO)", re.IGNORECASE)
_VERDICT_PATTERN = re.compile(r"VERDICT:\s*(SAFE|UNSAFE)", re.IGNORECASE)
_EXPLANATION_PATTERN = re.compile(r"Explanation:\s*(.+)", re.IGNORECASE | re.DOTALL)


class SafetyAgent(BaseAgent):
    """Specialised agent for content safety evaluation.

    Uses a prompt-based classification approach (no external API required).
    Parses structured YES/NO responses per risk category and computes an
    overall risk score.

    Args:
        config: AgentConfig with role='safety'.
        risk_threshold: Minimum risk score to flag content as unsafe
            (overrides the VERDICT field if model output is ambiguous).
        max_eval_length: Maximum character length of text to evaluate.
    """

    def __init__(
        self,
        config: AgentConfig,
        risk_threshold: float = 0.3,
        max_eval_length: int = 1024,
    ) -> None:
        super().__init__(config)
        self.risk_threshold = risk_threshold
        self.max_eval_length = max_eval_length
        logger.info(
            "SafetyAgent '%s' ready (risk_threshold=%.2f)",
            config.agent_id,
            risk_threshold,
        )

    def _build_safety_prompt(self, text: str) -> str:
        truncated = text[: self.max_eval_length]
        return _SAFETY_SYSTEM_PROMPT.format(text=truncated)

    def _parse_safety_response(self, response: str, text: str) -> SafetyVerdict:
        """Parse the model's safety evaluation response.

        Args:
            response: Raw model output from the safety prompt.
            text: Original text that was evaluated (for fallback heuristics).

        Returns:
            Structured :class:`SafetyVerdict`.
        """
        detected_categories: List[str] = []
        for match in _YES_PATTERN.finditer(response):
            category = match.group(1).lower()
            answer = match.group(2).upper()
            if answer == "YES" and category in _RISK_CATEGORIES:
                detected_categories.append(category)

        # Compute risk score as fraction of flagged categories
        risk_score = len(detected_categories) / len(_RISK_CATEGORIES)

        # Parse verdict
        verdict_match = _VERDICT_PATTERN.search(response)
        if verdict_match:
            model_says_safe = verdict_match.group(1).upper() == "SAFE"
        else:
            model_says_safe = risk_score < self.risk_threshold

        is_safe = model_says_safe and (risk_score < self.risk_threshold)

        # Parse explanation
        exp_match = _EXPLANATION_PATTERN.search(response)
        explanation = exp_match.group(1).strip() if exp_match else "No explanation provided."

        return SafetyVerdict(
            is_safe=is_safe,
            risk_score=float(risk_score),
            risk_categories=detected_categories,
            explanation=explanation,
            raw_response=response,
        )

    def evaluate(self, text: str) -> SafetyVerdict:
        """Evaluate a text string for safety (public API).

        Args:
            text: Text content to evaluate.

        Returns:
            :class:`SafetyVerdict` with verdict and category breakdown.
        """
        # No fabricated verdicts: safety evaluation must come from the real model.
        self._ensure_model_loaded()

        prompt = self._build_safety_prompt(text)
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self._device)
        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs, max_new_tokens=256, do_sample=False
            )
        gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        return self._parse_safety_response(response, text)

    def process(self, task: AgentTask) -> AgentResponse:
        """Process a safety evaluation task.

        Evaluates the task query (and context if available) for safety.
        Also evaluates any ``task.metadata['text_to_evaluate']`` if provided
        (for post-processing evaluation of another agent's output).

        Args:
            task: AgentTask; query or metadata['text_to_evaluate'] is evaluated.

        Returns:
            AgentResponse with ``metadata['safety_verdict']`` containing the
            :class:`SafetyVerdict` as a dict.
        """
        t0 = self._start_timer()

        # Decide what text to evaluate
        text_to_eval = task.metadata.get("text_to_evaluate", task.query)
        if task.context:
            text_to_eval = f"{task.context}\n{text_to_eval}"

        verdict = self.evaluate(text_to_eval)

        output_text = (
            f"[SAFE]" if verdict.is_safe else f"[UNSAFE: {', '.join(verdict.risk_categories)}]"
        )
        output_text += f" Risk={verdict.risk_score:.2f}. {verdict.explanation}"

        # Extract hidden states from the evaluated text for downstream use
        try:
            self._ensure_model_loaded()
            hs_dict = self.extract_hidden_states(text_to_eval[:256], layer_ids=[-1])
            latent = hs_dict.get(-1)
        except Exception:
            latent = None

        elapsed_ms = self._stop_timer(t0)

        return self._build_response(
            task,
            output_text=output_text,
            latent_state=latent,
            elapsed_ms=elapsed_ms,
            confidence=1.0 - verdict.risk_score,
            extra_meta={
                "safety_verdict": {
                    "is_safe": verdict.is_safe,
                    "risk_score": verdict.risk_score,
                    "risk_categories": verdict.risk_categories,
                    "explanation": verdict.explanation,
                },
            },
        )
