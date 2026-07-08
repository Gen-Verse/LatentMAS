"""
Prompt-level baseline interventions for Surgical MRRE ablations.

These classes implement the same ``.apply()`` context-manager interface expected
by :class:`~mrre_drift.eval.ifl.IFLValidator` so they can be passed as the
``intervention`` argument to :meth:`~mrre_drift.eval.ifl.IFLValidator.evaluate`.

Unlike representation-engineering baselines, these act **only** on the prompt
text — no hidden-state modification is performed.

Available baselines
-------------------
SystemPromptBaseline
    Prepends "Please respond in <language>." (or a custom template) to every
    prompt before generation.

FewShotBaseline
    Prepends ``n_shots`` in-language exemplar strings before each prompt.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, Generator, List, Optional

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


# ISO-639-1 → display name used in system prompts.
LANG_NAMES: Dict[str, str] = {
    "th": "Thai",
    "my": "Burmese",
    "km": "Khmer",
    "lo": "Lao",
    "am": "Amharic",
    "sw": "Swahili",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "si": "Sinhala",
    "bo": "Tibetan",
    "ka": "Georgian",
    "hy": "Armenian",
    "ar": "Arabic",
    "he": "Hebrew",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
}


class SystemPromptBaseline:
    """Prepend a language-forcing system instruction to every prompt.

    Parameters
    ----------
    validator
        The :class:`~mrre_drift.eval.ifl.IFLValidator` whose
        ``_prompt_transform`` this baseline will set inside its context.
    template
        Python format string with ``{lang_name}`` placeholder.
    lang_names
        Override map from ISO code to display name; defaults to
        :data:`LANG_NAMES`.
    """

    def __init__(
        self,
        validator,
        template: str = "Please respond in {lang_name}.\n\n",
        lang_names: Optional[Dict[str, str]] = None,
    ) -> None:
        self._validator = validator
        self._template = template
        self._lang_names = lang_names if lang_names is not None else LANG_NAMES

    def _transform(self, lang: str, prompt: str) -> str:
        lang_name = self._lang_names.get(lang, lang)
        prefix = self._template.format(lang_name=lang_name)
        return prefix + prompt

    @contextmanager
    def apply(self) -> Generator["SystemPromptBaseline", None, None]:
        """Install the prompt transform; remove it on exit."""
        prev = self._validator._prompt_transform
        self._validator._prompt_transform = self._transform
        try:
            yield self
        finally:
            self._validator._prompt_transform = prev


class FewShotBaseline:
    """Prepend in-language exemplar strings before each prompt.

    The exemplars act as implicit language-forcing without any hidden-state
    steering — a strong prompt-engineering baseline.

    Parameters
    ----------
    validator
        The :class:`~mrre_drift.eval.ifl.IFLValidator` to instrument.
    exemplars_by_language
        Mapping ``lang -> list of exemplar strings`` (target-language text,
        e.g. FLORES+ sentences or hand-crafted responses).
    n_shots
        Number of exemplars to prepend (taken from the front of each list).
    separator
        String inserted between consecutive exemplars and before the prompt.
    """

    def __init__(
        self,
        validator,
        exemplars_by_language: Dict[str, List[str]],
        n_shots: int = 3,
        separator: str = "\n\n",
    ) -> None:
        self._validator = validator
        self._exemplars = exemplars_by_language
        self._n_shots = n_shots
        self._separator = separator

    def _transform(self, lang: str, prompt: str) -> str:
        shots = self._exemplars.get(lang, [])[: self._n_shots]
        if not shots:
            return prompt
        return self._separator.join(shots) + self._separator + prompt

    @contextmanager
    def apply(self) -> Generator["FewShotBaseline", None, None]:
        """Install the prompt transform; remove it on exit."""
        prev = self._validator._prompt_transform
        self._validator._prompt_transform = self._transform
        try:
            yield self
        finally:
            self._validator._prompt_transform = prev
