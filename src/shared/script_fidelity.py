"""
Script Fidelity Rate (SFR) Evaluator.

SFR measures the fraction of generated tokens that belong to the expected
Unicode script of the target language. This is the primary metric for
assessing whether activation steering successfully induces language-surface
features (i.e. forces the model to generate in the correct script).

SFR = |{characters in target script}| / |{all non-whitespace characters}|
"""

import logging
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

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
# Unicode script ranges
# ---------------------------------------------------------------------------

# Mapping from ISO 639-1 language codes to Unicode ranges (inclusive)
# Format: List of (start_codepoint, end_codepoint) tuples
SCRIPT_UNICODE_RANGES: Dict[str, List[Tuple[int, int]]] = {
    # Southeast Asian scripts
    "th": [(0x0E00, 0x0E7F)],                          # Thai
    "my": [(0x1000, 0x109F), (0xA9E0, 0xA9FF), (0xAA60, 0xAA7F)],  # Myanmar (Burmese)
    "km": [(0x1780, 0x17FF), (0x19E0, 0x19FF)],        # Khmer
    "lo": [(0x0E80, 0x0EFF)],                          # Lao
    "jv": [(0xA980, 0xA9DF)],                          # Javanese
    "su": [(0x1B80, 0x1BBF), (0x1CC0, 0x1CCF)],        # Sundanese
    # Latin-script Southeast Asian languages
    "ceb": [(0x0041, 0x005A), (0x0061, 0x007A),        # Basic Latin A-Z a-z
             (0x00C0, 0x00FF)],                         # Latin Extended-A
    "fil": [(0x0041, 0x005A), (0x0061, 0x007A),
             (0x00C0, 0x00FF)],
    "id":  [(0x0041, 0x005A), (0x0061, 0x007A)],       # Indonesian (ASCII Latin)
    "ms":  [(0x0041, 0x005A), (0x0061, 0x007A)],       # Malay
    "vi":  [(0x0041, 0x005A), (0x0061, 0x007A),        # Vietnamese (Latin + diacritics)
             (0x00C0, 0x024F),                          # Latin Extended
             (0x1E00, 0x1EFF)],                         # Latin Extended Additional
    # African languages
    "am":  [(0x1200, 0x137F), (0x1380, 0x139F),        # Ethiopic (Amharic)
             (0x2D80, 0x2DDF), (0xAB00, 0xAB2F)],
    "sw":  [(0x0041, 0x005A), (0x0061, 0x007A)],       # Swahili (Latin)
    # Arabic-script languages (future)
    "ar":  [(0x0600, 0x06FF), (0x0750, 0x077F),
             (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
}

# Language name lookup
LANGUAGE_NAMES: Dict[str, str] = {
    "th": "Thai", "my": "Burmese", "km": "Khmer", "lo": "Lao",
    "jv": "Javanese", "su": "Sundanese", "ceb": "Cebuano",
    "fil": "Filipino", "id": "Indonesian", "ms": "Malay",
    "vi": "Vietnamese", "am": "Amharic", "sw": "Swahili",
    "ar": "Arabic",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SFRSample:
    """Per-sample SFR result.

    Attributes
    ----------
    prompt : str
    generated : str
    language : str
    sfr : float
        Script Fidelity Rate for this sample.
    n_chars : int
        Total non-whitespace characters.
    n_script_chars : int
        Characters in the target script.
    """
    prompt: str
    generated: str
    language: str
    sfr: float
    n_chars: int
    n_script_chars: int


@dataclass
class SFRReport:
    """Aggregated SFR results over a batch of samples.

    Attributes
    ----------
    samples : List[SFRSample]
    mean_sfr : float
    std_sfr : float
    min_sfr : float
    max_sfr : float
    per_language : Dict[str, float]
        Mean SFR broken down by language.
    n_total : int
    """
    samples: List[SFRSample]
    mean_sfr: float
    std_sfr: float
    min_sfr: float
    max_sfr: float
    per_language: Dict[str, float] = field(default_factory=dict)
    n_total: int = 0

    def to_dict(self) -> Dict:
        return {
            "mean_sfr": self.mean_sfr,
            "std_sfr": self.std_sfr,
            "min_sfr": self.min_sfr,
            "max_sfr": self.max_sfr,
            "per_language": self.per_language,
            "n_total": self.n_total,
            "samples": [
                {
                    "prompt": s.prompt[:80],
                    "language": s.language,
                    "sfr": s.sfr,
                    "n_chars": s.n_chars,
                    "n_script_chars": s.n_script_chars,
                }
                for s in self.samples
            ],
        }

    def summary(self) -> str:
        lines = [
            "SFR Report",
            f"  mean SFR : {self.mean_sfr:.4f} ± {self.std_sfr:.4f}",
            f"  range    : [{self.min_sfr:.4f}, {self.max_sfr:.4f}]",
            f"  n_total  : {self.n_total}",
        ]
        if self.per_language:
            lines.append("  Per-language:")
            for lang, sfr in sorted(self.per_language.items()):
                name = LANGUAGE_NAMES.get(lang, lang)
                lines.append(f"    {name:15s}: {sfr:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ScriptFidelityEvaluator:
    """Compute the Script Fidelity Rate (SFR) for generated text.

    SFR answers: "What fraction of generated characters are in the
    expected Unicode script of the target language?"

    This is the primary metric for evaluating *script-level language
    steering* — whether the model generates in the right alphabet/script,
    not just semantically correct content.

    Parameters
    ----------
    ignore_whitespace : bool, optional
        Whether to skip whitespace characters in the count.  Defaults to True.
    ignore_punctuation : bool, optional
        Whether to skip ASCII punctuation in the count.  Defaults to False.
    """

    def __init__(
        self,
        ignore_whitespace: bool = True,
        ignore_punctuation: bool = False,
    ) -> None:
        self.ignore_whitespace = ignore_whitespace
        self.ignore_punctuation = ignore_punctuation
        logger.info(
            "ScriptFidelityEvaluator | ignore_ws=%s ignore_punct=%s",
            ignore_whitespace,
            ignore_punctuation,
        )

    # ------------------------------------------------------------------
    # Core character-level check
    # ------------------------------------------------------------------

    def is_in_script(self, char: str, language: str) -> bool:
        """Check whether a character belongs to the expected script of a language.

        Parameters
        ----------
        char : str
            A single Unicode character.
        language : str
            ISO 639-1 language code.

        Returns
        -------
        bool
        """
        if language not in SCRIPT_UNICODE_RANGES:
            raise ValueError(
                f"Unknown language '{language}'. "
                f"Supported: {sorted(SCRIPT_UNICODE_RANGES.keys())}"
            )
        cp = ord(char)
        for start, end in SCRIPT_UNICODE_RANGES[language]:
            if start <= cp <= end:
                return True
        return False

    # ------------------------------------------------------------------
    # SFR computation
    # ------------------------------------------------------------------

    def compute_sfr(self, generated_text: str, target_language: str) -> float:
        """Compute Script Fidelity Rate for a single generated string.

        Parameters
        ----------
        generated_text : str
            The text generated by the model.
        target_language : str
            ISO 639-1 language code of the expected output script.

        Returns
        -------
        float
            SFR in ``[0.0, 1.0]``.  Returns 0.0 if the text is empty.
        """
        if not generated_text:
            return 0.0

        if target_language not in SCRIPT_UNICODE_RANGES:
            logger.warning(
                "Language '%s' not in SCRIPT_UNICODE_RANGES; SFR defaulting to 0.0",
                target_language,
            )
            return 0.0

        total_chars = 0
        script_chars = 0

        for char in generated_text:
            if self.ignore_whitespace and char.isspace():
                continue
            if self.ignore_punctuation and unicodedata.category(char).startswith("P"):
                continue

            total_chars += 1
            if self.is_in_script(char, target_language):
                script_chars += 1

        if total_chars == 0:
            return 0.0

        sfr = script_chars / total_chars
        return sfr

    def compute_sfr_detailed(
        self, generated_text: str, target_language: str
    ) -> Tuple[float, int, int]:
        """Compute SFR and return detailed counts.

        Returns
        -------
        Tuple of (sfr, n_script_chars, n_total_chars).
        """
        if not generated_text or target_language not in SCRIPT_UNICODE_RANGES:
            return 0.0, 0, 0

        total_chars = 0
        script_chars = 0

        for char in generated_text:
            if self.ignore_whitespace and char.isspace():
                continue
            if self.ignore_punctuation and unicodedata.category(char).startswith("P"):
                continue
            total_chars += 1
            if self.is_in_script(char, target_language):
                script_chars += 1

        sfr = script_chars / total_chars if total_chars > 0 else 0.0
        return sfr, script_chars, total_chars

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompts: List[str],
        target_languages: List[str],
        max_new_tokens: int = 128,
        batch_size: int = 4,
        device: str = "cpu",
    ) -> SFRReport:
        """Generate model outputs for a batch of prompts and compute SFR.

        Parameters
        ----------
        model : PreTrainedModel
        tokenizer : PreTrainedTokenizerBase
        prompts : List[str]
            Input prompts.
        target_languages : List[str]
            Expected output language for each prompt.
        max_new_tokens : int, optional
        batch_size : int, optional
        device : str, optional

        Returns
        -------
        SFRReport
        """
        if len(prompts) != len(target_languages):
            raise ValueError(
                f"Length mismatch: {len(prompts)} prompts vs "
                f"{len(target_languages)} languages."
            )

        _device = torch.device(device)
        model.eval()
        if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        samples: List[SFRSample] = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_langs = target_languages[i : i + batch_size]

            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(_device)
            prompt_len = enc["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            # Decode generated tokens (skip prompt)
            for j, (prompt, lang) in enumerate(zip(batch_prompts, batch_langs)):
                new_ids = output_ids[j, prompt_len:]
                generated = tokenizer.decode(new_ids, skip_special_tokens=True)
                sfr, n_script, n_total = self.compute_sfr_detailed(generated, lang)

                samples.append(
                    SFRSample(
                        prompt=prompt,
                        generated=generated,
                        language=lang,
                        sfr=sfr,
                        n_chars=n_total,
                        n_script_chars=n_script,
                    )
                )
                logger.debug(
                    "Sample %d | lang=%s sfr=%.4f generated=%r",
                    i + j,
                    lang,
                    sfr,
                    generated[:50],
                )

            logger.info(
                "Batch %d/%d complete",
                i // batch_size + 1,
                (len(prompts) + batch_size - 1) // batch_size,
            )

        return self._build_report(samples)

    def evaluate_generated(
        self,
        generated_texts: List[str],
        target_languages: List[str],
        prompts: Optional[List[str]] = None,
    ) -> SFRReport:
        """Compute SFR over pre-generated texts (no model inference).

        Parameters
        ----------
        generated_texts : List[str]
        target_languages : List[str]
        prompts : List[str], optional

        Returns
        -------
        SFRReport
        """
        if prompts is None:
            prompts = [""] * len(generated_texts)

        samples: List[SFRSample] = []
        for prompt, generated, lang in zip(prompts, generated_texts, target_languages):
            sfr, n_script, n_total = self.compute_sfr_detailed(generated, lang)
            samples.append(
                SFRSample(
                    prompt=prompt,
                    generated=generated,
                    language=lang,
                    sfr=sfr,
                    n_chars=n_total,
                    n_script_chars=n_script,
                )
            )

        return self._build_report(samples)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_report(self, samples: List[SFRSample]) -> SFRReport:
        """Aggregate per-sample results into an SFRReport."""
        import numpy as np

        sfrs = [s.sfr for s in samples]
        if not sfrs:
            return SFRReport(
                samples=[],
                mean_sfr=0.0,
                std_sfr=0.0,
                min_sfr=0.0,
                max_sfr=0.0,
                n_total=0,
            )

        sfr_arr = np.array(sfrs)

        # Per-language breakdown
        per_lang: Dict[str, List[float]] = {}
        for s in samples:
            per_lang.setdefault(s.language, []).append(s.sfr)
        per_lang_mean = {lang: float(np.mean(vs)) for lang, vs in per_lang.items()}

        report = SFRReport(
            samples=samples,
            mean_sfr=float(sfr_arr.mean()),
            std_sfr=float(sfr_arr.std()),
            min_sfr=float(sfr_arr.min()),
            max_sfr=float(sfr_arr.max()),
            per_language=per_lang_mean,
            n_total=len(samples),
        )
        logger.info(report.summary())
        return report


# ---------------------------------------------------------------------------
# Language Consistency (LC) — complements SFR for Latin-script target languages
# ---------------------------------------------------------------------------

# langid.py's supported ISO 639-1 set does not include Burmese ("my"); every other
# target language used in this project (th, lo, km, am, sw, id, ms, vi) is covered.
# Verified via langid.langid.LanguageIdentifier.from_modelstring(model).nb_classes.
LC_UNSUPPORTED_LANGUAGES = frozenset({"my"})


@dataclass
class LCSample:
    """Per-sample Language Consistency result."""
    generated: str
    target_language: str
    detected_language: Optional[str]
    is_consistent: Optional[bool]  # None when target_language is LC_UNSUPPORTED


@dataclass
class LCReport:
    """Aggregated Language Consistency results over a batch of samples."""
    samples: List[LCSample]
    consistency_rate: float  # fraction of *scorable* samples where detected == target
    n_scored: int
    n_unscorable: int
    per_language: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"LC Report\n"
            f"  consistency rate : {self.consistency_rate:.4f}\n"
            f"  n_scored         : {self.n_scored}\n"
            f"  n_unscorable     : {self.n_unscorable} (e.g. Burmese, unsupported by langid)\n"
            f"  per-language     : {self.per_language}"
        )


class LanguageConsistencyEvaluator:
    """Response-level "is the whole generation actually in the target language?" check.

    SFR is script-level (Unicode code-point ranges) and is therefore blind whenever the
    target language shares a script with a distractor language -- most notably every
    Latin-script target here (id, ms, sw, ceb, fil, vi): a fully-English response scores
    a *high* SFR for those languages because English is also Latin-script. LC uses a
    language-ID model (langid) on the whole response instead of a per-character script
    check, so it can actually tell Swahili from English drift, at the cost of being a
    coarser (whole-response, not per-character) signal.
    """

    def __init__(self) -> None:
        from langid.langid import LanguageIdentifier, model as _langid_model
        self._identifier = LanguageIdentifier.from_modelstring(_langid_model, norm_probs=True)
        logger.info("LanguageConsistencyEvaluator ready (langid backend).")

    def detect(self, text: str) -> Optional[str]:
        """Return the langid-predicted ISO 639-1 code, or None for empty input."""
        if not text or not text.strip():
            return None
        lang, _confidence = self._identifier.classify(text)
        return lang

    def compute_sample(self, generated_text: str, target_language: str) -> LCSample:
        if target_language in LC_UNSUPPORTED_LANGUAGES:
            return LCSample(
                generated=generated_text, target_language=target_language,
                detected_language=None, is_consistent=None,
            )
        detected = self.detect(generated_text)
        return LCSample(
            generated=generated_text, target_language=target_language,
            detected_language=detected,
            is_consistent=(detected == target_language) if detected is not None else None,
        )

    def evaluate_batch(
        self, generated_texts: List[str], target_languages: List[str]
    ) -> LCReport:
        if len(generated_texts) != len(target_languages):
            raise ValueError(
                f"generated_texts ({len(generated_texts)}) and target_languages "
                f"({len(target_languages)}) must be the same length."
            )
        samples = [
            self.compute_sample(text, lang)
            for text, lang in zip(generated_texts, target_languages)
        ]
        scorable = [s for s in samples if s.is_consistent is not None]
        n_unscorable = len(samples) - len(scorable)
        consistency_rate = (
            sum(1 for s in scorable if s.is_consistent) / len(scorable)
            if scorable else 0.0
        )

        per_lang: Dict[str, List[bool]] = {}
        for s in scorable:
            per_lang.setdefault(s.target_language, []).append(bool(s.is_consistent))
        per_lang_rate = {lang: sum(vs) / len(vs) for lang, vs in per_lang.items()}

        report = LCReport(
            samples=samples,
            consistency_rate=consistency_rate,
            n_scored=len(scorable),
            n_unscorable=n_unscorable,
            per_language=per_lang_rate,
        )
        logger.info(report.summary())
        return report
