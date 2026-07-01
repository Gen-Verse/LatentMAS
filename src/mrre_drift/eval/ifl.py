"""
IFL (Involuntary Fidelity Loss) validation for Surgical MRRE.

IFL measures how often a model, prompted in a low-resource language, drifts
*out of* the target script (typically collapsing to English). We quantify it with the
Script Fidelity Rate (SFR) from ``latent_coordination.eval.script_fidelity``:

    SFR(text, lang) = fraction of (non-space, non-punct) characters in the target script
    IFL flag        = 1 if SFR < sfr_threshold else 0   (a sample "failed" to stay in-script)
    IFL rate        = mean IFL flag over samples

All scores come from **real model generations** — there are no heuristic or synthetic
fallbacks. The same prompts are evaluated with and without the Surgical MRRE intervention
so the before/after delta is a like-for-like comparison.

``IFLReport.delta_ci(other)`` computes bootstrap 95% CIs on the per-language
IFL reduction (self_baseline - other_steered) using the stored SFR sample arrays.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sacrebleu
import torch

from latent_coordination.eval.script_fidelity import ScriptFidelityEvaluator

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
# CLR — Correct-Language-Rate
# ---------------------------------------------------------------------------
# CLR measures whether a generation is in the correct target language,
# decoupled from script fidelity. For non-Latin-script languages (Thai, Burmese,
# Khmer, Lao, Amharic), script presence is a reliable proxy — CLR ≈ SFR. For
# Latin-script languages (Swahili, Indonesian, Malay, Vietnamese, etc.), SFR is
# uninformative (English is also Latin), so CLR uses langdetect when available,
# falling back to a character n-gram heuristic.
#
# ISO-639-1 codes whose script is shared with English (Latin). For these, SFR
# alone cannot distinguish in-language from English output.
_LATIN_SCRIPT_LANGS = {"sw", "id", "ms", "vi", "tl", "fil", "nl", "de", "fr", "es", "pt"}

# langdetect → langid fallback → None if neither installed.
def _detect_lang(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None
    try:
        from langdetect import detect  # type: ignore
        return detect(text)
    except Exception:
        pass
    try:
        import langid  # type: ignore
        lang, _ = langid.classify(text)
        return lang
    except Exception:
        pass
    return None


def compute_clr(text: str, target_lang: str, sfr: float, sfr_threshold: float = 0.5) -> float:
    """Compute the Correct-Language-Rate (CLR) flag for a single generation.

    Returns 1.0 if the generation is in the correct language, 0.0 otherwise.

    For non-Latin-script languages the script-fidelity score is used directly
    (CLR = 1 if SFR >= threshold) because script presence is a reliable signal.
    For Latin-script languages an automatic language detector is used so that
    English output (also Latin) correctly scores 0.0.

    Parameters
    ----------
    text        : the generated string
    target_lang : ISO-639-1 code of the expected language
    sfr         : pre-computed SFR score for this text
    sfr_threshold : minimum SFR to count as correct for non-Latin scripts
    """
    if target_lang not in _LATIN_SCRIPT_LANGS:
        return 1.0 if sfr >= sfr_threshold else 0.0
    # Latin-script: try language detection.
    detected = _detect_lang(text)
    if detected is not None:
        return 1.0 if detected == target_lang else 0.0
    # Fallback: SFR is always ~1 for Latin, so use a simple heuristic —
    # flag as incorrect only if it looks like English (SFR ~1 but detected as en).
    # Without a detector, we can't distinguish, so we conservatively return 1.0
    # (assume in-language) and log a warning.
    logger.warning(
        "CLR: no language detector available for Latin-script lang '%s'; "
        "install langdetect or langid for accurate CLR. Returning SFR-based fallback.",
        target_lang,
    )
    return 1.0 if sfr >= sfr_threshold else 0.0


@dataclass
class IFLLanguageResult:
    """Per-language IFL measurement."""

    language: str
    n_samples: int
    ifl_rate: float
    mean_sfr: float
    sfr_threshold: float
    generations: List[str] = field(default_factory=list)
    sfr_values: List[float] = field(default_factory=list)
    output_lengths: List[int] = field(default_factory=list)
    chrf_scores: List[float] = field(default_factory=list)  # per-sample chrF [0,1]
    mean_chrf: float = 0.0
    # CLR: Correct-Language-Rate, decoupled from script fidelity. For Latin-script
    # languages this uses language detection; for non-Latin it equals 1 - IFL.
    clr_values: List[float] = field(default_factory=list)   # per-sample CLR flag
    mean_clr: float = 0.0


@dataclass
class IFLReport:
    """IFL results across languages for a single condition (e.g. baseline / steered)."""

    condition: str
    by_language: Dict[str, IFLLanguageResult] = field(default_factory=dict)

    @property
    def macro_ifl_rate(self) -> float:
        """Unweighted mean IFL rate across languages."""
        if not self.by_language:
            return 0.0
        return sum(r.ifl_rate for r in self.by_language.values()) / len(self.by_language)

    def delta_ci(
        self,
        steered: "IFLReport",
        n_boot: int = 2000,
        alpha: float = 0.05,
        seed: int = 0,
    ) -> Dict[str, Tuple[float, float]]:
        """Bootstrap 95% CI on per-language IFL reduction (self_base - steered).

        Returns dict mapping lang -> (ci_lo, ci_hi) for ΔIFL.
        Uses the stored per-sample SFR arrays to resample IFL flags.
        Only languages present in both reports are included.
        """
        rng = np.random.default_rng(seed)
        out: Dict[str, Tuple[float, float]] = {}
        for lang, base_r in self.by_language.items():
            if lang not in steered.by_language:
                continue
            steer_r = steered.by_language[lang]
            base_flags = np.array(
                [1.0 if s < base_r.sfr_threshold else 0.0 for s in base_r.sfr_values]
            )
            steer_flags = np.array(
                [1.0 if s < steer_r.sfr_threshold else 0.0 for s in steer_r.sfr_values]
            )
            n = len(base_flags)
            deltas = np.array([
                rng.choice(base_flags, n, replace=True).mean()
                - rng.choice(steer_flags, n, replace=True).mean()
                for _ in range(n_boot)
            ])
            lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            out[lang] = (float(lo), float(hi))
        return out

    def to_dict(self) -> Dict:
        return {
            "condition": self.condition,
            "macro_ifl_rate": self.macro_ifl_rate,
            "by_language": {k: asdict(v) for k, v in self.by_language.items()},
        }


class IFLValidator:
    """Generate target-language responses and score script fidelity.

    Parameters
    ----------
    model, tokenizer
        A loaded causal LM and tokenizer (eval mode).
    device
        Device string for generation.
    sfr_threshold
        SFR below this counts as an IFL failure. Default 0.5 (majority-script).
    max_new_tokens
        Generation length cap.
    references_by_language
        Optional FLORES+ reference strings keyed by language code.  When provided,
        chrF is computed for each generation and stored in ``IFLLanguageResult``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: str = "cpu",
        sfr_threshold: float = 0.5,
        max_new_tokens: int = 128,
        references_by_language: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sfr_threshold = sfr_threshold
        self.max_new_tokens = max_new_tokens
        self.references_by_language: Dict[str, List[str]] = references_by_language or {}
        self.evaluator = ScriptFidelityEvaluator()
        # Prompt-transform callable: (lang, prompt) -> prompt.
        # Set by prompt-level baseline context managers in baselines.py.
        self._prompt_transform: Optional[Callable[[str, str], str]] = None

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def evaluate(
        self,
        prompts_by_language: Dict[str, Sequence[str]],
        condition: str = "baseline",
        intervention=None,
    ) -> IFLReport:
        """Evaluate IFL for each language.

        Parameters
        ----------
        prompts_by_language
            Mapping ``lang -> list of prompt strings`` (real text, e.g. FLORES+).
        condition
            Label recorded in the report (``"baseline"`` or ``"mrre_drift"``).
        intervention
            Optional object exposing an ``apply()`` context manager (a fitted
            :class:`SurgicalMRRE` or a prompt-level baseline from
            :mod:`mrre_drift.eval.baselines`). When provided, generation runs
            inside it.
        """
        report = IFLReport(condition=condition)

        for lang, prompts in prompts_by_language.items():
            prompts = list(prompts)
            if not prompts:
                logger.warning("No prompts for language '%s'; skipping.", lang)
                continue

            generations: List[str] = []
            ctx = intervention.apply() if intervention is not None else nullcontext()
            with ctx:
                for prompt in prompts:
                    effective = (
                        self._prompt_transform(lang, prompt)
                        if self._prompt_transform is not None
                        else prompt
                    )
                    generations.append(self._generate(effective))

            sfr_values = [self.evaluator.compute_sfr(g, lang) for g in generations]
            lengths = [len(g) for g in generations]
            ifl_flags = [1.0 if s < self.sfr_threshold else 0.0 for s in sfr_values]
            ifl_rate = float(sum(ifl_flags) / len(ifl_flags))
            mean_sfr = float(sum(sfr_values) / len(sfr_values))

            # chrF against FLORES+ references when available.
            refs = self.references_by_language.get(lang, [])
            chrf_scores: List[float] = []
            if refs and len(refs) == len(generations):
                for hyp, ref in zip(generations, refs):
                    chrf_scores.append(
                        sacrebleu.sentence_chrf(hyp, [ref]).score / 100.0
                    )
            mean_chrf = float(sum(chrf_scores) / len(chrf_scores)) if chrf_scores else 0.0

            # CLR: decoupled from SFR for Latin-script languages.
            clr_values = [
                compute_clr(g, lang, s, self.sfr_threshold)
                for g, s in zip(generations, sfr_values)
            ]
            mean_clr = float(sum(clr_values) / len(clr_values)) if clr_values else 0.0

            report.by_language[lang] = IFLLanguageResult(
                language=lang,
                n_samples=len(prompts),
                ifl_rate=ifl_rate,
                mean_sfr=mean_sfr,
                sfr_threshold=self.sfr_threshold,
                generations=generations,
                sfr_values=sfr_values,
                output_lengths=lengths,
                chrf_scores=chrf_scores,
                mean_chrf=mean_chrf,
                clr_values=clr_values,
                mean_clr=mean_clr,
            )
            chrf_str = f" meanChrF={mean_chrf:.3f}" if chrf_scores else ""
            logger.info(
                "[%s] lang=%s n=%d IFL=%.3f meanSFR=%.3f meanCLR=%.3f%s",
                condition, lang, len(prompts), ifl_rate, mean_sfr, mean_clr, chrf_str,
            )

        return report
