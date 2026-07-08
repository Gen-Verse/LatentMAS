# helper.py
import re
from typing import List
import json
from pathlib import Path

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


# ---------------------------------------------------------------------
# Language normalization + answer prefixes
# ---------------------------------------------------------------------

_LANG_NORMALIZE = {
    "en": "EN",
    "fr": "FR",
    "de": "DE",
    "zh": "ZH",
    "ja": "JA",
    "ru": "RU",
    "es": "ES",
    "sw": "SW",
    "bn": "BN",
    "te": "TE",
    "th": "TH",
}


def normalize_lang_key(lang: str) -> str:
    """
    Normalize language key to canonical upper-case form, e.g. 'en' -> 'EN'.
    """
    if not lang:
        return "EN"
    key = lang.strip()
    lower = key.lower()
    return _LANG_NORMALIZE.get(lower, key.upper())


# Language-wise answer lead-ins, derived from your instructions templates.
_ANSWER_PREFIX = {
    "EN": "The answer is:  \\boxed{",
    "FR": "La réponse est :  \\boxed{",
    "DE": "Die Antwort ist:  \\boxed{",
    "ZH": "答案是： \\boxed{",
    "JA": "答えは： \\boxed{",
    "RU": "Ответ:  \\boxed{",
    "ES": "La respuesta es:  \\boxed{",
    "SW": "Jibu ni:  \\boxed{",
    "BN": "উত্তর হল:  \\boxed{",
    "TE": "సమాధానం:  \\boxed{",
    "TH": "คำตอบคือ:  \\boxed{",
}


def get_answer_prefix(lang: str) -> str:
    """
    Get the localized 'answer lead-in' phrase, e.g. 'The answer is: '.
    Defaults to English if language not found.
    """
    lang_key = normalize_lang_key(lang)
    return _ANSWER_PREFIX.get(lang_key, _ANSWER_PREFIX["EN"])


# ---------------------------------------------------------------------
# Punctuation sets (for secondary segmentation)
# ---------------------------------------------------------------------

_LANG_SENT_PUNCT = {
    "EN": ".!?",
    "FR": ".!?",
    "DE": ".!?",
    "ES": ".!?",
    "SW": ".!?",
    "RU": ".!?",

    "ZH": "。！？!?",   # full-width + ascii
    "JA": "。！？!?",

    "BN": "।.!?",     # danda + '.' + !?
    "TE": "।.!?",
    "TH": ".!?",
}

_DEFAULT_PUNCT = ".!?"


def _get_punct_set(lang: str):
    lang_key = normalize_lang_key(lang)
    punct = _LANG_SENT_PUNCT.get(lang_key, _DEFAULT_PUNCT)
    return set(punct)


# ---------------------------------------------------------------------
# Primary segmentation: split by blank lines (\n\n), preserving them
# ---------------------------------------------------------------------

def _split_by_blanklines(text: str) -> List[str]:
    """
    Split text into units separated by blank lines.

    - A 'blank line' is defined as: '\n' + optional spaces/tabs + '\n'.
    - The separating blank-line sequence is ATTACHED to the preceding unit.
    - All internal whitespace/newlines are preserved exactly.
    """
    if not text:
        return []

    units: List[str] = []
    buf: List[str] = []

    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if ch == "\n":
            # Look ahead: is this a blank line?  '\n' + spaces/tabs + '\n'
            j = i + 1
            while j < n and text[j] in (" ", "\t"):
                j += 1
            if j < n and text[j] == "\n":
                # Yes, it's a blank line separator; attach it to current unit.
                buf.append(text[i : j + 1])  # includes both newlines + spaces
                i = j + 1
                unit = "".join(buf)
                if unit.strip():
                    units.append(unit)
                buf = []
                continue

        # Default: accumulate character
        buf.append(ch)
        i += 1

    # Tail
    if buf:
        unit = "".join(buf)
        if unit.strip():
            units.append(unit)

    return units


# ---------------------------------------------------------------------
# Secondary segmentation: punctuation-aware splitting inside a unit
# ---------------------------------------------------------------------

def _split_unit_by_punct(unit: str, lang: str) -> List[str]:
    """
    Further split a unit into smaller steps using language-specific
    sentence-ending punctuation, while:

    - NOT splitting on decimals like '3.4' (digit '.' digit).
    - Preserving any trailing whitespace/newlines with the punctuation,
      e.g. 'here is it.\\n\\n' stays in one segment.

    Returns a list of substrings whose concatenation == original `unit`.
    """
    if not unit.strip():
        return []

    punct_set = _get_punct_set(lang)

    segments: List[str] = []
    start = 0
    i = 0
    n = len(unit)

    while i < n:
        ch = unit[i]
        is_boundary = False

        if ch in punct_set:
            if ch == "." and i > 0 and i + 1 < n:
                # If digit '.' digit -> decimal, don't cut
                if unit[i - 1].isdigit() and unit[i + 1].isdigit():
                    is_boundary = False
                else:
                    is_boundary = True
            else:
                is_boundary = True

        if is_boundary:
            # Include punctuation + any trailing whitespace/newlines
            j = i + 1
            while j < n and unit[j] in (" ", "\t", "\n", "\r"):
                j += 1

            segment = unit[start:j]
            if segment.strip():
                segments.append(segment)
            start = j
            i = j
        else:
            i += 1

    # Tail
    if start < n:
        segment = unit[start:]
        if segment.strip():
            segments.append(segment)

    return segments if segments else [unit]


# ---------------------------------------------------------------------
# Hybrid sentence tokenization
# ---------------------------------------------------------------------

_MIN_UNITS_DEFAULT = 100


def sentence_tokenize(text: str, lang: str, min_units: int = _MIN_UNITS_DEFAULT) -> List[str]:
    """
    Hybrid "sentence" tokenization tailored for reasoning traces:

    1. First, split by blank lines (\\n + optional spaces + \\n).
       This gives paragraph-level steps and preserves math formulas.
    2. If we already have >= `min_units` units, return them.
    3. Otherwise, further split each unit using punctuation-aware segmentation
       (language-specific), while:
         - preserving '\\n\\n' and internal formatting,
         - avoiding splits inside decimals like '3.4'.

    Output: list of units, each a substring of the original text.
    """
    if not text:
        return []

    # Step 1: paragraph-level segmentation
    units = _split_by_blanklines(text)

    if len(units) >= min_units:
        return units

    # Step 2: refine units via punctuation splitting
    refined: List[str] = []
    for u in units:
        refined.extend(_split_unit_by_punct(u, lang))

    return refined


# ---------------------------------------------------------------------
# Thinking trace extraction (using only </think> in response)
# ---------------------------------------------------------------------

def extract_think_text(response: str, lang: str) -> str:
    """
    Extract the thinking trace text from a full model response.

    YOUR SETUP:
    - <think> appears only in the *prompt*, NOT in the response.
    - The model writes a closing '</think>' in the response.

    Behavior:
    - If '</think>' is present: thinking = everything BEFORE '</think>'.
      (Formatting kept exactly.)
    - If '</think>' is missing:
        * use hybrid sentence_tokenize on the response
        * treat all but the LAST unit as thinking (last ≈ answer-ish).
    """
    if not response:
        return ""

    if "</think>" in response:
        idx = response.index("</think>")
        return response[:idx]

    # Fallback: no </think>, use all but last unit as thinking
    units = sentence_tokenize(response, lang)
    if len(units) <= 1:
        return ""
    think_text = "".join(units[:-1])
    return think_text


def extract_think_sentences(response: str, lang: str, min_units: int = _MIN_UNITS_DEFAULT) -> List[str]:
    """
    Convenience wrapper:
    - Extract thinking text as a raw substring.
    - Then apply hybrid sentence_tokenize to it.
    """
    think_text = extract_think_text(response, lang)
    return sentence_tokenize(think_text, lang, min_units=min_units)


# ---------------------------------------------------------------------
# Truncation utilities
# ---------------------------------------------------------------------

def truncate_think_sentences(sentences: List[str], ratio: float, reverse=False) -> List[str]:
    """
    Truncate the thinking trace at a given ratio (0 < ratio <= 1)
    based on unit count.

    “We approximate the location of critical reasoning steps by comparing performance under 
    prefix-only and suffix-only truncation at varying ratios; 
    overlapping breakpoints suggest a core region of influential reasoning content.”
    """
    if not sentences or ratio == 0:
        return []

    if ratio >= 1:
        return sentences
    
    if reverse:
        ratio = 1 - ratio

    ratio = max(0.0, min(1.0, ratio))
    if ratio <= 0.0:
        return []

    n = len(sentences)
    keep = max(1, int(round(n * ratio)))
    keep = min(keep, n)
    
    if reverse:
        # remove first k sentences → keep the rest
        return sentences[keep:]
    else:
        # keep first k sentences
        return sentences[:keep]


def build_truncated_think_block(response: str, lang: str, ratio: float, reverse=False) -> str:
    """
    Given a full model response and a truncation ratio, build:

        <think>
        ...truncated thinking...
        </think>

        <answer lead-in>

    where the answer lead-in is language-specific, e.g.:

        </think>

        The answer is:

    This string is meant to be appended to your original prompt (which already
    included '<think>' before the generation region).
    """
    sentences = extract_think_sentences(response, lang)
    truncated_sents = truncate_think_sentences(sentences, ratio, reverse)
    truncated_text = "".join(truncated_sents)  # keep all newlines as-is

    answer_prefix = get_answer_prefix(lang)

    if truncated_text:
        think_block = f"{truncated_text}\n</think>\n\n{answer_prefix}"
    else:
        # No thinking text – still emit tags + answer prompt
        think_block = f"</think>\n\n{answer_prefix}"

    return think_block
