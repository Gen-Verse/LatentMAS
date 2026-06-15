import random
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import json
from google import genai

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Lineesha Kamana"
__email__ = "lpk5305@psu.edu, hthakur@uccs.edu"
__status__ = "prototype"


# -----------------------
# Config
# -----------------------
API_KEY_ENV = "YOUR API"


# -----------------------
# Gemini client
# -----------------------
def get_client():
    api_key = API_KEY_ENV
    if not api_key:
        raise RuntimeError(f"Set {API_KEY_ENV} environment variable.")
    return genai.Client(api_key=api_key)


_NUM_RE = re.compile(r"(?<![\w/])(-?\d+(?:\.\d+)?)(?![\w/])")
_YEAR_RE = re.compile(r"^(19\d{2}|20\d{2})$")

@dataclass
class NumericEditResult:
    edited_question: str
    changed_from: str
    changed_to: str
    match_index: int
    span: Tuple[int, int]
    rule: str


def _is_safe_numeric_candidate_general(
    num_str: str,
    text: str,
    span: Tuple[int, int],
    *,
    avoid_modulus_constants: bool = True,
) -> bool:
    # Avoid likely years
    if _YEAR_RE.match(num_str):
        return False

    # Avoid ordinals like 1st/2nd/3rd/4th (rare in AIME but harmless)
    after = text[span[1]: span[1] + 2].lower()
    if after in {"st", "nd", "rd", "th"}:
        return False

    # Avoid fractions like 1/2
    before = text[max(0, span[0] - 1): span[0]]
    after1 = text[span[1]: span[1] + 1]
    if before == "/" or after1 == "/":
        return False

    # Avoid huge numbers (IDs)
    try:
        val = float(num_str)
        if abs(val) >= 1e6:
            return False
    except Exception:
        return False

    # Optional: avoid editing modulus constants like "mod 1000", "divided by 1000"
    if avoid_modulus_constants:
        window = text[max(0, span[0] - 25): min(len(text), span[1] + 25)].lower()
        if ("mod" in window) or ("remainder" in window) or ("divided by" in window):
            # If the number appears very near modulus language, skip it
            # (prevents changing "1000" in "remainder when N is divided by 1000")
            return False

    return True


def _perturb_number_small(num_str: str, rng: random.Random) -> Tuple[str, str]:
    is_int = re.fullmatch(r"-?\d+", num_str) is not None
    if is_int:
        n = int(num_str)
        if n in (0, 1, 2):
            return str(n + 1), "int:+1_small"
        delta = rng.choice([1, 2])
        return str(n + delta), f"int:+{delta}"

    x = float(num_str)
    if abs(x) < 1:
        delta = 0.1
    elif abs(x) < 10:
        delta = 0.5
    else:
        delta = 1.0
    new_x = x + delta
    new_str = str(int(new_x)) if new_x.is_integer() else str(new_x).rstrip("0").rstrip(".")
    return new_str, f"float:+{delta}"


def single_number_edit(
    question: str,
    *,
    seed: Optional[int] = None,
    preferred: str = "first",  # "first" or "random"
    avoid_modulus_constants: bool = True,
) -> NumericEditResult:
    """
    Works for MGSM and AIME-style questions containing LaTeX.
    Changes exactly one numeric span with a small perturbation.

    If avoid_modulus_constants=True, avoids editing numbers near 'mod', 'remainder', 'divided by'.
    """
    matches = list(_NUM_RE.finditer(question))
    if not matches:
        raise ValueError("No numeric spans found in question.")

    safe: List[Tuple[int, re.Match]] = []
    for i, m in enumerate(matches):
        span = m.span(1)
        num_str = m.group(1)
        if _is_safe_numeric_candidate_general(
            num_str, question, span,
            avoid_modulus_constants=avoid_modulus_constants
        ):
            safe.append((i, m))

    if not safe:
        # Fall back: allow modulus constants if that was the reason we failed
        if avoid_modulus_constants:
            return single_number_edit(
                question,
                seed=seed,
                preferred=preferred,
                avoid_modulus_constants=False,
            )
        raise ValueError("No safe numeric candidates found to edit.")

    rng = random.Random(seed)
    chosen_i, chosen_m = (safe[0] if preferred == "first" else rng.choice(safe))

    span = chosen_m.span(1)
    old = chosen_m.group(1)
    new, rule = _perturb_number_small(old, rng)

    edited = question[:span[0]] + new + question[span[1]:]

    # Validate: same count of numeric spans; exactly one position differs
    orig_nums = [m.group(1) for m in _NUM_RE.finditer(question)]
    edited_nums = [m.group(1) for m in _NUM_RE.finditer(edited)]
    if len(orig_nums) != len(edited_nums):
        raise RuntimeError("Numeric edit changed number of numeric spans unexpectedly.")
    diffs = [k for k, (a, b) in enumerate(zip(orig_nums, edited_nums)) if a != b]
    if diffs != [chosen_i]:
        raise RuntimeError(f"Expected exactly one numeric change at index {chosen_i}, got diffs={diffs}")

    return NumericEditResult(
        edited_question=edited,
        changed_from=old,
        changed_to=new,
        match_index=chosen_i,
        span=span,
        rule=rule,
    )


PARAPHRASE_PROMPT_TEMPLATE_AIME_LANG = r"""
You are rewriting a competition math problem statement.

Language constraint (MUST follow):
- The paraphrase MUST be written in the SAME language as the original question.
- The original question language is: {language_name}. Do NOT translate to any other language.

Hard constraints:
1) Preserve ALL numbers exactly (character-for-character).
2) Preserve ALL LaTeX math exactly as-is (anything inside $...$ must appear unchanged).
3) Keep the question asking for the same final quantity; the problem must be logically equivalent.
4) Reduce lexical overlap by paraphrasing and reordering sentences outside math mode.
5) Do NOT include any solution steps, explanations, or the final answer.
6) Do NOT add or remove any facts, entities, units, or constraints.

Return ONLY valid JSON with exactly these keys:
{{"paraphrase": "...", "changes": "..."}}

Original problem:
<<<
{problem}
>>>
""".strip()



def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return t


def _extract_math_segments(s: str) -> List[str]:
    # naive but effective for standard AIME/MGSM: segments inside single-dollar math
    return re.findall(r"\$.*?\$", s, flags=re.DOTALL)


def _numbers_multiset(s: str) -> List[str]:
    return sorted([m.group(1) for m in _NUM_RE.finditer(s)])


def paraphrase_reorder_with_gemini(
    question: str,
    client,
    *,
    language_name: str = "English",
    model_name: str = "gemini-2.5-flash",
    max_retries: int = 2,
    aime_strict_math: bool = True,
) -> dict:
    """
    Paraphrase/reorder while preserving numbers, and (optionally) preserving $...$ segments exactly.
    Works for AIME and MGSM (MGSM usually has no $...$).
    """
    # print(question)
    prompt = PARAPHRASE_PROMPT_TEMPLATE_AIME_LANG.format(
        problem=question,
        language_name=language_name,
    )
    orig_nums = _numbers_multiset(question)
    orig_math = _extract_math_segments(question) if aime_strict_math else None

    last_err = None
    raw_text = ""
    for _ in range(max_retries + 1):
        resp = client.models.generate_content(model=model_name, contents=prompt)
        raw_text = _strip_code_fences(resp.text)
        try:
            data = json.loads(raw_text)
            paraphrase = (data.get("paraphrase") or "").strip()
            changes = (data.get("changes") or "").strip()
            if not paraphrase:
                raise ValueError("Empty paraphrase.")

            # validate numbers
            if _numbers_multiset(paraphrase) != orig_nums:
                raise ValueError("Number multiset mismatch.")

            # validate math segments unchanged (order should typically be the same)
            if aime_strict_math:
                para_math = _extract_math_segments(paraphrase)
                if para_math != orig_math:
                    raise ValueError("LaTeX $...$ segments changed or reordered.")

            return {
                "edited_question": paraphrase,
                "changes": changes,
                "raw_response": raw_text,
                "ok": True,
            }
        except Exception as e:
            last_err = str(e)
            continue

    return {
        "edited_question": "",
        "changes": "",
        "raw_response": raw_text,
        "ok": False,
        "error": f"Failed after retries: {last_err}",
    }

