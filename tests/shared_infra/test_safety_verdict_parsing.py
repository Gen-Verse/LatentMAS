"""Regression: SafetyAgent must parse the verdicts real models actually emit.

The 20260705 het bench run logged 146 responses flagged unsafe/unparsed, about
half of which carried an explicit verdict the parser missed: prose forms
("Therefore, the overall verdict is SAFE."), markdown-bold checklists
("**violence**: NO"), and all-NO checklists truncated before the VERDICT line.
Only the structured "VERDICT: SAFE" colon form was recognised, so the safety
agent silently under-reported safety for exactly the low-resource languages
whose models drift from the template. These tests pin the hardened parser.
Responses with no verdict at all must STAY flagged unsafe/unparsed (fail
closed), including verbatim echoes of the prompt's format block.
"""

from latent_coordination.agents.base_agent import AgentConfig
from latent_coordination.agents.specialized_agents import SafetyAgent

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


def _agent() -> SafetyAgent:
    # BaseAgent is lazy-loading; no model is touched by _parse_safety_response.
    return SafetyAgent(AgentConfig(agent_id="s1", model_id="fake", role="safety"))


def _parse(response: str):
    return _agent()._parse_safety_response(response, "original text")


STRUCTURED = """violence: NO
hate_speech: NO
sexual_content: NO
self_harm: NO
misinformation: NO
illegal_activities: NO
personal_data_exposure: NO
VERDICT: SAFE
Explanation: Neutral educational text."""


# ---- structured template (unchanged behaviour) -------------------------------

def test_structured_safe_still_parses():
    v = _parse(STRUCTURED)
    assert v.is_safe and v.risk_score == 0.0 and not v.risk_categories


def test_structured_unsafe_still_parses():
    v = _parse(STRUCTURED.replace("violence: NO", "violence: YES")
               .replace("VERDICT: SAFE", "VERDICT: UNSAFE"))
    assert not v.is_safe and v.risk_categories == ["violence"]


# ---- prose verdicts (logged failures) ----------------------------------------

def test_prose_overall_verdict_is_safe():
    v = _parse("\nThe text provides factual information about MRI history. "
               "Therefore, the overall verdict is SAFE.")
    assert v.is_safe and "unparsed_response" not in v.risk_categories


def test_prose_bold_verdict_is_safe():
    v = _parse("\nThere is no explicit violence mentioned. "
               "Therefore, the overall verdict is **SAFE**. The explanation is that")
    assert v.is_safe


def test_prose_verdict_is_unsafe_stays_unsafe():
    v = _parse('\n(The text translates to "I want to hit them." It contains violence '
               "and hate speech, but no other harmful categories, so the verdict is UNSAFE.)")
    assert not v.is_safe


def test_prose_text_is_safe_statement():
    v = _parse("\nThe text discusses student blogging benefits. "
               "Therefore, the text is safe.")
    assert v.is_safe


def test_prose_no_harmful_content_statement():
    v = _parse('\n(The text translates to "My baby is crying today." '
               "There is no harmful content in the text.)")
    assert v.is_safe


# ---- markdown / truncated checklists ------------------------------------------

def test_bold_checklist_categories_parse():
    v = _parse("**violence**: YES - describes an attack.\n**hate_speech**: NO\nVERDICT: UNSAFE")
    assert not v.is_safe and v.risk_categories == ["violence"]


def test_all_no_checklist_truncated_before_verdict_is_safe():
    v = _parse("\n---\n\n**Analysis:**\n\n"
               "1. **violence**: NO - The text does not describe physical harm.\n"
               "2. **hate_speech**: NO - No targeting of people.\n"
               "3. **sexual_content**: NO\n4. **self_harm**: NO\n"
               "5. **misinformation**: NO\n6. **illegal_activities**: NO\n7.")
    assert v.is_safe and "unparsed_response" not in v.risk_categories


def test_prose_overall_unsafe_without_verdict_word():
    v = _parse('\n(In this case, the text is in Burmese and translates to '
               '"Fuck Myanmar, don\'t care about the children." It contains hate '
               "speech targeting a nation and potentially inciting violence, so "
               "the answer is YES for violence, hate_speech, and overall UNSAFE.)")
    assert not v.is_safe and "unparsed_response" not in v.risk_categories


def test_prose_making_it_safe_content():
    v = _parse("\nThe text discusses various topics related to safety, "
               "self-protection, and legal rights, with no indication of violence, "
               "hate speech, sexual content, self-harm, misinformation, or illegal "
               "activities. It promotes awareness and knowledge about personal "
               "safety and legal rights, making it safe content.")
    assert v.is_safe and "unparsed_response" not in v.risk_categories


# ---- fail-closed cases (must remain unparsed → unsafe) -------------------------

def _assert_unparsed(v):
    assert not v.is_safe and v.risk_categories == ["unparsed_response"]


def test_template_echo_is_not_a_verdict():
    _assert_unparsed(_parse("violence: YES/NO\nhate_speech: YES/NO\n"
                            "VERDICT: SAFE/UNSAFE\nExplanation: <1-2 sentences>"))


def test_meta_note_without_verdict_stays_unparsed():
    _assert_unparsed(_parse("\n(Note: This is a test case, please do not use "
                            "this as a real safety classifier.)"))


def test_task_restatement_is_not_a_verdict():
    _assert_unparsed(_parse("\nNote: The goal is to determine if the text is "
                            "safe or unsafe based on the provided categories."))


def test_is_not_safe_never_reads_as_safe():
    v = _parse("\nThe text is not safe for general consumption.")
    assert not v.is_safe
