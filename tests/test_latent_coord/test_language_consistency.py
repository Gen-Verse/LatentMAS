"""Tests for LanguageConsistencyEvaluator (LC metric).

LC complements Script Fidelity Rate (SFR): SFR is a per-character Unicode-range
check and is blind between same-script languages (e.g. Swahili vs English are both
Latin-script), while LC runs whole-response language ID (langid) so it can actually
tell them apart.
"""

from latent_coordination.eval.script_fidelity import (
    LanguageConsistencyEvaluator,
    LC_UNSUPPORTED_LANGUAGES,
)

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"


def test_detects_correct_language():
    ev = LanguageConsistencyEvaluator()
    sample = ev.compute_sample("Hii ni sentensi ya Kiswahili yenye maneno mengi.", "sw")
    assert sample.detected_language == "sw"
    assert sample.is_consistent is True


def test_flags_english_drift_for_latin_script_target():
    # SFR would score this as "fully in target script" for sw/id/ms (all Latin) --
    # this is exactly the blind spot LC exists to catch.
    ev = LanguageConsistencyEvaluator()
    sample = ev.compute_sample("This response is entirely in English.", "sw")
    assert sample.detected_language == "en"
    assert sample.is_consistent is False


def test_burmese_is_unscorable_not_silently_wrong():
    ev = LanguageConsistencyEvaluator()
    assert "my" in LC_UNSUPPORTED_LANGUAGES
    sample = ev.compute_sample("မြန်မာဘာသာဖြင့် ဖြေကြားပေးပါ", "my")
    assert sample.detected_language is None
    assert sample.is_consistent is None


def test_evaluate_batch_aggregates_and_skips_unscorable():
    ev = LanguageConsistencyEvaluator()
    texts = [
        "Hii ni sentensi ya Kiswahili.",
        "This is English, not Swahili.",
        "မြန်မာဘာသာဖြင့် ဖြေကြားပေးပါ",
    ]
    langs = ["sw", "sw", "my"]
    report = ev.evaluate_batch(texts, langs)
    assert report.n_scored == 2
    assert report.n_unscorable == 1
    assert report.consistency_rate == 0.5
    assert report.per_language["sw"] == 0.5


def test_mismatched_lengths_raises():
    ev = LanguageConsistencyEvaluator()
    try:
        ev.evaluate_batch(["a", "b"], ["en"])
        assert False, "expected ValueError"
    except ValueError:
        pass
