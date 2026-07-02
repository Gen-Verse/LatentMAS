"""Real (non-mocked) smoke tests for the benchmark loaders fixed this session.

Per dev_doc.md's zero-tolerance-mocks policy, these hit the actual HF datasets (all
cached locally already) rather than faking rows -- a loader with a schema bug (wrong
split name, wrong column name, missing dataset config) fails loudly here instead of
silently returning nothing.
"""

import itertools

import data

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"


def test_load_laobench_real_row():
    row = next(iter(data.load_laobench()))
    assert row["question"]
    assert row["gold"] in {"a", "b", "c", "d"}


def test_load_sea_helm_requires_known_language():
    import pytest
    with pytest.raises(ValueError, match="no config for language"):
        next(iter(data.load_sea_helm(lang="xx")))


def test_load_sea_helm_real_row():
    row = next(iter(data.load_sea_helm(lang="th")))
    assert row["question"]
    assert row["gold"] in {"a", "b", "c", "d"}


def test_load_mgsm_pro_real_row():
    row = next(iter(data.load_mgsm_pro(lang="en")))
    assert row["question"]
    assert row["gold"].isdigit() or row["gold"].lstrip("-").isdigit()


def test_load_mgsm_pro_rejects_unsupported_language():
    import pytest
    with pytest.raises(ValueError, match="does not have data for language"):
        next(iter(data.load_mgsm_pro(lang="th")))


def test_load_mathmist_rejects_unsupported_language():
    import pytest
    with pytest.raises(ValueError, match="does not have data for language"):
        next(iter(data.load_mathmist(lang="th")))


def test_load_mathmist_am_and_sw_real_rows():
    am_row = next(iter(data.load_mathmist(lang="am")))
    sw_row = next(iter(data.load_mathmist(lang="sw")))
    assert am_row["question"] and am_row["gold"]
    assert sw_row["question"] and sw_row["gold"]
