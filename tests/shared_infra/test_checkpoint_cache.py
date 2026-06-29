"""Tests for the CheckpointManager keyed result cache (fine-grained resume)."""

import tempfile

from shared.checkpointing import CheckpointManager


def test_keyed_cache_round_trip():
    cm = CheckpointManager(tempfile.mkdtemp(), "proj")
    key = "mech::author/Model-8B::benchmark::belebele::no_intervention"
    assert not cm.has_result(key)
    cm.cache_result(key, {"ifl": 0.31, "n": 200})
    assert cm.has_result(key)
    assert cm.get_result(key) == {"ifl": 0.31, "n": 200}


def test_distinct_keys_isolated():
    cm = CheckpointManager(tempfile.mkdtemp(), "proj")
    cm.cache_result("coord::M::mode::token_based_mas", {"acc": 0.5})
    cm.cache_result("coord::M::mode::latent_based_mas_ours", {"acc": 0.7})
    assert cm.get_result("coord::M::mode::token_based_mas") == {"acc": 0.5}
    assert cm.get_result("coord::M::mode::latent_based_mas_ours") == {"acc": 0.7}
    assert not cm.has_result("coord::M::mode::single_agent_baseline")


def test_get_missing_raises():
    cm = CheckpointManager(tempfile.mkdtemp(), "proj")
    import pytest
    with pytest.raises(FileNotFoundError):
        cm.get_result("nope")
