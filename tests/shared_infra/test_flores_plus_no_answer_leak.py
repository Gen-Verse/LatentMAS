"""Regression test: FLORES+ tasks must not leak the gold translation into
AgentTask.context, since every specialized agent's prompt embeds context verbatim
(see AgentTask.context's docstring). The gold answer belongs in AgentTask.reference,
read only by _compute_translation_quality for scoring -- never by an agent's prompt.

Found live on 2026-07-02: benchmark_runner._load_real_tasks used to set
context=tgt_text, so the first agent in every comm-mode received the answer key as
part of its own input (confirmed empirically: ReasoningAgent's output on one sample
directly echoed the Thai reference it had been handed as "context").
"""

from unittest.mock import patch

from latent_coordination.agents.base_agent import AgentTask
from latent_coordination.eval.benchmark_runner import MultiAgentBenchmarkRunner

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"


class _FakeRow(dict):
    pass


class _FakeDataset(list):
    pass


def _fake_load_dataset(repo_id, *, name=None, split=None, cache_dir=None, **kwargs):
    # datasets.load_dataset("openlanguagedata/flores_plus", name=<code>, split="devtest")
    if name == "eng_Latn":
        return _FakeDataset([_FakeRow(text=f"English sentence {i}.") for i in range(3)])
    return _FakeDataset([_FakeRow(text=f"<{name} script gold translation {i}>") for i in range(3)])


def test_agent_task_has_separate_reference_field():
    task = AgentTask(task_id="t1", query="hello")
    assert task.context == ""
    assert task.reference is None


def test_flores_plus_tasks_do_not_leak_reference_into_context(tmp_path):
    runner = MultiAgentBenchmarkRunner(output_dir=str(tmp_path), max_samples_per_language=3, languages=["th"])
    with patch("datasets.load_dataset", side_effect=_fake_load_dataset):
        tasks = runner._load_real_tasks()

    assert len(tasks) == 3
    for t in tasks:
        # The gold translation must never appear in context (what agents' prompts read).
        assert t.context == ""
        assert "gold translation" not in t.context
        # It must be available for scoring via `reference`.
        assert t.reference is not None
        assert "gold translation" in t.reference
        assert t.query.startswith("English sentence")


def test_compute_translation_quality_scores_against_reference_not_context(tmp_path):
    runner = MultiAgentBenchmarkRunner(output_dir=str(tmp_path))
    tasks = [
        AgentTask(task_id="a", query="src", context="DECOY -- must not be scored against",
                  reference="hello world", target_language="en"),
    ]
    metrics = runner._compute_translation_quality(["hello world"], tasks)
    # A perfect match against the real reference should score a high chrF.
    assert metrics["chrf"] > 90.0
