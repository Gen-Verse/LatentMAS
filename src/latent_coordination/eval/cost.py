"""Token + wall-clock cost accounting for multi-agent coordination (P3-T6).

Produces the accuracy-vs-token-cost frontier at N=4, 8, 16 agents for our
latent-MAS system and all baselines, in the style of G-Designer/AgentPrune
tables.

Key metrics reported
--------------------
- prompt_tokens     : total input tokens consumed across all agents per task
- completion_tokens : total generated tokens per task
- total_tokens      : prompt + completion per task
- wall_clock_ms     : end-to-end latency per task (real time, not flops)
- accuracy          : correctness score from :mod:`eval.correctness`

These are measured at N ∈ {4, 8, 16} agent-steps per task.  The latent-MAS
system should show sub-linear token growth in N (only hub transfers, no
decoded messages) while accuracy improves with N.

Usage
-----
    from latent_coordination.eval.cost import CostAccountant, CostReport

    acct = CostAccountant(tokenizer=tok)
    acct.record(
        system="latent_mas", n_agents=4,
        is_correct=True, prompt_tokens=120, completion_tokens=85, wall_ms=234.1
    )
    report = acct.finalize()
    report.print_frontier()
    report.save_json(Path("results/cost_frontier.json"))
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


logger = logging.getLogger(__name__)

CANONICAL_N_VALUES = (4, 8, 16)


# ---------------------------------------------------------------------------
# Per-task observation
# ---------------------------------------------------------------------------

@dataclass
class CostObservation:
    """One task's cost + correctness record."""
    system: str          # e.g. "latent_mas", "latentmas_baseline", "text_mas", "single_agent"
    n_agents: int        # number of agent-steps (4, 8, or 16)
    is_correct: bool
    prompt_tokens: int
    completion_tokens: int
    wall_ms: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------------------
# Aggregate result per (system, N) cell
# ---------------------------------------------------------------------------

@dataclass
class CostCell:
    """Aggregated cost + accuracy for one (system, N) cell of the frontier."""
    system: str
    n_agents: int
    n_tasks: int
    accuracy: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    mean_total_tokens: float
    mean_wall_ms: float
    std_total_tokens: float
    std_wall_ms: float
    accuracy_ci_95: Tuple[float, float]   # bootstrap 95% CI on accuracy


def _bootstrap_mean_ci(
    values: List[float],
    n_boot: int = 2000,
    seed: int = 0,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    if len(arr) < 2:
        return (float(arr.mean()) if len(arr) else float("nan"),) * 2
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# Main accountant
# ---------------------------------------------------------------------------

class CostAccountant:
    """Collect per-task observations and produce the accuracy-vs-cost frontier.

    Parameters
    ----------
    tokenizer
        Optional HF tokenizer used by :meth:`count_tokens` for convenience.
        Not required if callers supply pre-counted tokens.
    n_boot
        Bootstrap resamples for accuracy CI (default 2000).
    """

    def __init__(self, tokenizer=None, n_boot: int = 2000) -> None:
        self.tokenizer = tokenizer
        self.n_boot = n_boot
        self._obs: List[CostObservation] = []

    def count_tokens(self, text: str) -> int:
        """Count tokens in ``text`` using the configured tokenizer.

        Falls back to whitespace-split word count if no tokenizer is set.
        """
        if self.tokenizer is not None:
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        return len(text.split())

    def record(
        self,
        system: str,
        n_agents: int,
        is_correct: bool,
        prompt_tokens: int,
        completion_tokens: int,
        wall_ms: float,
    ) -> None:
        """Record one task's cost observation."""
        self._obs.append(CostObservation(
            system=system,
            n_agents=n_agents,
            is_correct=is_correct,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            wall_ms=wall_ms,
        ))

    def record_from_texts(
        self,
        system: str,
        n_agents: int,
        is_correct: bool,
        prompt_texts: List[str],
        completion_texts: List[str],
        wall_ms: float,
    ) -> None:
        """Record using raw text strings (token counts computed internally)."""
        prompt_toks = sum(self.count_tokens(t) for t in prompt_texts)
        compl_toks = sum(self.count_tokens(t) for t in completion_texts)
        self.record(system, n_agents, is_correct, prompt_toks, compl_toks, wall_ms)

    def finalize(self) -> "CostReport":
        """Aggregate observations into frontier cells and return a CostReport."""
        # Group by (system, n_agents).
        groups: Dict[Tuple[str, int], List[CostObservation]] = defaultdict(list)
        for obs in self._obs:
            groups[(obs.system, obs.n_agents)].append(obs)

        cells: List[CostCell] = []
        for (system, n_agents), obs_list in sorted(groups.items()):
            correct_flags = [float(o.is_correct) for o in obs_list]
            total_toks = [o.total_tokens for o in obs_list]
            wall_ms = [o.wall_ms for o in obs_list]
            acc_ci = _bootstrap_mean_ci(correct_flags, self.n_boot)
            cells.append(CostCell(
                system=system,
                n_agents=n_agents,
                n_tasks=len(obs_list),
                accuracy=float(np.mean(correct_flags)),
                mean_prompt_tokens=float(np.mean([o.prompt_tokens for o in obs_list])),
                mean_completion_tokens=float(np.mean([o.completion_tokens for o in obs_list])),
                mean_total_tokens=float(np.mean(total_toks)),
                mean_wall_ms=float(np.mean(wall_ms)),
                std_total_tokens=float(np.std(total_toks)),
                std_wall_ms=float(np.std(wall_ms)),
                accuracy_ci_95=acc_ci,
            ))
        return CostReport(cells=cells, n_obs=len(self._obs))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class CostReport:
    """Accuracy-vs-token-cost frontier across all (system, N) cells."""
    cells: List[CostCell]
    n_obs: int
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )

    def print_frontier(self) -> None:
        """Print a formatted frontier table to stdout."""
        header = (
            f"{'System':<22} {'N':>4} {'Acc':>6} {'CI95':>15}"
            f" {'TotalTok':>9} {'WallMs':>8} {'n':>5}"
        )
        print(header)
        print("-" * len(header))
        for c in self.cells:
            lo, hi = c.accuracy_ci_95
            print(
                f"{c.system:<22} {c.n_agents:>4} {c.accuracy:>6.3f}"
                f" [{lo:.3f},{hi:.3f}]"
                f" {c.mean_total_tokens:>9.1f} {c.mean_wall_ms:>8.1f} {c.n_tasks:>5}"
            )

    def frontier_for_paper(self) -> List[Dict]:
        """Return a list of rows formatted for LaTeX table generation."""
        return [
            {
                "system": c.system,
                "N": c.n_agents,
                "accuracy": round(c.accuracy, 4),
                "ci_lo": round(c.accuracy_ci_95[0], 4),
                "ci_hi": round(c.accuracy_ci_95[1], 4),
                "mean_total_tokens": round(c.mean_total_tokens, 1),
                "mean_wall_ms": round(c.mean_wall_ms, 1),
                "n_tasks": c.n_tasks,
            }
            for c in self.cells
        ]

    def to_dict(self) -> Dict:
        return {
            "timestamp_utc": self.timestamp_utc,
            "n_obs": self.n_obs,
            "cells": [asdict(c) for c in self.cells],
            "frontier_for_paper": self.frontier_for_paper(),
        }

    def save_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Cost frontier saved to %s", path)


# ---------------------------------------------------------------------------
# Convenience wrapper: run a workload at multiple N values
# ---------------------------------------------------------------------------

def run_at_n_values(
    run_fn,
    n_values: Tuple[int, ...] = CANONICAL_N_VALUES,
    **run_kwargs,
) -> CostAccountant:
    """Call ``run_fn(n_agents=N, **run_kwargs)`` for each N and collect costs.

    ``run_fn`` must accept ``n_agents: int`` and return an iterable of
    ``(is_correct: bool, prompt_tokens: int, completion_tokens: int, wall_ms: float)``.

    Parameters
    ----------
    run_fn
        A callable that runs the system at a given N and yields per-task tuples.
    n_values
        Agent-count values to sweep (default 4, 8, 16).
    run_kwargs
        Extra keyword arguments forwarded to ``run_fn`` (e.g. ``system="latent_mas"``).

    Returns
    -------
    A filled :class:`CostAccountant`.
    """
    system = run_kwargs.pop("system", "unknown")
    acct = CostAccountant()
    for n in n_values:
        logger.info("Running %s at N=%d", system, n)
        for is_correct, prompt_toks, compl_toks, wall_ms in run_fn(n_agents=n, **run_kwargs):
            acct.record(
                system=system,
                n_agents=n,
                is_correct=is_correct,
                prompt_tokens=prompt_toks,
                completion_tokens=compl_toks,
                wall_ms=wall_ms,
            )
    return acct
