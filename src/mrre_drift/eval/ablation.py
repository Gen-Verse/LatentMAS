"""Ablation harness for Surgical MRRE (P1-T6).

Runs all ablation conditions through the IFLValidator and returns a
structured AblationTable that can be serialised for the paper.

Conditions
----------
no_intervention      Raw model, no steering.
stage1_only          Stage-1 cross-lingual enhancement only (no anchoring).
stage2_only          Stage-2 target-language anchoring only (no enhancement).
full_ramped          Both stages with the surgical depth-ramped anchoring (ours).
full_uniform_anchor  Both stages, anchoring norms equalised (vanilla MRRE, no ramp).
randomized_layers    Both stages, layer sets drawn randomly from the valid pool
                     (averaged over ``n_random_seeds`` seeds — promoted to main table
                     per plan to let reviewers verify the "surgical" claim).
system_prompt        Prompt-level: prepend "Please respond in <Language>.\n\n".
few_shot             Prompt-level: prepend ``n_shots`` FLORES+ exemplars.

The randomized-layers condition is the main control for the "surgical" claim:
if ``full_ramped ≤ randomized_layers`` on macro IFL, trigger layer attribution.

Usage
-----
    from mrre_drift.eval.ablation import AblationRunner

    runner = AblationRunner(
        model, tokenizer, device="cuda:0",
        conditions=["no_intervention", "full_ramped", "randomized_layers"],
        n_random_seeds=3,
        n_shots=3,
    )
    table = runner.run(surgical, prompts_by_lang, references_by_lang=refs)
    print(table.summary())
    table.save_json(Path("results/ablations/table.json"))
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"
__version__ = "0.1.0"

logger = logging.getLogger(__name__)

ALL_CONDITIONS: List[str] = [
    "no_intervention",
    "stage1_only",
    "stage2_only",
    "full_ramped",
    "full_uniform_anchor",
    "randomized_layers",
    "system_prompt",
    "few_shot",
]


# ---------------------------------------------------------------------------
# Thin intervention adapters (match IFLValidator's .apply() CM interface)
# ---------------------------------------------------------------------------

class _NoOpIntervention:
    @contextmanager
    def apply(self):
        yield self


class _Stage1OnlyAdapter:
    def __init__(self, surgical):
        self._s = surgical

    @contextmanager
    def apply(self):
        with self._s.apply_stage1_only():
            yield self


class _Stage2OnlyAdapter:
    def __init__(self, surgical):
        self._s = surgical

    @contextmanager
    def apply(self):
        with self._s.apply_stage2_only():
            yield self


class _UniformAnchorAdapter:
    def __init__(self, surgical):
        self._s = surgical

    @contextmanager
    def apply(self):
        with self._s.apply_uniform_anchor():
            yield self


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AblationConditionResult:
    """IFL/SFR/CLR results for a single ablation condition."""
    condition: str
    macro_ifl_rate: float
    macro_sfr: float
    macro_clr: float
    macro_chrf: float
    ifl_reduction_vs_baseline: float   # baseline_ifl - this_ifl
    by_language: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class AblationTable:
    """Full ablation table across all requested conditions."""
    timestamp_utc: str
    model_id: str
    target_languages: List[str]
    conditions: List[AblationConditionResult] = field(default_factory=list)
    # Decision flag: did randomized_layers beat full_ramped on macro IFL?
    surgical_beats_random: Optional[bool] = None
    note: str = ""

    def summary(self) -> str:
        header = f"Ablation Table — {self.model_id} — langs: {self.target_languages}\n"
        header += f"{'Condition':<25} {'MacroIFL':>9} {'MacroSFR':>9} {'MacroCLR':>9} {'MacroChrF':>10} {'ΔIFL(vs_base)':>14}\n"
        header += "-" * 78 + "\n"
        rows = ""
        for c in self.conditions:
            rows += (
                f"{c.condition:<25} {c.macro_ifl_rate:>9.4f} {c.macro_sfr:>9.4f}"
                f" {c.macro_clr:>9.4f} {c.macro_chrf:>10.4f} {c.ifl_reduction_vs_baseline:>14.4f}\n"
            )
        flag = ""
        if self.surgical_beats_random is not None:
            flag = f"\nSurgical > Random: {self.surgical_beats_random}"
            if not self.surgical_beats_random:
                flag += "  ← WARNING: activate layer attribution (P1-T6b)"
        return header + rows + flag

    def to_dict(self) -> Dict:
        return {
            "timestamp_utc": self.timestamp_utc,
            "model_id": self.model_id,
            "target_languages": self.target_languages,
            "conditions": [asdict(c) for c in self.conditions],
            "surgical_beats_random": self.surgical_beats_random,
            "note": self.note,
        }

    def save_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Ablation table saved to %s", path)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class AblationRunner:
    """Run all ablation conditions and produce an AblationTable.

    Parameters
    ----------
    model, tokenizer
        Loaded causal LM (eval mode).
    device
        Device for generation.
    conditions
        Subset of :data:`ALL_CONDITIONS` to evaluate; defaults to all.
    n_random_seeds
        Number of random-layer seeds to average over (default 3 per plan).
    n_shots
        Number of FLORES+ exemplars for the few-shot baseline.
    sfr_threshold
        SFR threshold for IFL flag (default 0.5).
    max_new_tokens
        Generation length cap.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        conditions: Optional[List[str]] = None,
        n_random_seeds: int = 3,
        n_shots: int = 3,
        sfr_threshold: float = 0.5,
        max_new_tokens: int = 128,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.conditions = conditions or ALL_CONDITIONS
        self.n_random_seeds = n_random_seeds
        self.n_shots = n_shots
        self.sfr_threshold = sfr_threshold
        self.max_new_tokens = max_new_tokens

        invalid = [c for c in self.conditions if c not in ALL_CONDITIONS]
        if invalid:
            raise ValueError(f"Unknown conditions: {invalid}. Valid: {ALL_CONDITIONS}")

    def _make_validator(self, references_by_lang: Optional[Dict] = None):
        from mrre_drift.eval.ifl import IFLValidator
        return IFLValidator(
            self.model, self.tokenizer,
            device=self.device,
            sfr_threshold=self.sfr_threshold,
            max_new_tokens=self.max_new_tokens,
            references_by_language=references_by_lang or {},
        )

    def _aggregate(self, report) -> Dict[str, float]:
        """Aggregate per-language IFL report into macro metrics."""
        if not report.by_language:
            return {"macro_ifl": 0.0, "macro_sfr": 0.0, "macro_clr": 0.0, "macro_chrf": 0.0}
        results = list(report.by_language.values())
        macro_ifl = sum(r.ifl_rate for r in results) / len(results)
        macro_sfr = sum(r.mean_sfr for r in results) / len(results)
        macro_clr = sum(r.mean_clr for r in results) / len(results)
        macro_chrf = sum(r.mean_chrf for r in results) / len(results)
        return {
            "macro_ifl": macro_ifl,
            "macro_sfr": macro_sfr,
            "macro_clr": macro_clr,
            "macro_chrf": macro_chrf,
            "by_language": {lang: asdict(r) for lang, r in report.by_language.items()},
        }

    def run(
        self,
        surgical,
        prompts_by_lang: Dict[str, List[str]],
        exemplars_by_lang: Optional[Dict[str, List[str]]] = None,
        references_by_lang: Optional[Dict[str, List[str]]] = None,
        prompt_pairs: Optional[List] = None,
        forcing_pairs: Optional[List] = None,
        checkpoint_manager=None,
        cache_prefix: Optional[str] = None,
    ) -> AblationTable:
        """Run all requested conditions and return the ablation table.

        Parameters
        ----------
        surgical
            A fitted :class:`~mrre_drift.mrre.surgical.SurgicalMRRE` instance.
        prompts_by_lang
            Evaluation prompts per language.
        exemplars_by_lang
            Few-shot exemplars per language (required for ``few_shot`` condition).
        references_by_lang
            FLORES+ reference strings per language for chrF computation.
        prompt_pairs, forcing_pairs
            Calibration data for re-fitting randomized-layer instances.
        checkpoint_manager, cache_prefix
            Optional caching to skip already-completed conditions.
        """
        from mrre_drift.eval.baselines import FewShotBaseline, SystemPromptBaseline

        langs = list(prompts_by_lang.keys())
        model_id = getattr(self.model, "name_or_path", "unknown")
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        validator = self._make_validator(references_by_lang)

        # Compute baseline (no_intervention) first — needed for ΔIFL.
        baseline_ifl = 0.0

        condition_results: List[AblationConditionResult] = []

        # Build the intervention map.
        def _run_condition(name: str, intervention) -> Dict:
            if checkpoint_manager is not None and cache_prefix:
                key = f"{cache_prefix}::abl::{name}"
                if checkpoint_manager.has_result(key):
                    logger.info("Condition '%s' loaded from cache.", name)
                    return checkpoint_manager.get_result(key)
            report = validator.evaluate(prompts_by_lang, condition=name, intervention=intervention)
            agg = self._aggregate(report)
            if checkpoint_manager is not None and cache_prefix:
                checkpoint_manager.cache_result(f"{cache_prefix}::abl::{name}", agg)
            return agg

        for cond in self.conditions:
            logger.info("Running ablation condition: %s", cond)

            if cond == "no_intervention":
                agg = _run_condition(cond, _NoOpIntervention())
                baseline_ifl = agg["macro_ifl"]

            elif cond == "stage1_only":
                agg = _run_condition(cond, _Stage1OnlyAdapter(surgical))

            elif cond == "stage2_only":
                agg = _run_condition(cond, _Stage2OnlyAdapter(surgical))

            elif cond == "full_ramped":
                agg = _run_condition(cond, surgical)

            elif cond == "full_uniform_anchor":
                agg = _run_condition(cond, _UniformAnchorAdapter(surgical))

            elif cond == "randomized_layers":
                if prompt_pairs is None or forcing_pairs is None:
                    logger.warning(
                        "randomized_layers requires prompt_pairs + forcing_pairs; skipping."
                    )
                    continue
                seed_ilfs = []
                seed_sfrs = []
                seed_clrs = []
                seed_chrfs = []
                for seed in range(self.n_random_seeds):
                    rand_surgical = surgical.fit_randomized_layers(
                        prompt_pairs, forcing_pairs, seed=seed
                    )
                    key_seed = f"{cond}_seed{seed}"
                    agg_s = _run_condition(key_seed, rand_surgical)
                    seed_ilfs.append(agg_s["macro_ifl"])
                    seed_sfrs.append(agg_s["macro_sfr"])
                    seed_clrs.append(agg_s["macro_clr"])
                    seed_chrfs.append(agg_s["macro_chrf"])
                agg = {
                    "macro_ifl": sum(seed_ilfs) / len(seed_ilfs),
                    "macro_sfr": sum(seed_sfrs) / len(seed_sfrs),
                    "macro_clr": sum(seed_clrs) / len(seed_clrs),
                    "macro_chrf": sum(seed_chrfs) / len(seed_chrfs),
                    "by_language": {},
                    "n_seeds": self.n_random_seeds,
                }

            elif cond == "system_prompt":
                baseline = SystemPromptBaseline(validator)
                agg = _run_condition(cond, baseline)

            elif cond == "few_shot":
                if not exemplars_by_lang:
                    logger.warning("few_shot requires exemplars_by_lang; skipping.")
                    continue
                baseline = FewShotBaseline(validator, exemplars_by_lang, n_shots=self.n_shots)
                agg = _run_condition(cond, baseline)

            else:
                continue

            delta = baseline_ifl - agg["macro_ifl"]
            condition_results.append(AblationConditionResult(
                condition=cond,
                macro_ifl_rate=agg["macro_ifl"],
                macro_sfr=agg["macro_sfr"],
                macro_clr=agg["macro_clr"],
                macro_chrf=agg["macro_chrf"],
                ifl_reduction_vs_baseline=delta,
                by_language=agg.get("by_language", {}),
            ))

        # Decision: does surgical beat random?
        surgical_beats_random: Optional[bool] = None
        ramped = next((c for c in condition_results if c.condition == "full_ramped"), None)
        random = next((c for c in condition_results if c.condition == "randomized_layers"), None)
        if ramped is not None and random is not None:
            surgical_beats_random = ramped.macro_ifl_rate < random.macro_ifl_rate
            if not surgical_beats_random:
                logger.warning(
                    "SURGICAL ≤ RANDOM on macro IFL (%.4f ≤ %.4f). "
                    "Consider adding activation-patching layer attribution (P1-T6b).",
                    ramped.macro_ifl_rate, random.macro_ifl_rate,
                )

        table = AblationTable(
            timestamp_utc=ts,
            model_id=model_id,
            target_languages=langs,
            conditions=condition_results,
            surgical_beats_random=surgical_beats_random,
        )
        logger.info("Ablation complete.\n%s", table.summary())
        return table
