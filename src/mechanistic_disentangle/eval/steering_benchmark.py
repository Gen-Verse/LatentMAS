"""Benchmark runner for executing Mechanistic Disentanglement activation steering evaluations."""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from latent_coordination.eval.script_fidelity import ScriptFidelityEvaluator
from latent_coordination.eval.metrics import MetricsComputer

__author__ = "Himon Thakur"
__copyright__ = "Copyright [2026], Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkReport:
    """Contains results for all configurations evaluated on a dataset."""
    timestamp: str
    results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def generate_ablation_table(self) -> pd.DataFrame:
        """Create a pandas DataFrame formatting the ablation table for publication."""
        rows = []
        for config, metrics in self.results.items():
            rows.append({
                "Configuration": config.replace("_", " ").title(),
                "Accuracy": metrics.get("accuracy", 0.0),
                "Script Consistency (SFR)": metrics.get("mean_sfr", 0.0),
                "Manifold Adherence (PPL)": metrics.get("perplexity", 999.9),
                "IFL Rate": metrics.get("ifl_rate", 0.0),
            })
        return pd.DataFrame(rows)

    def save_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("BenchmarkReport saved to %s", path)


def _require_model(model, tokenizer) -> None:
    """Raise ValueError if model or tokenizer is None.

    All evaluation paths require a real loaded model. Passing None is not
    permitted — it indicates a misconfiguration in the calling pipeline.

    Raises
    ------
    ValueError
        If either model or tokenizer is None.
    """
    if model is None:
        raise ValueError(
            "BenchmarkRunner requires a real model. "
            "model=None is not allowed. Load the model before calling run_baseline() or run_steered()."
        )
    if tokenizer is None:
        raise ValueError(
            "BenchmarkRunner requires a real tokenizer. "
            "tokenizer=None is not allowed."
        )


class BenchmarkRunner:
    """Orchestrates running steering evaluations against various baseline configurations."""

    def __init__(self, output_dir: Optional[Path | str] = "results/mechanistic") -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("results/mechanistic")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sfr_eval = ScriptFidelityEvaluator()

    def _generate_predictions(self, model, tokenizer, prompts: List[str], max_new_tokens: int = 64) -> List[str]:
        """Generate text predictions from the real model."""
        device = next(model.parameters()).device
        predictions = []
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out[0][enc["input_ids"].shape[1]:]
            predictions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        return predictions

    def _score_options(self, model, tokenizer, prompt: str, options: List[str]) -> str:
        """Score multiple choice options using log-likelihood."""
        device = next(model.parameters()).device
        best_opt = None
        best_lp = -float('inf')
        
        prompt_enc = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = prompt_enc["input_ids"].shape[1]
        
        for opt in options:
            text = prompt + opt
            enc = tokenizer(text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                out = model(enc["input_ids"])
            
            logits = out.logits[0, :-1, :]
            labels = enc["input_ids"][0, 1:]
            
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            
            opt_len = enc["input_ids"].shape[1] - prompt_len
            if opt_len <= 0:
                opt_len = 1
                
            opt_lp = token_log_probs[-opt_len:].sum().item()
            
            if opt_lp > best_lp:
                best_lp = opt_lp
                best_opt = opt
                
        return best_opt.strip()

    def run_baseline(
        self,
        model,
        tokenizer,
        samples,
        config_name: str = "no_intervention",
    ) -> Dict[str, float]:
        """Run evaluation on vanilla model without steering.

        Parameters
        ----------
        model : PreTrainedModel
            A real loaded HuggingFace model. Must not be None.
        tokenizer : PreTrainedTokenizerBase
            A real loaded tokenizer. Must not be None.
        samples : List
            Dataset samples with .text, .reference_answer, .language attributes.
        config_name : str
            Label for this evaluation configuration.

        Returns
        -------
        Dict[str, float]
            Accuracy, mean_sfr, perplexity, ifl_rate.
        """
        _require_model(model, tokenizer)
        logger.info("Running baseline configuration: %s", config_name)

        prompts = [s.text for s in samples]
        references = [str(s.reference_answer) for s in samples]
        target_langs = [s.language for s in samples]

        is_mcqa = all(r.strip() in {"1", "2", "3", "4"} for r in references)
        
        if is_mcqa:
            logger.info("Detected MCQA format. Using log-likelihood scoring.")
            predictions = []
            options = [" 1", " 2", " 3", " 4"]
            for prompt in prompts:
                pred = self._score_options(model, tokenizer, prompt, options)
                predictions.append(pred)
        else:
            predictions = self._generate_predictions(model, tokenizer, prompts)

        sfr_report = self.sfr_eval.evaluate_generated(predictions, target_langs, prompts)
        acc = MetricsComputer.compute_accuracy(predictions, references)
        ifl = float(np.mean([1.0 - (1.0 if s >= 0.5 else 0.0) for s in [sm.sfr for sm in sfr_report.samples]]))
        ppl = MetricsComputer.compute_perplexity(model, tokenizer, predictions)

        return {
            "accuracy": acc,
            "mean_sfr": sfr_report.mean_sfr,
            "perplexity": ppl,
            "ifl_rate": ifl,
        }

    def run_steered(
        self,
        model,
        tokenizer,
        samples,
        steerer,
        config_name: str,
        steering_vectors,
        layer_ids,
        eta: float,
        apply_subspace_projection: bool = False,
        use_schedule: bool = True,
    ) -> Dict[str, float]:
        """Run evaluation on steered model using LatentSteerer hooks.

        Parameters
        ----------
        model : PreTrainedModel
            A real loaded HuggingFace model. Must not be None.
        tokenizer : PreTrainedTokenizerBase
            A real loaded tokenizer. Must not be None.
        samples : List
            Dataset samples.
        steerer : LatentSteerer
            Configured steerer for activation injection.
        config_name : str
            Label for this configuration.
        steering_vectors : Dict[int, Tensor]
            Per-layer steering vectors derived from real contrastive activations.
        layer_ids : List[int]
            Layers at which to inject steering vectors.
        eta : float
            Scaling factor for magnitude normalization.
        apply_subspace_projection : bool
            Whether to apply SVD subspace projection before injection.

        Returns
        -------
        Dict[str, float]
            Accuracy, mean_sfr, perplexity, ifl_rate.
        """
        _require_model(model, tokenizer)
        logger.info("Running steered configuration: %s", config_name)

        prompts = [s.text for s in samples]
        references = [str(s.reference_answer) for s in samples]
        target_langs = [s.language for s in samples]

        is_mcqa = all(r.strip() in {"1", "2", "3", "4"} for r in references)
        predictions = []
        
        if is_mcqa:
            logger.info("Detected MCQA format. Using log-likelihood scoring.")
            options = [" 1", " 2", " 3", " 4"]
            with steerer.apply(model, steering_vectors, layer_ids, eta, apply_subspace_projection, use_schedule):
                for prompt in prompts:
                    pred = self._score_options(model, tokenizer, prompt, options)
                    predictions.append(pred)
        else:
            for prompt, lang in zip(prompts, target_langs):
                pred = steerer.generate(
                    model=model,
                    tokenizer=tokenizer,
                    input_text=prompt,
                    steering_vectors=steering_vectors,
                    layer_ids=layer_ids,
                    eta=eta,
                    max_new_tokens=64,
                    apply_subspace_projection=apply_subspace_projection,
                    use_schedule=use_schedule,
                )
                predictions.append(pred)

        sfr_report = self.sfr_eval.evaluate_generated(predictions, target_langs, prompts)
        acc = MetricsComputer.compute_accuracy(predictions, references)
        ifl = float(np.mean([1.0 - (1.0 if s >= 0.5 else 0.0) for s in [sm.sfr for sm in sfr_report.samples]]))
        ppl = MetricsComputer.compute_perplexity(model, tokenizer, predictions)

        return {
            "accuracy": acc,
            "mean_sfr": sfr_report.mean_sfr,
            "perplexity": ppl,
            "ifl_rate": ifl,
        }

    ALL_BASELINES = (
        "no_intervention", "standard_clas", "mrre_two_stage",
        "gaussian_scheduled", "aggressive_oversteering",
    )

    # Per-baseline steering configuration. Each steered baseline MUST differ from
    # every other in at least one knob, otherwise two baselines produce identical
    # results (the mrre_two_stage == standard_clas dispatch bug). Knobs:
    #   use_schedule  : Gaussian depth schedule (True) vs flat/uniform weights (False)
    #   subspace_proj : project sv onto reasoning subspace before injection
    #   eta_mult      : multiplier on the base injection fraction
    #   * subspace_proj is only effective when the steerer is built with a
    #     decomposer (per-layer map); otherwise it degrades to no projection.
    # vanilla_mrre (Li et al.) is omitted: in this implementation it is identical
    # to standard_clas (uniform injection, no projection), so the configs would be
    # byte-identical and the _warn_on_identical_baselines guard would fire.
    STEERING_CONFIGS = {
        # single-stage CLAS: uniform injection, no projection
        "standard_clas":          {"use_schedule": False, "subspace_proj": False, "eta_mult": 1.0},
        # depth-scheduled injection only
        "gaussian_scheduled":     {"use_schedule": True,  "subspace_proj": False, "eta_mult": 1.0},
        # two-stage surgical method: scheduled injection + reasoning-subspace anchoring
        "mrre_two_stage":         {"use_schedule": True,  "subspace_proj": True,  "eta_mult": 1.0},
        # scheduled injection at 3x strength
        "aggressive_oversteering": {"use_schedule": True, "subspace_proj": False, "eta_mult": 3.0},
    }

    def run_suite(
        self,
        model,
        tokenizer,
        samples,
        steerer,
        steering_vectors,
        layer_ids,
        eta: float,
        apply_subspace_proj: bool = False,
        baselines=None,
        checkpoint_manager=None,
        cache_prefix: Optional[str] = None,
    ) -> BenchmarkReport:
        """Run the evaluation suite over the selected baselines.

        Parameters
        ----------
        baselines : list[str] or None
            Subset of :attr:`ALL_BASELINES` to run. Defaults to all five.
        checkpoint_manager, cache_prefix :
            If both given, each baseline's result is cached under
            ``f"{cache_prefix}::baseline::{name}"`` and reused on a later run — so a crash or a
            re-run with a different ``--baselines`` subset never recomputes a finished baseline.
        """
        _require_model(model, tokenizer)
        baselines = list(baselines) if baselines else list(self.ALL_BASELINES)
        invalid = [b for b in baselines if b not in self.ALL_BASELINES]
        if invalid:
            raise ValueError(f"Unknown baseline(s): {invalid}. Valid: {self.ALL_BASELINES}")
        logger.info("Executing benchmark suite | baselines=%s", baselines)
        results = {}

        # Each baseline runs with its own distinct steering configuration (see
        # STEERING_CONFIGS). ``apply_subspace_proj`` is an upper bound: a baseline can
        # only request projection if the suite enables it AND the steerer was built
        # with a decomposer; otherwise the knob degrades to no projection.
        def _run_one(name: str):
            if name == "no_intervention":
                return self.run_baseline(model, tokenizer, samples, "no_intervention")
            cfg = self.STEERING_CONFIGS[name]
            return self.run_steered(
                model, tokenizer, samples, steerer, name,
                steering_vectors, layer_ids,
                eta=eta * cfg["eta_mult"],
                apply_subspace_projection=cfg["subspace_proj"] and apply_subspace_proj,
                use_schedule=cfg["use_schedule"],
            )

        for name in baselines:
            key = f"{cache_prefix}::baseline::{name}" if cache_prefix else None
            if checkpoint_manager is not None and key and checkpoint_manager.has_result(key):
                results[name] = checkpoint_manager.get_result(key)
                logger.info("Baseline '%s' loaded from cache.", name)
                continue
            results[name] = _run_one(name)
            if checkpoint_manager is not None and key:
                checkpoint_manager.cache_result(key, results[name])

        # Guard: distinct steered baselines must not yield byte-identical metrics.
        # This is exactly how the mrre_two_stage == standard_clas bug slipped through,
        # so surface it loudly rather than letting it reach a paper table again.
        self._warn_on_identical_baselines(results)

        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report = BenchmarkReport(timestamp=ts, results=results)

        out_path = self.output_dir / f"benchmark_report_{ts}.json"
        report.save_json(out_path)

        return report

    @staticmethod
    def _warn_on_identical_baselines(results: Dict[str, Dict[str, float]]) -> None:
        """Log a loud warning if any two distinct baselines have identical metrics.

        Identical metrics mean the baselines did not actually run differently — the
        bug that made ``mrre_two_stage`` byte-identical to ``standard_clas``. Steered
        baselines that request subspace projection but find no decomposer wired into
        the steerer are the most common cause, so the warning names that explicitly.
        """
        names = [n for n in results if n != "no_intervention"]
        seen: Dict[tuple, str] = {}
        for name in names:
            sig = tuple(sorted((k, round(float(v), 6)) for k, v in results[name].items()))
            if sig in seen:
                logger.warning(
                    "Baseline '%s' produced metrics IDENTICAL to '%s'. The baselines "
                    "are not running differently — check STEERING_CONFIGS and that the "
                    "steerer was constructed with a decomposer so subspace projection "
                    "is effective (decomposer=None silently disables it).",
                    name, seen[sig],
                )
            else:
                seen[sig] = name
