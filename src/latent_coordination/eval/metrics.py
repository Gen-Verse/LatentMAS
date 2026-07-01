"""Metrics computation suite for the Mechanistic Disentanglement evaluation pipeline."""

import logging
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union

import torch
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


@dataclass
class MetricsSuite:
    """Wrapper dataclass for all computed metrics."""
    accuracy: float
    perplexity: float
    bleu: Dict[str, float]
    rouge: Dict[str, float]
    language_accuracy: float
    ifl_rate: float

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([self.to_dict()])


class MetricsComputer:
    """Computes various text generation quality metrics."""

    @staticmethod
    def compute_accuracy(predictions: List[str], references: List[str]) -> float:
        """Compute exact match or substring containment accuracy."""
        if not predictions or not references:
            return 0.0
        correct = 0
        for p, r in zip(predictions, references):
            if r.strip().lower() in p.strip().lower():
                correct += 1
        return correct / len(predictions)

    @staticmethod
    def compute_perplexity(model, tokenizer, texts: List[str], batch_size: int = 4) -> float:
        """Compute average per-token perplexity of generated texts."""
        if not texts:
            return 0.0
        device = next(model.parameters()).device
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch = [text for text in batch if text.strip()]
            if not batch:
                continue

            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss  # average negative log likelihood per token
                # count tokens actually evaluated (excluding padding)
                n_tokens = attention_mask.sum().item()
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens

        mean_loss = total_loss / (total_tokens + 1e-12)
        try:
            return math.exp(mean_loss)
        except OverflowError:
            return float("inf")

    @staticmethod
    def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BLEU with sacrebleu (required; raises if not installed)."""
        try:
            import sacrebleu
            bleu = sacrebleu.corpus_bleu(predictions, [references])
            return {
                "bleu_1": bleu.precisions[0] if len(bleu.precisions) > 0 else 0.0,
                "bleu_2": bleu.precisions[1] if len(bleu.precisions) > 1 else 0.0,
                "bleu_3": bleu.precisions[2] if len(bleu.precisions) > 2 else 0.0,
                "bleu_4": bleu.score,
            }
        except ImportError as exc:
            raise RuntimeError(
                "sacrebleu is required for BLEU evaluation. Install with: pip install sacrebleu"
            ) from exc

    @staticmethod
    def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE with rouge-score (required; raises if not installed)."""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            r1s, r2s, rls = [], [], []
            for p, r in zip(predictions, references):
                scores = scorer.score(r, p)
                r1s.append(scores["rouge1"].fmeasure)
                r2s.append(scores["rouge2"].fmeasure)
                rls.append(scores["rougeL"].fmeasure)
            return {
                "rouge1": float(np.mean(r1s)),
                "rouge2": float(np.mean(r2s)),
                "rougeL": float(np.mean(rls)),
            }
        except ImportError as exc:
            raise RuntimeError(
                "rouge-score is required for ROUGE evaluation. Install with: pip install rouge-score"
            ) from exc

    @staticmethod
    def compute_language_accuracy(predictions: List[str], target_lang: str) -> float:
        """Determine what fraction of predictions match the target language."""
        from latent_coordination.eval.script_fidelity import ScriptFidelityEvaluator
        evaluator = ScriptFidelityEvaluator()
        sfrs = [evaluator.compute_sfr(p, target_lang) for p in predictions]
        # Threshold: if > 50% of characters are in target script, count as correct lang
        return float(np.mean([1.0 if s >= 0.5 else 0.0 for s in sfrs]))

    @staticmethod
    def compute_ifl_rate(generated_texts: List[str], target_language: str) -> float:
        """Compute Incorrect Format / Language (IFL) rate.

        Fraction of outputs that are NOT primarily in the target language's script.
        """
        acc = MetricsComputer.compute_language_accuracy(generated_texts, target_language)
        return 1.0 - acc

    @classmethod
    def compute_suite(
        cls,
        predictions: List[str],
        references: List[str],
        target_language: str,
        model=None,
        tokenizer=None,
    ) -> MetricsSuite:
        """Run all metrics and return a consolidated suite."""
        acc = cls.compute_accuracy(predictions, references)
        ppl = cls.compute_perplexity(model, tokenizer, predictions) if model and tokenizer else 0.0
        bleu = cls.compute_bleu(predictions, references)
        rouge = cls.compute_rouge(predictions, references)
        lang_acc = cls.compute_language_accuracy(predictions, target_language)
        ifl = cls.compute_ifl_rate(predictions, target_language)

        return MetricsSuite(
            accuracy=acc,
            perplexity=ppl,
            bleu=bleu,
            rouge=rouge,
            language_accuracy=lang_acc,
            ifl_rate=ifl,
        )
