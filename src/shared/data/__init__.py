"""Shared dataset loading infra (benchmark loaders + contrastive lexicon).

Non-technical shared infrastructure per strategy.md §3.4: consumed by BOTH
``latent_coordination`` and ``mechanistic_disentangle`` without either
importing the other.
"""

from shared.data.dataset_loader import DatasetLoader, Sample
from shared.data.lexicon import ContrastiveLexicon

__all__ = ["DatasetLoader", "Sample", "ContrastiveLexicon"]
