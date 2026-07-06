"""Geo_L: precomputed per-language mechanistic risk profiles for Module D.

Strategy.md §4.2: the CVAE topology prior conditions on ``x = [q ‖ Geo_L]``,
where ``Geo_L`` is a COMPRESSED summary vector (3–8 scalars per language, not a
raw 65-dim concatenation — a raw vector fit on 6–14 languages risks CVAE
posterior collapse).

Firewall (strategy.md §6 Rules 1+3): this module only *consumes* a precomputed
artifact. All SVD/CLAP/Logit-Lens math that produces the numbers lives in
``mechanistic_disentangle`` — see ``scripts/export_geo_profiles.py`` — and the
artifact crosses the firewall as plain JSON data. Nothing here computes
decompositions or projections.

Artifact format (JSON):

    {
      "feature_names": ["english_mass_auc", "clap_dealignment_slope", ...],
      "profiles": {
        "th": [0.42, -0.13, 0.88],
        "lo": [0.61, -0.02, 0.95],
        ...
      }
    }

All profile vectors must share the artifact's ``feature_names`` length.
Zero-fallback policy: a missing artifact or an unknown language raises — no
silent zero vectors, which would train the prior to ignore its conditioning.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger(__name__)

# Compressed-summary bounds per strategy.md §4.2.
MIN_GEO_DIM = 3
MAX_GEO_DIM = 8


class GeoProfile:
    """Loader/lookup for precomputed per-language Geo_L summary vectors.

    Args:
        artifact_path: Path to the JSON artifact (see module docstring).
        strict_dim: When True (default), enforce the 3–8 dim compressed-summary
            bound from the strategy audit. Pass False only for the explicit
            raw-dimensionality ablation (strategy.md §7 item 7d).
    """

    def __init__(self, artifact_path: str | Path, strict_dim: bool = True) -> None:
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Geo_L artifact not found: {path}. Geometry conditioning requires a "
                "precomputed per-language profile — produce one with "
                "scripts/export_geo_profiles.py (mechanistic_disentangle side) or "
                "disable cvae.condition_on_geometry. No zero-vector fallback exists "
                "by design (it would train the prior to ignore its conditioning)."
            )
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        if "profiles" not in raw or not raw["profiles"]:
            raise ValueError(f"Geo_L artifact {path} has no 'profiles' entries.")
        self.feature_names: List[str] = list(raw.get("feature_names", []))
        self._profiles: Dict[str, Tensor] = {}

        dims = set()
        for lang, vec in raw["profiles"].items():
            t = torch.tensor(vec, dtype=torch.float32)
            if t.dim() != 1:
                raise ValueError(f"Geo_L profile for '{lang}' must be a flat vector.")
            dims.add(t.numel())
            self._profiles[lang.lower()] = t
        if len(dims) != 1:
            raise ValueError(
                f"Geo_L profiles have inconsistent dimensionality across languages: {sorted(dims)}."
            )
        self.geo_dim = dims.pop()
        if self.feature_names and len(self.feature_names) != self.geo_dim:
            raise ValueError(
                f"feature_names has {len(self.feature_names)} entries but profiles are "
                f"{self.geo_dim}-dimensional."
            )
        if strict_dim and not (MIN_GEO_DIM <= self.geo_dim <= MAX_GEO_DIM):
            raise ValueError(
                f"Geo_L must be a compressed summary of {MIN_GEO_DIM}–{MAX_GEO_DIM} dims "
                f"(strategy.md §4.2); got {self.geo_dim}. Pass strict_dim=False only for "
                "the raw-dimensionality ablation."
            )
        logger.info(
            "GeoProfile loaded: %d languages x %d dims from %s",
            len(self._profiles), self.geo_dim, path,
        )

    @property
    def languages(self) -> List[str]:
        return sorted(self._profiles)

    def __contains__(self, language: str) -> bool:
        return language.lower() in self._profiles

    def vector(self, language: str) -> Tensor:
        """Return the Geo_L vector for a language (raises on unknown)."""
        key = language.lower()
        if key not in self._profiles:
            raise KeyError(
                f"No Geo_L profile for language '{language}'. Available: "
                f"{self.languages}. Re-export the artifact with this language included "
                "rather than substituting a fabricated vector."
            )
        return self._profiles[key].clone()

    def batch(self, languages: List[str]) -> Tensor:
        """Stack Geo_L vectors for a list of languages → (B, geo_dim)."""
        return torch.stack([self.vector(lang) for lang in languages])
