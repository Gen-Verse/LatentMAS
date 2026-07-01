"""Run provenance manifest — G6 global standard.

Every experimental run should write a manifest so results can be reproduced
exactly from: git SHA, config hash, seed, library versions, compute profile,
and W&B run ID.

Usage
-----
    from shared.manifest import write_manifest

    write_manifest(
        run_dir=Path("results/surgical/20260629T120000Z"),
        config={"model_id": "...", "seed": 42, ...},
        seed=42,
        wandb_run_id="abc123",          # optional
        compute_profile_path=Path("infra/compute_profile.json"),  # optional
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


logger = logging.getLogger(__name__)

# Library names whose versions are tracked in every manifest.
_TRACKED_LIBS = [
    "transformers", "accelerate", "bitsandbytes", "sacrebleu",
    "torch", "numpy", "datasets", "hydra_core", "wandb",
    "unbabel_comet", "sentence_transformers", "lm_eval",
]


def _git_sha() -> str:
    """Return the current git HEAD SHA, or 'unknown' if not in a git repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return sha + ("-dirty" if dirty else "")
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _lib_versions() -> Dict[str, str]:
    """Return installed version strings for the tracked libraries."""
    versions: Dict[str, str] = {}
    for lib in _TRACKED_LIBS:
        try:
            import importlib.metadata
            versions[lib] = importlib.metadata.version(lib)
        except Exception:
            versions[lib] = "not_installed"
    versions["python"] = sys.version
    return versions


def _config_hash(config: Any) -> str:
    """MD5 hash of the JSON-serialised config (for cache-key construction)."""
    try:
        serialised = json.dumps(config, sort_keys=True, default=str).encode()
    except (TypeError, ValueError):
        serialised = str(config).encode()
    return hashlib.md5(serialised).hexdigest()[:12]


def build_manifest(
    config: Any,
    seed: int,
    wandb_run_id: Optional[str] = None,
    compute_profile_path: Optional[Path] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    """Build the provenance manifest dict (does not write to disk)."""
    compute_profile: Dict = {}
    if compute_profile_path is not None:
        try:
            with open(compute_profile_path) as f:
                compute_profile = json.load(f)
        except Exception as exc:
            logger.warning("Could not load compute profile from %s: %s", compute_profile_path, exc)

    return {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "config_hash": _config_hash(config),
        "seed": seed,
        "library_versions": _lib_versions(),
        "wandb_run_id": wandb_run_id,
        "compute_profile": compute_profile,
        **(extra or {}),
    }


def write_manifest(
    run_dir: Path,
    config: Any,
    seed: int,
    wandb_run_id: Optional[str] = None,
    compute_profile_path: Optional[Path] = None,
    extra: Optional[Dict] = None,
    filename: str = "manifest.json",
) -> Path:
    """Build and write the manifest JSON to ``run_dir / filename``.

    Parameters
    ----------
    run_dir
        Directory where the run's outputs are stored.
    config
        The run configuration (dict, dataclass, or any JSON-serialisable object).
    seed
        Global RNG seed used for this run.
    wandb_run_id
        W&B run ID string (from ``wandb.run.id``), or None if W&B not used.
    compute_profile_path
        Path to the hardware profile JSON written by ``infra/compute_profile.py``.
    extra
        Any additional provenance fields to include.
    filename
        JSON filename within ``run_dir`` (default ``manifest.json``).

    Returns
    -------
    Path to the written manifest file.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(config, seed, wandb_run_id, compute_profile_path, extra)

    # Serialise config — convert dataclass / arbitrary objects to dict.
    try:
        from dataclasses import asdict
        manifest["config"] = asdict(config) if hasattr(config, "__dataclass_fields__") else config
    except Exception:
        manifest["config"] = str(config)

    out_path = run_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Manifest written → %s (git=%s, config_hash=%s)",
                out_path, manifest["git_sha"][:12], manifest["config_hash"])
    return out_path
