"""
Checkpoint management for multi-stage research pipelines.

Provides save/load/exists semantics for arbitrary Python objects using
torch.save, with human-readable metadata alongside each checkpoint.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Save and load pipeline-stage checkpoints to/from disk.

    Each checkpoint is stored as a ``.pt`` file under
    ``{checkpoint_dir}/{project_name}/{stage_name}_latest.pt``.
    A companion ``.json`` metadata file records the timestamp and stage.

    Args:
        checkpoint_dir: Root directory for all checkpoints.
        project_name: Sub-directory name (e.g. ``"mechanistic"`` or ``"coordination"``).
    """

    def __init__(self, checkpoint_dir: str | Path, project_name: str = "project") -> None:
        self.root = Path(checkpoint_dir) / project_name
        self.root.mkdir(parents=True, exist_ok=True)
        logger.info("CheckpointManager: root=%s", self.root)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def save(self, obj: Any, stage_name: str) -> Path:
        """Persist *obj* under *stage_name*.

        Args:
            obj: Any Python / PyTorch-serialisable object.
            stage_name: Identifier for this pipeline stage (e.g. ``"stage_a"``).

        Returns:
            Path to the saved ``.pt`` file.
        """
        ckpt_path = self.root / f"{stage_name}_latest.pt"
        # Atomic write: serialise to a temp file then rename, so an interruption mid-write
        # can never leave a half-written checkpoint that breaks a later --resume.
        tmp_path = ckpt_path.with_suffix(".pt.tmp")
        torch.save(obj, tmp_path)
        os.replace(tmp_path, ckpt_path)

        meta = {
            "stage": stage_name,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "path": str(ckpt_path),
        }
        meta_path = self.root / f"{stage_name}_latest.json"
        tmp_meta = meta_path.with_suffix(".json.tmp")
        with open(tmp_meta, "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp_meta, meta_path)

        logger.info("Checkpoint saved | stage=%s path=%s", stage_name, ckpt_path)
        return ckpt_path

    def load_latest(self, stage_name: str) -> Any:
        """Load the latest checkpoint for *stage_name*.

        Args:
            stage_name: Stage identifier matching a prior :meth:`save` call.

        Returns:
            The deserialized object.

        Raises:
            FileNotFoundError: If no checkpoint exists for *stage_name*.
        """
        ckpt_path = self.root / f"{stage_name}_latest.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found for stage '{stage_name}' at {ckpt_path}"
            )
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        logger.info("Checkpoint loaded | stage=%s path=%s", stage_name, ckpt_path)
        return obj

    def exists(self, stage_name: str) -> bool:
        """Return True if a checkpoint for *stage_name* exists on disk."""
        return (self.root / f"{stage_name}_latest.pt").exists()

    def delete(self, stage_name: str) -> None:
        """Remove the checkpoint files for *stage_name* (both .pt and .json)."""
        for suffix in (".pt", ".json"):
            p = self.root / f"{stage_name}_latest{suffix}"
            if p.exists():
                p.unlink()
                logger.info("Deleted checkpoint file: %s", p)

    def list_stages(self):
        """Return the list of stage names that have saved checkpoints."""
        return sorted(
            p.stem.replace("_latest", "")
            for p in self.root.glob("*_latest.pt")
        )

    # ------------------------------------------------------------------
    # Fine-grained keyed result cache
    # ------------------------------------------------------------------
    # Used for per-unit work (e.g. a single (model, benchmark, baseline) result or a single
    # (model, comm_mode) result) so that re-running a *different* combination — or recovering
    # from a crash — reuses everything already computed. Keys are arbitrary stable strings;
    # they are slugified for the filename so "/" and "::" are safe.

    _CACHE_SUBDIR = "_results"

    @staticmethod
    def _slug(key: str) -> str:
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in key)
        return safe[:200]

    def _cache_path(self, key: str) -> Path:
        d = self.root / self._CACHE_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self._slug(key)}.pt"

    def cache_result(self, key: str, obj: Any) -> Path:
        """Atomically persist a per-unit result under *key*."""
        path = self._cache_path(key)
        tmp = path.with_suffix(".pt.tmp")
        torch.save({"key": key, "obj": obj}, tmp)
        os.replace(tmp, path)
        logger.info("Cached result | key=%s", key)
        return path

    def has_result(self, key: str) -> bool:
        """Return True if a cached result exists for *key*."""
        return self._cache_path(key).exists()

    def get_result(self, key: str) -> Any:
        """Load a cached per-unit result. Raises FileNotFoundError if absent."""
        path = self._cache_path(key)
        if not path.exists():
            raise FileNotFoundError(f"No cached result for key '{key}' at {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        logger.info("Loaded cached result | key=%s", key)
        return payload["obj"]
