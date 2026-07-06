"""Staircase ablation runner (LRL-MRRE-MAS strategy.md §7.3).

Config-driven realisation of the corrected staircase table: each row is the
base YAML config plus a set of dotted-path overrides that toggle the REAL
module knobs the pipeline already honors (nothing here fabricates results —
every row is a full ``CoordinationPipeline`` run):

=====  ======================  ==================  ============================
Row    Name                    Modules active      Real knobs toggled
=====  ======================  ==================  ============================
0      proxy_baseline          none                FLORES+ only → accuracy is
                                                   the completeness proxy
1      rescored_baseline       none                + mgsm/belebele enabled →
                                                   real correctness metric
2      hub_regularized         A+B                 adapter_training.enabled
3      geometry_routed         A+B+D               + routing_strategy=
                                                   cvae_topology,
                                                   condition_on_geometry
4      latent_reasoned         A+B+D+C             + latent_reasoning.enabled
5      closed_loop_full        A+B+D+C+E           + verification.enabled
6      verifier_disabled       A+B+D+C (E off)     row-5 config minus E
7a_*   loss-term split         A+B variants        mu_cka / gamma_dae zeroed
=====  ======================  ==================  ============================

Audit-required additions (strategy.md §7.3):

* **7a** — intra-Module-A+B loss split: with ``L_adapt = L_recon + μ·L_CKA +
  γ·L_DAE``, L_recon is the base objective, so the realisable split is
  recon-only / recon+CKA / recon+DAE / full (row 2).
* **7b** — the OneFlow single-agent baseline is an explicit column of EVERY
  row: ``single_agent_baseline`` stays in ``communication.eval_modes``
  throughout, never reported separately.
* **7d/7e** — Geo_L dimensionality and drift-probe linear-vs-MLP rows are
  expressible as ``ablation.extra_rows`` entries in the YAML (override
  ``cvae.geo_profile_path`` to a raw-65 artifact / ``verification.probe_arch:
  mlp``) once the corresponding artifact exists; see configs/*.yaml.

Rows 3+ require ``cvae.geo_profile_path`` in the base config (export with
``scripts/export_geo_profiles.py``) — the pipeline fails fast otherwise.

Each row runs in its own output/checkpoint directory: the Stage-E result cache
key does not (by design) encode module toggles, so sharing a checkpoint dir
across rows would silently reuse another row's cached mode results.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger(__name__)

# Every row keeps single_agent_baseline in eval_modes (addition 7b: the
# OneFlow baseline is an explicit row of the staircase table, always run
# alongside, never reported separately).
_BASELINE_MODES = ["single_agent_baseline", "token_based_mas"]
_ALL_MODES = ["single_agent_baseline", "token_based_mas", "latent_based_mas_ours"]

# Module-toggle building blocks (dotted config paths the pipeline honors).
_CORRECTNESS_OFF = {
    "benchmarks.mgsm.enabled": False,
    "benchmarks.belebele.enabled": False,
    "benchmarks.sea_vision.enabled": False,
    "benchmarks.sea_safeguardbench.enabled": False,
}
_CORRECTNESS_ON = {
    "benchmarks.mgsm.enabled": True,
    "benchmarks.belebele.enabled": True,
}
_MODULES_AB = {"universal_latent_space.adapter_training.enabled": True}
_MODULE_D = {
    "orchestration.routing_strategy": "cvae_topology",
    "cvae.condition_on_geometry": True,
}
_MODULE_C = {"latent_reasoning.enabled": True}
_MODULE_E = {"verification.enabled": True}


@dataclass
class AblationRow:
    """One staircase configuration: a named set of config overrides."""

    row_id: str
    name: str
    modules: str
    isolates: str
    overrides: Dict[str, object] = field(default_factory=dict)


def _row(row_id, name, modules, isolates, *override_dicts, modes=None) -> AblationRow:
    overrides: Dict[str, object] = {}
    for d in override_dicts:
        overrides.update(d)
    overrides["communication.eval_modes"] = list(modes or _ALL_MODES)
    # Rows are correctness-scored from row 1 onward; row 0 is the historical
    # completeness proxy.
    return AblationRow(row_id=row_id, name=name, modules=modules,
                       isolates=isolates, overrides=overrides)


STAIRCASE_ROWS: List[AblationRow] = [
    _row("0", "proxy_baseline", "none",
         "starting point (old completeness proxy)",
         _CORRECTNESS_OFF,
         {"universal_latent_space.adapter_training.enabled": False,
          "latent_reasoning.enabled": False,
          "verification.enabled": False,
          "orchestration.routing_strategy": "attention"},
         modes=_BASELINE_MODES),
    _row("1", "rescored_baseline", "none",
         "metric-change effect only (correctness metric applied)",
         _CORRECTNESS_ON,
         {"universal_latent_space.adapter_training.enabled": False,
          "latent_reasoning.enabled": False,
          "verification.enabled": False,
          "orchestration.routing_strategy": "attention"},
         modes=_BASELINE_MODES),
    _row("2", "hub_regularized", "A+B",
         "interlingua CKA-DAE hub value",
         _CORRECTNESS_ON, _MODULES_AB,
         {"latent_reasoning.enabled": False,
          "verification.enabled": False,
          "orchestration.routing_strategy": "attention"}),
    _row("3", "geometry_routed", "A+B+D",
         "geometry-conditioned CVAE routing value",
         _CORRECTNESS_ON, _MODULES_AB, _MODULE_D,
         {"latent_reasoning.enabled": False, "verification.enabled": False}),
    _row("4", "latent_reasoned", "A+B+D+C",
         "latent recurrent reasoning value",
         _CORRECTNESS_ON, _MODULES_AB, _MODULE_D, _MODULE_C,
         {"verification.enabled": False}),
    _row("5", "closed_loop_full", "A+B+D+C+E",
         "drift-verification + re-plan value (full system)",
         _CORRECTNESS_ON, _MODULES_AB, _MODULE_D, _MODULE_C, _MODULE_E),
    _row("6", "verifier_disabled", "A+B+D+C (E off)",
         "marginal utility of E vs row 5",
         _CORRECTNESS_ON, _MODULES_AB, _MODULE_D, _MODULE_C,
         {"verification.enabled": False}),
    # 7a — intra-Module-A+B loss-term split (recon is the base term; the
    # full combination is row 2 itself).
    _row("7a_recon_only", "ab_split_recon_only", "A+B (L_recon)",
         "reconstruction term alone",
         _CORRECTNESS_ON, _MODULES_AB,
         {"universal_latent_space.adapter_training.mu_cka": 0.0,
          "universal_latent_space.adapter_training.gamma_dae": 0.0,
          "latent_reasoning.enabled": False, "verification.enabled": False,
          "orchestration.routing_strategy": "attention"}),
    _row("7a_recon_cka", "ab_split_recon_cka", "A+B (L_recon + L_CKA)",
         "CKA alignment term's marginal value",
         _CORRECTNESS_ON, _MODULES_AB,
         {"universal_latent_space.adapter_training.gamma_dae": 0.0,
          "latent_reasoning.enabled": False, "verification.enabled": False,
          "orchestration.routing_strategy": "attention"}),
    _row("7a_recon_dae", "ab_split_recon_dae", "A+B (L_recon + L_DAE)",
         "DAE robustness term's marginal value",
         _CORRECTNESS_ON, _MODULES_AB,
         {"universal_latent_space.adapter_training.mu_cka": 0.0,
          "latent_reasoning.enabled": False, "verification.enabled": False,
          "orchestration.routing_strategy": "attention"}),
]


def set_dotted(cfg: Dict, dotted_key: str, value) -> None:
    """Set ``cfg['a']['b']['c'] = value`` for ``dotted_key='a.b.c'``, creating
    intermediate dicts. Fails loudly if a segment exists but is not a dict."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        child = node.get(part)
        if child is None:
            child = node[part] = {}
        elif not isinstance(child, dict):
            raise TypeError(
                f"Cannot apply override '{dotted_key}': config node '{part}' is "
                f"{type(child).__name__}, not a mapping."
            )
        node = child
    node[parts[-1]] = value


def derive_row_config(base_cfg: Dict, row: AblationRow, out_root: Path | str) -> Dict:
    """Base config + row overrides + per-row isolated output/checkpoint dirs."""
    cfg = copy.deepcopy(base_cfg)
    for key, value in row.overrides.items():
        set_dotted(cfg, key, value)
    row_dir = Path(out_root) / f"row_{row.row_id}_{row.name}"
    set_dotted(cfg, "project.output_dir", str(row_dir / "results"))
    # Isolation is mandatory: Stage E's result cache key encodes models +
    # languages + sample cap but NOT module toggles, so two rows sharing a
    # checkpoint dir would silently reuse each other's cached mode results.
    set_dotted(cfg, "checkpointing.checkpoint_dir", str(row_dir / "checkpoints"))
    return cfg


def load_extra_rows(base_cfg: Dict) -> List[AblationRow]:
    """Rows 7d/7e-style additions from ``ablation.extra_rows`` in the YAML.

    Each entry: ``{name: ..., overrides: {dotted.key: value, ...}}``; rows
    inherit nothing implicit — spell out every toggle that differs from the
    base config.
    """
    rows = []
    for i, entry in enumerate((base_cfg.get("ablation", {}) or {}).get("extra_rows") or []):
        if not isinstance(entry, dict) or "name" not in entry or "overrides" not in entry:
            raise ValueError(
                f"ablation.extra_rows[{i}] must be a mapping with 'name' and "
                f"'overrides' keys; got: {entry!r}"
            )
        rows.append(AblationRow(
            row_id=f"x{i}", name=str(entry["name"]), modules="custom",
            isolates=str(entry.get("isolates", "custom row")),
            overrides=dict(entry["overrides"]),
        ))
    return rows


def select_rows(names: Optional[List[str]], extra: List[AblationRow]) -> List[AblationRow]:
    """Resolve a ``--rows`` selection (by row_id or name) against all rows."""
    all_rows = STAIRCASE_ROWS + extra
    if not names:
        return all_rows
    by_key = {}
    for r in all_rows:
        by_key[r.row_id] = r
        by_key[r.name] = r
    missing = [n for n in names if n not in by_key]
    if missing:
        raise ValueError(
            f"Unknown staircase row(s): {missing}. "
            f"Valid: {[r.row_id for r in all_rows]} or {[r.name for r in all_rows]}."
        )
    seen, selected = set(), []
    for n in names:
        r = by_key[n]
        if r.row_id not in seen:
            seen.add(r.row_id)
            selected.append(r)
    return selected


def run_staircase(
    base_cfg: Dict,
    rows: List[AblationRow],
    out_root: Path | str,
    stages: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict:
    """Run the selected rows end-to-end and write the consolidated artifact.

    With ``dry_run=True`` no pipeline is instantiated; the consolidated dict
    contains each row's derived config instead of results (use this to audit
    exactly what would run before spending GPU-days).
    """
    out_root = Path(out_root)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    consolidated: Dict = {
        "timestamp": ts,
        "dry_run": dry_run,
        "stages": stages or "all",
        "rows": {},
    }

    for row in rows:
        cfg = derive_row_config(base_cfg, row, out_root)
        entry: Dict = {
            "row_id": row.row_id,
            "name": row.name,
            "modules": row.modules,
            "isolates": row.isolates,
            "overrides": row.overrides,
        }
        if dry_run:
            entry["derived_config"] = cfg
        else:
            from latent_coordination.pipeline.coordination_pipeline import (
                CoordinationPipeline,
            )
            logger.info("=== Staircase row %s (%s): modules=%s ===",
                        row.row_id, row.name, row.modules)
            pipeline = CoordinationPipeline(cfg)
            final_report = pipeline.run(stages=stages)
            entry["results_by_mode"] = (
                (final_report.get("results") or {}).get("results_by_mode")
                if isinstance(final_report, dict) else None
            ) or (final_report or {}).get("results_by_mode", {})
            entry["headline_framing"] = (final_report or {}).get("headline_framing")
        consolidated["rows"][row.row_id] = entry

    out_root.mkdir(parents=True, exist_ok=True)
    suffix = "dryrun" if dry_run else "results"
    out_path = out_root / f"staircase_{ts}_{suffix}.json"
    from shared.serialization import to_json_safe
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(consolidated), f, indent=2, ensure_ascii=False)
    consolidated["artifact_path"] = str(out_path)
    logger.info("Consolidated staircase artifact written to %s", out_path)
    return consolidated
