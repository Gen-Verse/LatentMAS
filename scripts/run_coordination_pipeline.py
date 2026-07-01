"""
Latent Coordination Pipeline Runner: Multi-Agent Latent Coordination.

Usage:
    python scripts/run_coordination_pipeline.py --config configs/latent_coordination.yaml
    python scripts/run_coordination_pipeline.py --config configs/latent_coordination.yaml --stages A,B,C
    python scripts/run_coordination_pipeline.py --config configs/latent_coordination.yaml --skip-cvae-training
    python scripts/run_coordination_pipeline.py --config configs/latent_coordination.yaml --communication-mode latent
    python scripts/run_coordination_pipeline.py --config configs/latent_coordination.yaml --resume
"""

import os
# Reduce CUDA fragmentation OOMs. Must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys
import yaml
import time
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compute_scan import run_compute_scan

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------
STAGE_MAP = {
    "A": "Universal Latent Space Adapter Training",
    "B": "CVAE Topology Prior Training",
    "C": "Intent Centroid Fitting",
    "D": "Multi-Agent Orchestration & Benchmark",
    "E": "Communication Mode Ablation",
    "F": "Scalability Study (n_agents sweep)",
    "G": "Safety Evaluation (SEA-SafeguardBench)",
    "H": "Visualization & Report",
}

ALL_STAGES = list(STAGE_MAP.keys())

VALID_COMM_MODES = {"token", "latent", "hybrid"}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Latent Coordination Multi-Agent Latent Coordination pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config file (e.g. configs/latent_coordination.yaml).",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        metavar="KEY=VALUE,...",
        help=(
            "Comma-separated agent config overrides in KEY=VALUE format. "
            "Keys: orchestrator.device, reasoning_agent.model_id, etc."
        ),
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        metavar="A,B,C,...",
        help=(
            f"Comma-separated stages to run. "
            f"Available: {', '.join(f'{k}={v}' for k, v in STAGE_MAP.items())}. "
            f"Defaults to all stages."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from latest checkpoint.",
    )
    parser.add_argument(
        "--skip-cvae-training",
        action="store_true",
        default=False,
        help="Skip CVAE training (stage B). Use an existing checkpoint if available.",
    )
    parser.add_argument(
        "--communication-mode",
        type=str,
        default=None,
        choices=sorted(VALID_COMM_MODES),
        metavar="MODE",
        help="Force a single communication mode: token | latent | hybrid.",
    )
    parser.add_argument(
        "--comm-modes",
        type=str,
        default=None,
        metavar="M1,M2,...",
        help=(
            "Comma-separated benchmark eval modes to run (subset of: single_agent_baseline, "
            "token_based_mas, latent_based_mas_ours). Defaults to all. Cached per mode, so "
            "adding a mode later only runs the new one."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["auto", "hf", "vllm"],
        help="Generation backend for token-only modes (auto|hf|vllm). vLLM gated to Ampere+.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        metavar="th,my,...",
        help="Comma-separated ISO-639-1 target languages (subset of the FLORES+ benchmark set).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Override project.output_dir in the config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate config and imports without running any stage.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        raise ValueError(f"Config is empty or invalid: {config_path}")
    return cfg


def _parse_agent_overrides(agents_arg: str) -> Dict[str, str]:
    """Parse KEY=VALUE,... string into dict."""
    result: Dict[str, str] = {}
    for pair in agents_arg.split(","):
        pair = pair.strip()
        if "=" not in pair:
            raise ValueError(f"Invalid agent override (expected KEY=VALUE): '{pair}'")
        k, v = pair.split("=", 1)
        result[k.strip()] = v.strip()
    return result


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    logger = logging.getLogger(__name__)

    if args.output_dir is not None:
        cfg.setdefault("project", {})["output_dir"] = str(args.output_dir)
        logger.info("Override project.output_dir → %s", args.output_dir)

    if args.communication_mode is not None:
        cfg.setdefault("communication", {})["modes"] = [args.communication_mode]
        logger.info("Override communication.modes → [%s]", args.communication_mode)

    if args.agents is not None:
        try:
            overrides = _parse_agent_overrides(args.agents)
        except ValueError as exc:
            raise ValueError(f"--agents parse error: {exc}") from exc
        # Apply overrides: match agent by id prefix
        agents_list: List[Dict[str, Any]] = cfg.get("agents", [])
        for key, val in overrides.items():
            parts = key.split(".", 1)
            agent_id, field = (parts[0], parts[1]) if len(parts) == 2 else (parts[0], None)
            for agent in agents_list:
                if agent.get("id") == agent_id and field:
                    agent[field] = val
                    logger.info("Override agent '%s'.%s → %s", agent_id, field, val)

    if args.skip_cvae_training:
        cfg.setdefault("cvae", {}).setdefault("training", {})["skip"] = True
        logger.info("CVAE training will be skipped (--skip-cvae-training).")

    if args.comm_modes is not None:
        modes = [m.strip() for m in args.comm_modes.split(",") if m.strip()]
        cfg.setdefault("communication", {})["eval_modes"] = modes
        logger.info("Override communication.eval_modes → %s", modes)

    if args.backend is not None:
        cfg.setdefault("communication", {})["backend"] = args.backend
        logger.info("Override communication.backend → %s", args.backend)

    if args.languages is not None:
        langs = [l.strip() for l in args.languages.split(",") if l.strip()]
        cfg["target_languages"] = langs
        logger.info("Override target_languages → %s", langs)

    return cfg


def resolve_stages(stages_arg: Optional[str], skip_cvae: bool) -> List[str]:
    if stages_arg is None:
        stages = ALL_STAGES.copy()
    else:
        stages = [s.strip().upper() for s in stages_arg.split(",")]
        invalid = [s for s in stages if s not in STAGE_MAP]
        if invalid:
            raise ValueError(f"Unknown stage(s): {invalid}. Valid: {ALL_STAGES}")

    if skip_cvae and "B" in stages:
        stages.remove("B")
        logging.getLogger(__name__).info(
            "--skip-cvae-training: stage B removed from run list."
        )
    return stages


# ---------------------------------------------------------------------------
# Bootstrap logging
# ---------------------------------------------------------------------------

def _bootstrap_logging(log_dir: Optional[str], level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(Path(log_dir) / f"coordination_pipeline_{ts}.log")
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        handlers.append(fh)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------

def _build_run_summary(
    cfg: dict,
    stages: List[str],
    start_time: float,
    end_time: float,
    success: bool,
    error: Optional[str] = None,
) -> dict:
    return {
        "pipeline": "latent_coordination",
        "version": __version__,
        "timestamp_utc": datetime.utcfromtimestamp(start_time).isoformat(),
        "elapsed_seconds": round(end_time - start_time, 2),
        "stages_requested": stages,
        "config_project": cfg.get("project", {}),
        "n_agents": len(cfg.get("agents", [])),
        "communication_modes": cfg.get("communication", {}).get("modes", []),
        "success": success,
        "error": error,
    }


def save_run_summary(summary: dict, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary_path = out / f"run_summary_{ts}.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    logging.getLogger(__name__).info("Run summary saved → %s", summary_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    run_compute_scan(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "compute_scan.json"))
    args = parse_args()

    cfg = load_config(args.config)

    log_cfg = cfg.get("logging", {})
    _bootstrap_logging(
        log_dir=log_cfg.get("log_dir"),
        level=log_cfg.get("level", "INFO"),
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("Latent Coordination Pipeline Runner  v%s", __version__)
    logger.info("Config : %s", args.config.resolve())
    logger.info("=" * 70)

    # Apply overrides
    try:
        cfg = apply_overrides(cfg, args)
    except ValueError as exc:
        logger.error("Config override error: %s", exc)
        return 1

    # Resolve stages
    try:
        stages = resolve_stages(args.stages, args.skip_cvae_training)
    except ValueError as exc:
        logger.error("Stage resolution error: %s", exc)
        return 1

    logger.info("Stages to run: %s", stages)
    logger.info("Stage descriptions: %s", {s: STAGE_MAP[s] for s in stages})

    # Dry-run
    if args.dry_run:
        logger.info("[DRY-RUN] Config validation passed. Checking imports…")
        try:
            from latent_coordination.pipeline.coordination_pipeline import CoordinationPipeline  # noqa: F401
            logger.info("[DRY-RUN] Imports OK. Exiting without running pipeline.")
        except ImportError as exc:
            logger.warning("[DRY-RUN] Import warning (may be OK in dev): %s", exc)
        return 0

    # Set up full logging
    try:
        from latent_coordination.utils.logging_utils import setup_logging
        setup_logging(cfg)
        logger = logging.getLogger(__name__)
    except ImportError:
        logger.warning(
            "latent_coordination.utils.logging_utils not found; using bootstrap logger."
        )

    # Import pipeline
    try:
        from latent_coordination.pipeline.coordination_pipeline import CoordinationPipeline
    except ImportError as exc:
        logger.error("Failed to import CoordinationPipeline: %s", exc)
        logger.debug(traceback.format_exc())
        return 1

    output_dir = cfg.get("project", {}).get("output_dir", "results/coordination")
    start_time = time.time()
    success = False
    error_msg: Optional[str] = None

    try:
        pipeline = CoordinationPipeline(config=cfg, resume=args.resume)
        logger.info("Pipeline instantiated. Starting execution…")
        pipeline.run(stages=stages)
        success = True
        logger.info("Pipeline completed successfully.")

    except KeyboardInterrupt:
        error_msg = "Interrupted by user (KeyboardInterrupt)."
        logger.warning(error_msg)

    except Exception as exc:  # pylint: disable=broad-except
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Pipeline failed: %s", error_msg)
        logger.debug(traceback.format_exc())

    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info("Total elapsed time: %.2f s (%.1f min)", elapsed, elapsed / 60)

        summary = _build_run_summary(
            cfg=cfg,
            stages=stages,
            start_time=start_time,
            end_time=end_time,
            success=success,
            error=error_msg,
        )
        try:
            save_run_summary(summary, output_dir)
        except Exception as summ_exc:  # pylint: disable=broad-except
            logger.warning("Could not save run summary: %s", summ_exc)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
