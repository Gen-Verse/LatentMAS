"""
Mechanistic Disentanglement Pipeline Runner: Lexicon, Extraction, SVD, Isomorphism,
Steering, Benchmark, Viz, Report.

Usage:
    python scripts/run_mechanistic_pipeline.py --config configs/mechanistic_smoketest.yaml
    python scripts/run_mechanistic_pipeline.py --config configs/mechanistic_smoketest.yaml --stages A,B,C
    python scripts/run_mechanistic_pipeline.py --config configs/mechanistic_smoketest.yaml --model aisingapore/Llama-SEA-LION-v3-8B-IT --device cuda:0
    python scripts/run_mechanistic_pipeline.py --config configs/mechanistic_smoketest.yaml --resume
    python scripts/run_mechanistic_pipeline.py --config configs/mechanistic_smoketest.yaml --dry-run
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
from typing import Optional, List

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


STAGE_MAP = {
    "A": "Lexicon + Data Loading",
    "B": "Activation Extraction",
    "C": "SVD Subspace Decomposition",
    "D": "Isomorphism Analysis (CKA/Procrustes/RSA)",
    "E": "Steering Vector Building",
    "F": "Benchmark Evaluation",
    "G": "Visualizations",
    "H": "Final Report",
}
ALL_STAGES = list(STAGE_MAP.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Mechanistic Disentanglement pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument("--model", type=str, default=None, help="Override model.model_id in the config.")
    parser.add_argument("--device", type=str, default=None, help="Override model.device in the config.")
    parser.add_argument(
        "--stages", type=str, default=None, metavar="A,B,C,...",
        help=f"Comma-separated stages to run. Available: {', '.join(f'{k}={v}' for k, v in STAGE_MAP.items())}. Defaults to all.",
    )
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from latest checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=None, metavar="DIR", help="Override project.output_dir.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Validate config and imports without running any stage.")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        raise ValueError(f"Config is empty or invalid: {config_path}")
    return cfg


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    logger = logging.getLogger(__name__)
    if args.model is not None:
        cfg.setdefault("model", {})["model_id"] = args.model
        logger.info("Override model.model_id -> %s", args.model)
    if args.device is not None:
        cfg.setdefault("model", {})["device"] = args.device
        logger.info("Override model.device -> %s", args.device)
    if args.output_dir is not None:
        cfg.setdefault("project", {})["output_dir"] = str(args.output_dir)
        logger.info("Override project.output_dir -> %s", args.output_dir)
    return cfg


def resolve_stages(stages_arg: Optional[str]) -> List[str]:
    if stages_arg is None:
        return ALL_STAGES.copy()
    stages = [s.strip().upper() for s in stages_arg.split(",")]
    invalid = [s for s in stages if s not in STAGE_MAP]
    if invalid:
        raise ValueError(f"Unknown stage(s): {invalid}. Valid: {ALL_STAGES}")
    return stages


def _bootstrap_logging(log_dir: Optional[str], level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(Path(log_dir) / f"mechanistic_pipeline_{ts}.log")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        handlers.append(fh)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def save_run_summary(summary: dict, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary_path = out / f"run_summary_{ts}.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    logging.getLogger(__name__).info("Run summary saved -> %s", summary_path)


def main() -> int:
    run_compute_scan(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "compute_scan.json"))
    args = parse_args()
    cfg = load_config(args.config)

    log_cfg = cfg.get("logging", {})
    _bootstrap_logging(log_dir=log_cfg.get("log_dir"), level=log_cfg.get("level", "INFO"))
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("Mechanistic Disentanglement Pipeline Runner  v%s", __version__)
    logger.info("Config : %s", args.config.resolve())
    logger.info("=" * 70)

    cfg = apply_overrides(cfg, args)

    try:
        stages = resolve_stages(args.stages)
    except ValueError as exc:
        logger.error("Stage resolution error: %s", exc)
        return 1

    logger.info("Stages to run: %s", stages)
    logger.info("Stage descriptions: %s", {s: STAGE_MAP[s] for s in stages})

    if args.dry_run:
        logger.info("[DRY-RUN] Config validation passed. Checking imports...")
        try:
            from latent_coordination.pipeline.mechanistic_pipeline import MechanisticPipeline  # noqa: F401
            logger.info("[DRY-RUN] Imports OK. Exiting without running pipeline.")
        except ImportError as exc:
            logger.warning("[DRY-RUN] Import warning (may be OK in dev): %s", exc)
        return 0

    try:
        from latent_coordination.pipeline.mechanistic_pipeline import MechanisticPipeline
    except ImportError as exc:
        logger.error("Failed to import MechanisticPipeline: %s", exc)
        logger.debug(traceback.format_exc())
        return 1

    output_dir = cfg.get("project", {}).get("output_dir", "results/mechanistic")
    start_time = time.time()
    success = False
    error_msg: Optional[str] = None

    try:
        pipeline = MechanisticPipeline(config=cfg, resume=args.resume)
        logger.info("Pipeline instantiated. Starting execution...")
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
        summary = {
            "pipeline": "mechanistic_disentanglement",
            "version": __version__,
            "timestamp_utc": datetime.utcfromtimestamp(start_time).isoformat(),
            "elapsed_seconds": round(end_time - start_time, 2),
            "stages_requested": stages,
            "config_project": cfg.get("project", {}),
            "success": success,
            "error": error_msg,
        }
        try:
            save_run_summary(summary, output_dir)
        except Exception as summ_exc:  # pylint: disable=broad-except
            logger.warning("Could not save run summary: %s", summ_exc)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
