"""Full Mechanistic Disentanglement Pipeline: Lexicon, Extraction, Decomposition, Isomorphism, Steering, Evaluation, Viz."""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from latent_coordination.data.lexicon import ContrastiveLexicon
from latent_coordination.data.dataset_loader import DatasetLoader
from latent_coordination.geometry.activation_extractor import ActivationExtractor
from latent_coordination.geometry.svd_decomposer import SVDSubspaceDecomposer
from latent_coordination.geometry.isomorphism import GeometricIsomorphismAnalyzer
from latent_coordination.steering.gaussian_scheduler import GaussianDepthScheduler
from latent_coordination.steering.magnitude_norm import MagnitudeNormalizer
from latent_coordination.steering.latent_steerer import LatentSteerer
from latent_coordination.steering.vector_builder import SteeringVectorBuilder
from latent_coordination.eval.steering_benchmark import BenchmarkRunner
from latent_coordination.viz.geometry_plots import GeometryPlotter
from latent_coordination.viz.steering_plots import SteeringPlotter
from latent_coordination.viz.mechanistic_plots import MechanisticPlotter
from shared.checkpointing import CheckpointManager
from shared.logging_utils import setup_logging

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
class PipelineConfig:
    """Config parameters orchestrating the Mechanistic Disentanglement research pipeline."""
    model_id: str = "Qwen/Qwen3.5-9B"
    target_languages: List[str] = field(default_factory=lambda: ["th", "my", "km"])
    n_probe_pairs: int = 15
    mu_frac: float = 0.60
    sigma_frac: float = 0.15
    alpha_0: float = 1.2
    eta: float = 0.5
    n_svd_components: int = 16
    output_dir: str = "results/mechanistic"
    device: str = "cpu"
    dtype: str = "float16"            # V100-safe; bf16 auto-downgrades in the loader
    load_in_8bit: bool = False        # enable to fit 8-9B models on a 16GB V100
    batch_size: int = 4
    apply_subspace_projection: bool = True
    checkpoint_interval: int = 1
    n_samples_per_language: int = 100   # default cap; per-benchmark overrides in benchmark_samples
    benchmark_samples: Dict[str, int] = field(default_factory=dict)
    seed: int = 42

    def bench_n(self, name: str) -> int:
        """Per-benchmark sample cap, falling back to the global n_samples_per_language."""
        return int(self.benchmark_samples.get(name, self.n_samples_per_language))
    safeguard_repo_id: Optional[str] = None   # HF id for the SEA safety benchmark (optional)
    benchmark_selection: Optional[List[str]] = None   # subset of {belebele,sea_vision,sea_vl,safety}; None=all
    baseline_selection: Optional[List[str]] = None    # subset of the 5 baselines; None=all

    def to_dict(self) -> Dict:
        return asdict(self)


class MechanisticPipeline:
    """Orchestrates the complete mechanistic disentanglement and steering pipeline."""

    def __init__(self, config: PipelineConfig | Dict, resume: bool = False) -> None:
        self.resume = resume
        if isinstance(config, dict):
            langs = []
            target_langs = config.get("target_languages", {})
            if isinstance(target_langs, dict):
                for val in target_langs.values():
                    if isinstance(val, list):
                        langs.extend(val)
                    else:
                        langs.append(val)
            elif isinstance(target_langs, list):
                langs = target_langs
            else:
                langs = [target_langs] if target_langs else []

            model_cfg = config.get("model", {})
            lexicon_cfg = config.get("lexicon", {})
            svd_cfg = config.get("svd_decomposition", {})
            project_cfg = config.get("project", {})
            steering_cfg = config.get("steering", {})
            gaussian_cfg = steering_cfg.get("gaussian", {})
            ckpt_cfg = config.get("checkpointing", {})
            bench_cfg = config.get("benchmarks", {})
            sea_vision_cfg = bench_cfg.get("sea_vision", {})

            pipeline_config = PipelineConfig(
                model_id=model_cfg.get("model_id", "Qwen/Qwen3.5-9B"),
                target_languages=langs,
                n_probe_pairs=lexicon_cfg.get("n_pairs_per_language", 15),
                mu_frac=gaussian_cfg.get("mu_frac", 0.60),
                sigma_frac=gaussian_cfg.get("sigma_frac", 0.15),
                alpha_0=gaussian_cfg.get("alpha_0", 1.2),
                eta=gaussian_cfg.get("eta", 0.5),
                n_svd_components=svd_cfg.get("n_components", 16),
                output_dir=project_cfg.get("output_dir", "results/mechanistic"),
                device=model_cfg.get("device", "cpu"),
                dtype=model_cfg.get("torch_dtype", "float16"),
                load_in_8bit=model_cfg.get("load_in_8bit", False),
                batch_size=model_cfg.get("batch_size", 4),
                apply_subspace_projection=steering_cfg.get("apply_subspace_projection", True),
                checkpoint_interval=ckpt_cfg.get("interval_stages", 1),
                n_samples_per_language=sea_vision_cfg.get("n_samples_per_language", 100),
                benchmark_samples={
                    "belebele": bench_cfg.get("belebele", {}).get("n_samples_per_language", 200),
                    "sea_vision": bench_cfg.get("sea_vision", {}).get("n_samples_per_language", 100),
                    "sea_vl": bench_cfg.get("sea_vl", {}).get("n_samples_per_language", 100),
                },
                seed=project_cfg.get("seed", 42),
                safeguard_repo_id=bench_cfg.get("sea_safeguardbench", {}).get("repo_id"),
                benchmark_selection=config.get("benchmark_selection"),
                baseline_selection=config.get("baseline_selection"),
            )
            self.config = pipeline_config
        else:
            self.config = config

        # Reproducibility: seed all RNGs before any stochastic stage.
        from shared.seeding import set_seed
        set_seed(int(self.config.seed))

        if self.config.device == "auto":
            from shared.parallelism import DeviceManager
            device = str(DeviceManager.get_best_device())
            logger.info("Auto-scheduling device. Selected: %s", device)
            self.config.device = device

        self.timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(self.config.output_dir) / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging inside the run directory
        setup_logging("mechanistic_pipeline", self.run_dir, level=logging.INFO)
        logger.info("Mechanistic Disentanglement Pipeline initialized at directory: %s", self.run_dir)

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(self.config.output_dir) / "checkpoints",
            project_name="mechanistic"
        )

    def _load_base_model(self):
        """Load the base model + tokenizer via the shared accelerate loader.

        Centralises model loading so V100-safe dtype handling and optional 8-bit
        quantisation are applied consistently across Stage B and Stage F.
        """
        from shared.model_loader import ModelLoadSpec, load_model_and_tokenizer

        spec = ModelLoadSpec(
            model_id=self.config.model_id,
            device=self.config.device,
            dtype=self.config.dtype,
            load_in_8bit=self.config.load_in_8bit,
            output_hidden_states=False,
            trust_remote_code=True,
        )
        return load_model_and_tokenizer(spec)

    def run(self, stages: Optional[List[str]] = None) -> Dict:
        """Executes the pipeline stage-by-stage with resume support."""
        if stages is None:
            stages = ["A", "B", "C", "D", "E", "F", "G", "H"]

        logger.info("Starting pipeline execution. Config: %s", self.config)
        logger.info("Stages to run: %s", stages)

        lexicon, dataset = None, None
        en_states, tgt_states = None, None
        decomposers = None
        isomorphism_reports = None
        steering_vectors_dict = None
        benchmark_report = None
        final_report = {}

        # Stage A: Lexicon & Data Loading
        if "A" in stages:
            lexicon, dataset = self._run_stage_a()
        elif self.resume and self.checkpoint_manager.exists("stage_a"):
            lexicon, dataset = self.checkpoint_manager.load_latest("stage_a")

        # Stage B: Activation Extraction
        if "B" in stages:
            if lexicon is None:
                if self.checkpoint_manager.exists("stage_a"):
                    lexicon, _ = self.checkpoint_manager.load_latest("stage_a")
                else:
                    raise RuntimeError("Lexicon not available for Stage B; run Stage A first.")
            en_states, tgt_states = self._run_stage_b(lexicon)
        elif self.resume and self.checkpoint_manager.exists("stage_b"):
            en_states, tgt_states = self.checkpoint_manager.load_latest("stage_b")

        # Stage C: SVD Decomposition
        if "C" in stages:
            if en_states is None or tgt_states is None:
                if self.checkpoint_manager.exists("stage_b"):
                    en_states, tgt_states = self.checkpoint_manager.load_latest("stage_b")
                else:
                    raise RuntimeError("States not available for Stage C; run Stage B first.")
            decomposers = self._run_stage_c(en_states, tgt_states)
        elif self.resume and self.checkpoint_manager.exists("stage_c"):
            decomposers = self.checkpoint_manager.load_latest("stage_c")

        # Stage D: Isomorphism Analysis
        if "D" in stages:
            if en_states is None or tgt_states is None:
                if self.checkpoint_manager.exists("stage_b"):
                    en_states, tgt_states = self.checkpoint_manager.load_latest("stage_b")
                else:
                    raise RuntimeError("States not available for Stage D; run Stage B first.")
            if decomposers is None and self.checkpoint_manager.exists("stage_c"):
                decomposers = self.checkpoint_manager.load_latest("stage_c")
            isomorphism_reports = self._run_stage_d(en_states, tgt_states, decomposers)
        elif self.resume and self.checkpoint_manager.exists("stage_d"):
            isomorphism_reports = self.checkpoint_manager.load_latest("stage_d")

        # Stage E: Steering Vector Building
        if "E" in stages:
            if decomposers is None:
                if self.checkpoint_manager.exists("stage_c"):
                    decomposers = self.checkpoint_manager.load_latest("stage_c")
                else:
                    raise RuntimeError("Decomposers not available for Stage E; run Stage C first.")
            if en_states is None or tgt_states is None:
                if self.checkpoint_manager.exists("stage_b"):
                    en_states, tgt_states = self.checkpoint_manager.load_latest("stage_b")
                else:
                    raise RuntimeError("States not available for Stage E; run Stage B first.")
            steering_vectors_dict = self._run_stage_e(decomposers, en_states, tgt_states)
        elif self.resume and self.checkpoint_manager.exists("stage_e"):
            steering_vectors_dict = self.checkpoint_manager.load_latest("stage_e")

        # Stage F: Benchmark Evaluation
        if "F" in stages:
            if dataset is None:
                if self.checkpoint_manager.exists("stage_a"):
                    _, dataset = self.checkpoint_manager.load_latest("stage_a")
                else:
                    raise RuntimeError("Dataset not available for Stage F; run Stage A first.")
            if decomposers is None:
                if self.checkpoint_manager.exists("stage_c"):
                    decomposers = self.checkpoint_manager.load_latest("stage_c")
                else:
                    raise RuntimeError("Decomposers not available for Stage F; run Stage C first.")
            if steering_vectors_dict is None:
                if self.checkpoint_manager.exists("stage_e"):
                    steering_vectors_dict = self.checkpoint_manager.load_latest("stage_e")
                else:
                    raise RuntimeError("Steering vectors not available for Stage F; run Stage E first.")
            benchmark_report = self._run_stage_f(dataset, decomposers, steering_vectors_dict)
        elif self.resume and self.checkpoint_manager.exists("stage_f"):
            benchmark_report = self.checkpoint_manager.load_latest("stage_f")

        # Stage G: Visualizations
        if "G" in stages:
            if decomposers is None:
                if self.checkpoint_manager.exists("stage_c"):
                    decomposers = self.checkpoint_manager.load_latest("stage_c")
            if en_states is None or tgt_states is None:
                if self.checkpoint_manager.exists("stage_b"):
                    en_states, tgt_states = self.checkpoint_manager.load_latest("stage_b")
            if steering_vectors_dict is None:
                if self.checkpoint_manager.exists("stage_e"):
                    steering_vectors_dict = self.checkpoint_manager.load_latest("stage_e")
            if benchmark_report is None:
                if self.checkpoint_manager.exists("stage_f"):
                    benchmark_report = self.checkpoint_manager.load_latest("stage_f")

            # Stage G is purely diagnostic plotting; its inputs (benchmark results,
            # geometry) are already saved/cached by earlier stages. A viz failure
            # (e.g. partially restored states on --resume) must not abort the run
            # before Stage H compiles the final report.
            try:
                self._run_stage_g(decomposers, en_states, tgt_states, steering_vectors_dict, benchmark_report)
            except Exception as exc:  # noqa: BLE001 - diagnostic stage, never fatal
                logger.warning("Stage G (diagnostic plots) failed; continuing to report. %r", exc)

        # Stage H: Final Report compilation
        if "H" in stages:
            if isomorphism_reports is None:
                if self.checkpoint_manager.exists("stage_d"):
                    isomorphism_reports = self.checkpoint_manager.load_latest("stage_d")
            if benchmark_report is None:
                if self.checkpoint_manager.exists("stage_f"):
                    benchmark_report = self.checkpoint_manager.load_latest("stage_f")
            final_report = self._run_stage_h(isomorphism_reports, benchmark_report)

        return final_report

    def _run_stage_a(self) -> Tuple[ContrastiveLexicon, List]:
        """Stage A: Lexicon and Dataset setup."""
        if self.resume and self.checkpoint_manager.exists("stage_a"):
            logger.info("Resuming Stage A from checkpoint.")
            return self.checkpoint_manager.load_latest("stage_a")

        logger.info("Running Stage A: Setting up lexicons and datasets.")
        lexicon = ContrastiveLexicon()
        loader = DatasetLoader()
        
        dataset = []
        sel = set(self.config.benchmark_selection) if self.config.benchmark_selection else None
        def _enabled(name: str) -> bool:
            return sel is None or name in sel

        # 1. Belebele
        if _enabled("belebele"):
            try:
                ds = loader.load_belebele(self.config.target_languages, max_per_language=self.config.bench_n("belebele"))
                dataset.extend(ds)
                logger.info("Integrated %d samples from Belebele.", len(ds))
            except RuntimeError as e:
                logger.warning("Skipping Belebele: %s", e)

        # 2. SEA-Vision
        if _enabled("sea_vision"):
            try:
                ds = loader.load_sea_vision(self.config.target_languages, max_per_language=self.config.bench_n("sea_vision"))
                dataset.extend(ds)
                logger.info("Integrated %d samples from SEA-Vision.", len(ds))
            except RuntimeError as e:
                logger.warning("Skipping SEA-Vision: %s", e)

        # 3. SEA-VL (native-language captions; per-language cap)
        if _enabled("sea_vl"):
            try:
                ds = loader.load_sea_vl(self.config.target_languages, max_per_language=self.config.bench_n("sea_vl"))
                dataset.extend(ds)
                logger.info("Integrated %d samples from SEA-VL.", len(ds))
            except RuntimeError as e:
                logger.warning("Skipping SEA-VL: %s", e)

        # 4. SEA safety benchmark (optional; only if a real repo_id is configured)
        if _enabled("safety") and self.config.safeguard_repo_id:
            try:
                ds = loader.load_sea_safeguardbench(
                    languages=self.config.target_languages,
                    max_samples=self.config.n_samples_per_language,
                    repo_id=self.config.safeguard_repo_id,
                )
                dataset.extend(ds)
                logger.info("Integrated %d samples from safety benchmark '%s'.",
                            len(ds), self.config.safeguard_repo_id)
            except (RuntimeError, ValueError) as e:
                logger.warning("Skipping safety benchmark: %s", e)
        else:
            logger.info(
                "Safety benchmark disabled (no benchmarks.sea_safeguardbench.repo_id set)."
            )

        if not dataset:
            raise RuntimeError("Failed to load any real datasets for evaluation. Mock data is forbidden.")

        self.checkpoint_manager.save((lexicon, dataset), "stage_a")
        return lexicon, dataset

    def _run_stage_b(self, lexicon: ContrastiveLexicon) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, Dict[int, torch.Tensor]]]:
        """Stage B: Extract hidden state representations using real model activations."""
        if self.resume and self.checkpoint_manager.exists("stage_b"):
            logger.info("Resuming Stage B from checkpoint.")
            states = self.checkpoint_manager.load_latest("stage_b")
            # Recover per-model layer_ids from the cached state keys so later stages
            # (Stage F steering, Stage G viz) do not fall back to a wrong default.
            try:
                en_states = states[0]
                if en_states:
                    first_lang = next(iter(en_states))
                    self._layer_ids = sorted(int(k) for k in en_states[first_lang].keys())
            except Exception:  # noqa: BLE001 — best-effort recovery
                pass
            return states

        logger.info("Running Stage B: Loading model '%s' and extracting real activations.", self.config.model_id)

        model, tokenizer = self._load_base_model()

        extractor = ActivationExtractor(
            model=model,
            tokenizer=tokenizer,
            device=self.config.device,
            pooling="mean",
        )
        n_layers = extractor.n_layers
        # Use middle and upper layers for contrastive analysis
        layer_ids = [
            int(n_layers * 0.5),
            int(n_layers * 0.75),
        ]
        logger.info("Using layers %s out of %d total layers.", layer_ids, n_layers)

        en_states: Dict[str, Dict[int, torch.Tensor]] = {}
        tgt_states: Dict[str, Dict[int, torch.Tensor]] = {}

        for lang in self.config.target_languages:
            pairs = lexicon.get_pairs(lang, n_pairs=self.config.n_probe_pairs)
            if not pairs:
                logger.warning("No pairs available for language '%s'. Skipping.", lang)
                continue
            en_texts = [p[0] for p in pairs]
            tgt_texts = [p[1] for p in pairs]

            logger.info("Extracting activations for language '%s' (%d pairs).", lang, len(pairs))
            en_layer_states, tgt_layer_states = extractor.extract_contrastive_pairs(
                en_texts,
                tgt_texts,
                layer_ids=layer_ids,
                batch_size=self.config.batch_size,
            )
            en_states[lang] = en_layer_states
            tgt_states[lang] = tgt_layer_states
            logger.info("Language '%s' extraction complete.", lang)

        # Store model reference for later stages
        self._model = model
        self._tokenizer = tokenizer
        self._layer_ids = layer_ids

        self.checkpoint_manager.save((en_states, tgt_states), "stage_b")
        return en_states, tgt_states

    def _run_stage_c(
        self,
        en_states: Dict[str, Dict[int, torch.Tensor]],
        tgt_states: Dict[str, Dict[int, torch.Tensor]],
    ) -> Dict[str, Dict[int, SVDSubspaceDecomposer]]:
        """Stage C: Perform contrastive SVD to decompose representations."""
        if self.resume and self.checkpoint_manager.exists("stage_c"):
            logger.info("Resuming Stage C from checkpoint.")
            return self.checkpoint_manager.load_latest("stage_c")

        logger.info("Running Stage C: Fitting SVD Subspace Decomposers in parallel.")
        decomposers = {}

        def _fit_lang(lang: str):
            lang_decomposers = {}
            for layer in en_states[lang].keys():
                dec = SVDSubspaceDecomposer(n_components=self.config.n_svd_components, device=self.config.device)
                dec.fit(en_states[lang][layer], tgt_states[lang][layer])
                lang_decomposers[layer] = dec
            return lang, lang_decomposers

        from shared.parallelism import ParallelRunner
        args_list = [(lang,) for lang in self.config.target_languages]
        results = ParallelRunner.run_threads(_fit_lang, args_list, max_workers=len(args_list), desc="Fitting SVD Decomposers")
        for lang, lang_decomposers in results:
            decomposers[lang] = lang_decomposers

        self.checkpoint_manager.save(decomposers, "stage_c")
        return decomposers

    def _run_stage_d(
        self,
        en_states: Dict[str, Dict[int, torch.Tensor]],
        tgt_states: Dict[str, Dict[int, torch.Tensor]],
        decomposers: Optional[Dict[str, Dict[int, SVDSubspaceDecomposer]]] = None,
    ) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Stage D: Compute cross-lingual geometric isomorphism reports."""
        if self.resume and self.checkpoint_manager.exists("stage_d"):
            logger.info("Resuming Stage D from checkpoint.")
            return self.checkpoint_manager.load_latest("stage_d")

        logger.info("Running Stage D: Analyzing Geometric Isomorphisms in parallel.")
        analyzer = GeometricIsomorphismAnalyzer()
        reports = {}

        def _analyze_lang(lang: str):
            lang_reports = {}
            try:
                from mrre_drift.interpret.craf import CrossLingualAlignmentProbe
                clap_probe = CrossLingualAlignmentProbe(model=None, tokenizer=None, device=self.config.device)
            except ImportError:
                clap_probe = None

            for layer in en_states[lang].keys():
                en_h = en_states[lang][layer]
                tgt_h = tgt_states[lang][layer]
                report = analyzer.compute_all(en_h, tgt_h, lang)
                
                ur_cka = None
                if decomposers and lang in decomposers and layer in decomposers[lang]:
                    en_ur = decomposers[lang][layer].project_to_reasoning(en_h)
                    tgt_ur = decomposers[lang][layer].project_to_reasoning(tgt_h)
                    report_ur = analyzer.compute_all(en_ur, tgt_ur, lang)
                    ur_cka = report_ur.cka

                clap_delta = None
                if clap_probe:
                    try:
                        clap_res = clap_probe._decompose_layer(layer, en_h, tgt_h)
                        clap_delta = clap_res.alignment_delta
                    except Exception:
                        pass

                lang_reports[layer] = {
                    "raw_cka": report.cka,
                    "ur_projected_cka": ur_cka,
                    "cka": report.cka,
                    "clap_delta": clap_delta,
                    "rsa": report.rsa_spearman,
                    "procrustes_residual": report.procrustes_disparity,
                    "distortion_ratio": report.magnitude_distortion_ratio,
                }
            return lang, lang_reports

        from shared.parallelism import ParallelRunner
        args_list = [(lang,) for lang in self.config.target_languages]
        results = ParallelRunner.run_threads(_analyze_lang, args_list, max_workers=len(args_list), desc="Analyzing Geometric Isomorphisms")
        for lang, lang_reports in results:
            reports[lang] = lang_reports

        self.checkpoint_manager.save(reports, "stage_d")
        return reports

    def _run_stage_e(
        self,
        decomposers: Dict[str, Dict[int, SVDSubspaceDecomposer]],
        en_states: Dict[str, Dict[int, torch.Tensor]],
        tgt_states: Dict[str, Dict[int, torch.Tensor]],
    ) -> Dict[str, SteeringVectorBuilder]:
        """Stage E: Build cross-lingual steering vectors."""
        if self.resume and self.checkpoint_manager.exists("stage_e"):
            logger.info("Resuming Stage E from checkpoint.")
            return self.checkpoint_manager.load_latest("stage_e")

        logger.info("Running Stage E: Generating activation steering vectors in parallel.")
        vectors_dict = {}
        builder = SteeringVectorBuilder()

        def _build_lang(lang: str):
            layer_ids = list(en_states[lang].keys())
            mean_diff = builder.build_mean_diff(en_states[lang], tgt_states[lang], layer_ids)

            subspace_vectors = {}
            for lid in layer_ids:
                dec = decomposers[lang][lid]
                en_proj = dec.project_to_reasoning(en_states[lang][lid])
                tgt_proj = dec.project_to_reasoning(tgt_states[lang][lid])
                subspace_vectors[lid] = tgt_proj.mean(dim=0) - en_proj.mean(dim=0)

            return lang, {
                "mean_diff": mean_diff,
                "subspace_projected": subspace_vectors,
            }

        from shared.parallelism import ParallelRunner
        args_list = [(lang,) for lang in self.config.target_languages]
        results = ParallelRunner.run_threads(_build_lang, args_list, max_workers=len(args_list), desc="Building Steering Vectors")
        for lang, lang_vectors in results:
            vectors_dict[lang] = lang_vectors

        self.checkpoint_manager.save(vectors_dict, "stage_e")
        return vectors_dict

    def _build_layer_decomposers(
        self,
        layer_ids: List[int],
        decomposers: Dict[str, Dict[int, SVDSubspaceDecomposer]],
        en_states_w: Dict[str, Dict[int, torch.Tensor]],
        tgt_states_w: Dict[str, Dict[int, torch.Tensor]],
    ) -> Dict[int, SVDSubspaceDecomposer]:
        """Build one reasoning-subspace decomposer per steering layer.

        The aggregate steering vector at each layer blends all target languages, so
        the projection that anchors it onto the reasoning subspace should reflect all
        languages too. When pooled hidden states are available we fit a fresh
        per-layer decomposer on the language-pooled ``(en, tgt)`` pairs; otherwise we
        fall back to a fitted single-language decomposer for that layer. Returns an
        empty dict when nothing is available (projection then degrades to a no-op,
        and ``run_suite`` warns about identical baselines).
        """
        langs = sorted(self.config.target_languages)
        layer_decs: Dict[int, SVDSubspaceDecomposer] = {}
        for lid in layer_ids:
            # Preferred: pooled refit across all languages with cached states for this layer.
            en_pool, tgt_pool = [], []
            for lang in langs:
                if lid in en_states_w.get(lang, {}) and lid in tgt_states_w.get(lang, {}):
                    en_pool.append(en_states_w[lang][lid].float())
                    tgt_pool.append(tgt_states_w[lang][lid].float())
            if en_pool:
                try:
                    dec = SVDSubspaceDecomposer(
                        n_components=self.config.n_svd_components, device=self.config.device
                    )
                    dec.fit(torch.cat(en_pool, dim=0), torch.cat(tgt_pool, dim=0))
                    layer_decs[lid] = dec
                    continue
                except Exception as exc:  # noqa: BLE001 - fall back to a fitted single-lang decomposer
                    logger.warning("Pooled decomposer fit failed for layer %d: %s", lid, exc)
            # Fallback: reuse an already-fitted per-language decomposer for this layer.
            for lang in langs:
                if lang in decomposers and lid in decomposers[lang]:
                    layer_decs[lid] = decomposers[lang][lid]
                    logger.info(
                        "Layer %d: using '%s' decomposer for subspace projection "
                        "(pooled states unavailable).", lid, lang,
                    )
                    break
        return layer_decs

    @staticmethod
    def _steering_vector_layers(steering_vectors_dict: Dict) -> List[int]:
        """Layers that actually carry mean-diff steering vectors, unioned over languages.

        Stage F must inject at exactly these layers; injecting elsewhere produces a
        silent no-op ("0 layers active"). Returns a sorted list, or [] if none.
        """
        layers: set = set()
        for lang_vecs in (steering_vectors_dict or {}).values():
            if not isinstance(lang_vecs, dict):
                continue
            mean_diff = lang_vecs.get("mean_diff")
            vecs = getattr(mean_diff, "vectors", None)
            if isinstance(vecs, dict):
                layers.update(int(k) for k in vecs.keys())
        return sorted(layers)

    def _run_stage_f(
        self,
        dataset: List,
        decomposers: Dict[str, Dict[int, SVDSubspaceDecomposer]],
        steering_vectors_dict: Dict,
    ) -> Dict:
        """Stage F: Evaluate steered activations against baselines using real model inference."""
        # The monolithic stage_f checkpoint only short-circuits when NO explicit
        # benchmark/baseline selection is active. With a selection, re-enter run_suite so the
        # per-baseline cache reuses finished baselines AND computes newly-requested ones — this
        # is what makes "resume with a different combination" correct.
        selection_active = bool(self.config.benchmark_selection or self.config.baseline_selection)
        if self.resume and not selection_active and self.checkpoint_manager.exists("stage_f"):
            logger.info("Resuming Stage F from checkpoint.")
            return self.checkpoint_manager.load_latest("stage_f")

        logger.info("Running Stage F: Running evaluations and benchmark suites with real model.")

        # Load model if not already loaded from Stage B
        if not hasattr(self, "_model") or self._model is None:
            logger.info("Loading model '%s' for evaluation.", self.config.model_id)
            self._model, self._tokenizer = self._load_base_model()

        n_layers = self._resolve_n_layers(self._model)
        # Derive injection layers from the steering vectors that actually exist.
        # The vectors are built at per-model depth-relative layers (~0.5L, 0.75L),
        # which differ across backbones (e.g. Llama [16,24], Gemma [21,31], SeaLLMs
        # [14,21]). On --resume the Stage-B early-return skips setting self._layer_ids,
        # so a hardcoded fallback would inject at the wrong layers and silently
        # no-op steering (0 layers active). Anchoring to the vector keys prevents that.
        layer_ids = self._steering_vector_layers(steering_vectors_dict)
        if not layer_ids:
            layer_ids = getattr(self, "_layer_ids", None) or [
                int(n_layers * 0.5), int(n_layers * 0.75)
            ]
            logger.warning(
                "No steering-vector layers found; falling back to %s.", layer_ids
            )
        self._layer_ids = layer_ids
        logger.info("Stage F injection layers: %s (n_layers=%d)", layer_ids, n_layers)

        # Build distortion-ratio-weighted aggregate steering vectors.
        # Languages whose LRL activations are more compact (higher distortion ratio)
        # get less weight so high-resource directions do not dominate.
        norm_analyzer = MagnitudeNormalizer()
        # Try to recover hidden states for distortion weighting; fall back to equal weights.
        _cached_states = None
        if self.checkpoint_manager.exists("stage_b"):
            try:
                _cached_states = self.checkpoint_manager.load_latest("stage_b")
            except Exception:
                pass
        en_states_w = _cached_states[0] if _cached_states else {}
        tgt_states_w = _cached_states[1] if _cached_states else {}

        agg_vectors: Dict[int, torch.Tensor] = {}
        for lid in layer_ids:
            lang_vecs = []
            lang_weights = []
            for lang, lang_vecs_dict in steering_vectors_dict.items():
                mean_diff = lang_vecs_dict.get("mean_diff")
                if mean_diff and hasattr(mean_diff, "vectors") and lid in mean_diff.vectors:
                    sv = mean_diff.vectors[lid].float()
                    lang_vecs.append(sv)
                    # Weight inversely proportional to distortion ratio so compact
                    # (low-resource) language vectors are not overwhelmed.
                    if lang in en_states_w and lid in en_states_w.get(lang, {}) and lid in tgt_states_w.get(lang, {}):
                        ratio = norm_analyzer.compute_distortion_ratio(
                            en_states_w[lang][lid], tgt_states_w[lang][lid], lid
                        )
                        lang_weights.append(1.0 / max(ratio, 1e-6))
                    else:
                        lang_weights.append(1.0)
            if lang_vecs:
                w = torch.tensor(lang_weights, dtype=torch.float32)
                w = w / w.sum()
                stacked = torch.stack(lang_vecs, dim=0)  # (n_langs, hidden_dim)
                agg_vectors[lid] = (stacked * w.unsqueeze(1)).sum(dim=0)
            else:
                logger.warning("No steering vectors available for layer %d.", lid)

        runner = BenchmarkRunner(output_dir=self.run_dir)
        scheduler = GaussianDepthScheduler(
            alpha_0=self.config.alpha_0,
            mu_s=float(int(n_layers * 0.6)),
            sigma_s=float(int(n_layers * 0.15)),
            n_layers=n_layers
        )
        normalizer = MagnitudeNormalizer()
        # Per-layer reasoning-subspace decomposers so the mrre_two_stage baseline's
        # subspace projection is functional. Without this the steerer holds
        # decomposer=None and projection is a silent no-op, collapsing mrre_two_stage
        # onto standard_clas/gaussian_scheduled.
        layer_decomposers = self._build_layer_decomposers(
            layer_ids, decomposers, en_states_w, tgt_states_w
        )
        if not layer_decomposers:
            logger.warning(
                "No per-layer decomposers available; mrre_two_stage subspace "
                "projection will be a no-op for this run."
            )
        steerer = LatentSteerer(scheduler, normalizer, decomposer=layer_decomposers or None)
        # Per-layer reasoning-subspace decomposers so the mrre_two_stage baseline's
        # subspace projection is functional. Without this the steerer holds
        # decomposer=None and projection is a silent no-op, collapsing mrre_two_stage
        # onto standard_clas/gaussian_scheduled.
        layer_decomposers = self._build_layer_decomposers(
            layer_ids, decomposers, en_states_w, tgt_states_w
        )
        if not layer_decomposers:
            logger.warning(
                "No per-layer decomposers available; mrre_two_stage subspace "
                "projection will be a no-op for this run."
            )
        steerer = LatentSteerer(scheduler, normalizer, decomposer=layer_decomposers or None)

        import hashlib
        model_slug = "".join(
            c if (c.isalnum() or c in "-_.") else "_" for c in str(self.config.model_id)
        )
        # Data-scope fingerprint: a cached (model, baseline) result is only valid for the exact
        # languages + benchmark mix + sample counts it was computed on. Without this, a later
        # full-dataset run with --resume would silently reuse a smaller partial result.
        scope = "|".join([
            ",".join(sorted(self.config.target_languages)),
            ",".join(sorted(b for b in ("belebele", "sea_vision", "sea_vl")
                            if self.config.benchmark_selection is None
                            or b in self.config.benchmark_selection)),
            f"n={self.config.bench_n('belebele')}-{self.config.bench_n('sea_vision')}-{self.config.bench_n('sea_vl')}",
        ])
        scope_hash = hashlib.md5(scope.encode()).hexdigest()[:8]
        report = runner.run_suite(
            model=self._model,
            tokenizer=self._tokenizer,
            samples=dataset,
            steerer=steerer,
            steering_vectors=agg_vectors,
            layer_ids=layer_ids,
            eta=self.config.eta,
            apply_subspace_proj=self.config.apply_subspace_projection,
            baselines=self.config.baseline_selection,   # None → all
            checkpoint_manager=self.checkpoint_manager,
            cache_prefix=f"mech::{model_slug}::{scope_hash}",
        )

        report_dict = report.to_dict()
        self.checkpoint_manager.save(report_dict, "stage_f")
        return report_dict

    def _run_stage_g(
        self,
        decomposers: Dict[str, Dict[int, SVDSubspaceDecomposer]],
        en_states: Dict[str, Dict[int, torch.Tensor]],
        tgt_states: Dict[str, Dict[int, torch.Tensor]],
        steering_vectors_dict: Dict,
        benchmark_report: Dict,
    ) -> None:
        """Stage G: Visualize spectral shapes, trajectories, and curves using real pipeline data."""
        logger.info("Running Stage G: Generating diagnostic plots from real extracted states.")

        viz_dir = self.run_dir / "plots"
        viz_dir.mkdir(parents=True, exist_ok=True)

        geom_plotter = GeometryPlotter()
        steer_plotter = SteeringPlotter()
        mech_plotter = MechanisticPlotter()

        # Determine layer_ids from available state keys
        layer_ids = self._layer_ids if hasattr(self, "_layer_ids") else []
        if not layer_ids and en_states:
            first_lang = next(iter(en_states))
            layer_ids = sorted(en_states[first_lang].keys())

        # 1. Singular value spectra — from real SVD decomposers
        sv_dict = {}
        if decomposers:
            for lang in self.config.target_languages:
                if lang in decomposers:
                    first_layer = next(iter(decomposers[lang].keys()))
                    res = decomposers[lang][first_layer].get_result()
                    sv_dict[lang] = res.singular_values.numpy()
        if sv_dict:
            geom_plotter.plot_svd_spectrum(sv_dict, viz_dir / "svd_spectrum.png")

        # 2. Magnitude Distortion Curves — from real hidden state norms
        distortion_by_layer: Dict = {}
        if en_states and tgt_states:
            norm_analyzer = MagnitudeNormalizer()
            for lid in layer_ids:
                distortion_by_layer[lid] = {}
                for lang in self.config.target_languages:
                    if lang in en_states and lid in en_states[lang] and lid in tgt_states[lang]:
                        ratio = norm_analyzer.compute_distortion_ratio(
                            en_states[lang][lid], tgt_states[lang][lid]
                        )
                        distortion_by_layer[lid][lang] = float(ratio)
            if distortion_by_layer:
                geom_plotter.plot_magnitude_distortion_by_layer(
                    distortion_by_layer, self.config.target_languages, viz_dir / "magnitude_distortion.png"
                )

        # 3. Gaussian schedule — from real model depth
        n_layers = 32  # fallback; overwritten below when model is loaded
        if hasattr(self, "_model") and self._model is not None:
            try:
                n_layers = self._resolve_n_layers(self._model)
            except AttributeError:
                logger.warning("Could not resolve n_layers from model; falling back to 32.")
        scheduler = GaussianDepthScheduler(
            alpha_0=self.config.alpha_0,
            mu_s=float(int(n_layers * 0.6)),
            sigma_s=float(int(n_layers * 0.15)),
            n_layers=n_layers
        )
        steer_plotter.plot_gaussian_schedule(scheduler, n_layers, viz_dir / "gaussian_schedule.png")

        # 4. Softmax probability drift — from real benchmark report if available
        if benchmark_report and "results" in benchmark_report:
            res = benchmark_report["results"]
            before_sfr = res.get("no_intervention", {}).get("mean_sfr", 0.0)
            after_sfr = res.get("gaussian_scheduled", {}).get("mean_sfr", 0.0)
            probs_before = {lid: before_sfr for lid in layer_ids}
            probs_after = {lid: after_sfr for lid in layer_ids}
            steer_plotter.plot_softmax_drift(probs_before, probs_after, layer_ids, viz_dir / "softmax_drift.png")

        # 5. CKA isomorphism similarity — computed from real states
        if en_states and tgt_states:
            from latent_coordination.geometry.isomorphism import GeometricIsomorphismAnalyzer
            cka_analyzer = GeometricIsomorphismAnalyzer()
            languages = [l for l in self.config.target_languages if l in en_states]
            n = len(languages)
            if n > 0 and layer_ids:
                lid = layer_ids[0]
                cka_mat = np.zeros((n, n))
                for i, la in enumerate(languages):
                    for j, lb in enumerate(languages):
                        if la in en_states and lb in en_states and lid in en_states[la] and lid in en_states[lb]:
                            cka_mat[i, j] = cka_analyzer.compute_cka(en_states[la][lid], en_states[lb][lid])
                geom_plotter.plot_cka_similarity(cka_mat, languages, viz_dir / "cka_matrix.png")

        # 6. Residual Stream Norm Evolution — from real extracted states
        if en_states:
            first_lang = next(iter(en_states))
            real_hidden = {lid: en_states[first_lang][lid] for lid in layer_ids if lid in en_states[first_lang]}
            if real_hidden:
                steer_plotter.plot_residual_stream_norms(real_hidden, viz_dir / "residual_norms.png")

        # 7. SVD of real model weight matrix
        if hasattr(self, "_model") and self._model is not None and layer_ids:
            try:
                lid = layer_ids[-1]
                decoder_layers = LatentSteerer._resolve_decoder_layers(self._model)
                W = decoder_layers[lid].mlp.down_proj.weight.detach().float().cpu()
                mech_plotter.plot_weight_matrix_svd(W, lid, viz_dir / "weight_svd.png")
            except AttributeError as e:
                logger.warning("Could not access model weight matrix for SVD: %s", e)

        # 8–11. Logit-based plots — from real model forward passes on a small probe set
        if hasattr(self, "_model") and self._model is not None and hasattr(self, "_tokenizer"):
            probe_texts = [
                "The capital of Thailand is",
                "Two plus two equals",
                "The primary language of Vietnam is",
                "Bangkok is located in",
            ]
            try:
                enc = self._tokenizer(probe_texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
                enc = {k: v.to(self._model.device) for k, v in enc.items()}
                with torch.no_grad():
                    logits_before = self._model(**enc).logits.detach().float().cpu()

                # Apply first steering vector to get steered logits
                if steering_vectors_dict and layer_ids:
                    first_lang = next(iter(steering_vectors_dict))
                    # steering_vectors_dict[lang]["mean_diff"] is a SteeringVectors dataclass
                    # (layer_ids/vectors/method/metadata), not a plain {layer_id: Tensor}
                    # dict -- unwrap .vectors to get the actual per-layer mapping.
                    sv_obj = steering_vectors_dict[first_lang].get("mean_diff")
                    sv = sv_obj.vectors if sv_obj is not None else {}
                    first_lid = layer_ids[0]
                    if first_lid in sv:
                        hook_handles = []
                        # `self._model.device` is the *first* shard's device under
                        # device_map="auto"; the target layer's actual activations can live
                        # on a different shard (e.g. cuda:4 vs cuda:1), so move the steering
                        # vector to the hidden state's device inside the hook rather than
                        # pre-binding it to a possibly-wrong device.
                        sv_vec_cpu = sv[first_lid].float().cpu()

                        def _steer_hook(module, inputs, output):
                            if isinstance(output, tuple):
                                hs = output[0]
                                sv_vec = sv_vec_cpu.to(device=hs.device, dtype=hs.dtype)
                                hs = hs + sv_vec.unsqueeze(0).unsqueeze(0)
                                return (hs,) + output[1:]
                            sv_vec = sv_vec_cpu.to(device=output.device, dtype=output.dtype)
                            return output + sv_vec.unsqueeze(0).unsqueeze(0)

                        handle = self._model.model.layers[first_lid].register_forward_hook(_steer_hook)
                        hook_handles.append(handle)
                        with torch.no_grad():
                            logits_after = self._model(**enc).logits.detach().float().cpu()
                        for h in hook_handles:
                            h.remove()
                    else:
                        logits_after = logits_before
                else:
                    logits_after = logits_before

                steer_plotter.plot_entropy_heatmap(
                    {"Before": logits_before, "After": logits_after},
                    viz_dir / "entropy_heatmap.png"
                )
                steer_plotter.plot_logit_correlation(
                    logits_before[0], logits_after[0], viz_dir / "logit_correlation.png"
                )
                steer_plotter.plot_topk_density(
                    [logits_before, logits_after], ["Before", "After"], viz_dir / "topk_density.png"
                )
                steer_plotter.plot_steering_shift(
                    logits_before[0, 0], logits_after[0, 0], viz_dir / "steering_shift.png"
                )
            except Exception as e:
                logger.error("Logit-based plots failed: %s", e, exc_info=True)

        # 12–13. Attention entropy plots — from real model attention outputs
        if hasattr(self, "_model") and self._model is not None and hasattr(self, "_tokenizer"):
            try:
                probe_text = "Calculate the sum of five and three."
                enc = self._tokenizer([probe_text], return_tensors="pt").to(self._model.device)
                with torch.no_grad():
                    out = self._model(**enc, output_attentions=True)
                attn_before = [a.detach().float().cpu().numpy() for a in out.attentions]  # list of (1, heads, seq, seq)

                # With steering hook applied
                if steering_vectors_dict and layer_ids:
                    first_lang = next(iter(steering_vectors_dict))
                    # steering_vectors_dict[lang]["mean_diff"] is a SteeringVectors dataclass
                    # (layer_ids/vectors/method/metadata), not a plain {layer_id: Tensor}
                    # dict -- unwrap .vectors to get the actual per-layer mapping.
                    sv_obj = steering_vectors_dict[first_lang].get("mean_diff")
                    sv = sv_obj.vectors if sv_obj is not None else {}
                    first_lid = layer_ids[0]
                    if first_lid in sv:
                        # See the logit-based-plots block above: bind to the layer's actual
                        # device inside the hook, not `self._model.device` (the first shard).
                        sv_vec_cpu = sv[first_lid].float().cpu()

                        def _hook2(module, inputs, output):
                            if isinstance(output, tuple):
                                hs = output[0] + sv_vec_cpu.to(device=output[0].device, dtype=output[0].dtype).unsqueeze(0).unsqueeze(0)
                                return (hs,) + output[1:]
                            return output + sv_vec_cpu.to(device=output.device, dtype=output.dtype).unsqueeze(0).unsqueeze(0)

                        handle = self._model.model.layers[first_lid].register_forward_hook(_hook2)
                        with torch.no_grad():
                            out2 = self._model(**enc, output_attentions=True)
                        handle.remove()
                        attn_after = [a.detach().float().cpu().numpy() for a in out2.attentions]
                    else:
                        attn_after = attn_before
                else:
                    attn_after = attn_before

                mech_plotter.plot_attention_entropy(
                    [attn_before, attn_after], ["Before", "After"], viz_dir / "attention_entropy.png"
                )
                mech_plotter.plot_attention_patterns_grid(
                    attn_before[layer_ids[0] if layer_ids else 0], n_layers, viz_dir / "attention_grid.png"
                )
            except Exception as e:
                logger.error("Attention plots failed: %s", e, exc_info=True)

        # 14–15. Layer norms comparison — from real en_states
        if en_states and tgt_states and layer_ids:
            first_lang = next(iter(en_states))
            states_before_np = [
                en_states[first_lang][lid].float().numpy()
                for lid in layer_ids
                if lid in en_states[first_lang]
            ]
            states_after_np = [
                tgt_states[first_lang][lid].float().numpy()
                for lid in layer_ids
                if lid in tgt_states[first_lang]
            ]
            if states_before_np and states_after_np:
                mech_plotter.plot_layer_norms_comparison(
                    states_before_np, states_after_np, ["English", f"{first_lang.upper()}"],
                    viz_dir / "layer_norms_comparison.png"
                )
                mech_plotter.plot_activation_statistics_ribbon(
                    [states_before_np, states_after_np], ["English", f"{first_lang.upper()}"],
                    "norm", viz_dir / "activation_stats_ribbon.png"
                )

        # 16–17. Tokenizer alignment — from the real tokenizer loaded in Stage B
        if hasattr(self, "_tokenizer") and self._tokenizer is not None:
            probe_texts = [
                "Verify translation calculation in Thai",
                "Low-resource language cross-lingual latent steering",
            ]
            mech_plotter.plot_tokenizer_alignment(
                self._tokenizer,
                self._tokenizer,
                probe_texts,
                viz_dir / "tokenizer_alignment.png"
            )
            mech_plotter.plot_vocab_overlap(
                self._tokenizer,
                self._tokenizer,
                viz_dir / "vocab_overlap.png"
            )

        logger.info("All visual plots saved in %s", viz_dir)

    @staticmethod
    def _resolve_n_layers(model) -> int:
        """Resolve total transformer depth from a HuggingFace model."""
        from shared.model_utils import get_transformer_layers
        return len(get_transformer_layers(model))

    def _run_stage_h(self, isomorphism_reports: Dict, benchmark_report: Dict) -> Dict:
        """Stage H: Final JSON report consolidation."""
        logger.info("Running Stage H: Compiling final summary report.")
        final_report = {
            "timestamp": self.timestamp,
            "config": self.config.to_dict(),
            "geometric_isomorphism": isomorphism_reports,
            "evaluation": benchmark_report,
            "plots_directory": str(self.run_dir / "plots"),
            "status": "completed",
        }

        report_path = self.run_dir / "final_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)

        logger.info("Final report compiled successfully at %s", report_path)
        return final_report
