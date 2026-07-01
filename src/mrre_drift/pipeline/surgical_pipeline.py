"""
Surgical MRRE pipeline — end-to-end staged runner.

Stages
------
A  Hidden-state mapping     : Logit Lens (+ optional CLAP) over probe texts → CollapseProfile
                              (safe enhancement / anchoring layers).
B  Fit Surgical MRRE        : compute Stage-1 enhancement + Stage-2 anchoring vectors from real
                              (English, target) prompt pairs and language-forcing pairs.
C  IFL evaluation           : generate target-language responses with and without the
                              intervention; score Script Fidelity Rate → IFL, with optional
                              DSL length-bias correction.
D  Report                   : persist a JSON summary (baseline vs steered, per language).

Every stage checkpoints via the shared :class:`CheckpointManager` and honours ``--resume``.
All data is real (FLORES+ via the mechanistic ``ContrastiveLexicon``); there are no synthetic
or heuristic fallbacks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict as _dc_asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from shared.checkpointing import CheckpointManager
from shared.logging_utils import setup_logging
from shared.seeding import set_seed

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger(__name__)


STAGE_MAP = {
    "A": "Hidden-State Mapping (Logit Lens + CLAP → CollapseProfile)",
    "B": "Fit Surgical MRRE (enhancement + anchoring vectors)",
    "C": "IFL Evaluation (baseline vs surgical, + DSL correction)",
    "D": "Report",
}
ALL_STAGES = list(STAGE_MAP.keys())

# Default language-forcing templates (English vs target-language instruction wrappers).
# Mirrors configs/mrre_stage2.yaml; overridable via config["forcing_prompt_templates"].
_DEFAULT_FORCING = {
    "english": "Please respond in English. {query}",
    "th": "กรุณาตอบเป็นภาษาไทย {query}",
    "my": "မြန်မာဘာသာဖြင့် ဖြေကြားပေးပါ {query}",
    "km": "សូមឆ្លើយជាភាសាខ្មែរ {query}",
    "lo": "ກະລຸນາຕອບເປັນພາສາລາວ {query}",
    "am": "እባክዎ በአማርኛ ይመልሱ {query}",
    "sw": "Tafadhali jibu kwa Kiswahili. {query}",
}


@dataclass
class SurgicalPipelineConfig:
    """Normalised configuration for the Surgical MRRE pipeline."""

    model_id: str = "aisingapore/Llama-SEA-LION-v3-8B-IT"
    device: str = "cuda:0"
    dtype: str = "float16"
    load_in_8bit: bool = False
    target_languages: List[str] = field(default_factory=lambda: ["th", "my"])
    output_dir: str = "results/surgical"
    seed: int = 42

    # Stage A — mapping
    n_probe_texts: int = 16
    top_k: int = 10
    collapse_threshold: float = 0.40
    vision_fusion_fraction: float = 0.0      # 0.0 for text-only LLMs
    anchoring_tail_fraction: float = 0.25
    use_craf: bool = True

    # Stage B — fit
    enhancement_fractions: List[float] = field(default_factory=lambda: [0.40, 0.55, 0.65])
    anchoring_fractions: List[float] = field(default_factory=lambda: [0.75, 0.875])
    # Anchoring-dominant defaults: anchoring (target-ward, tail layers) is the IFL-suppression
    # mechanism; enhancement (english-ward, mid layers) is for reasoning and degrades script
    # fidelity, so it is kept modest. eta bounds per-layer injection to ~alpha*eta of the norm.
    alpha_enhancement: float = 0.3
    alpha_anchoring: float = 0.6
    eta: float = 0.05
    n_prompt_pairs: int = 32
    n_forcing_pairs: int = 32
    forcing_templates: Dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_FORCING))

    # Stage C — IFL eval
    n_samples_per_language: int = 100
    max_new_tokens: int = 128
    sfr_threshold: float = 0.5
    ifl_probe_mode: str = "forcing"   # "forcing" (English query + target instruction) | "native"
    dsl_enabled: bool = True
    length_bins: List[int] = field(default_factory=lambda: [0, 10, 50, 200])

    def to_dict(self) -> Dict:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, cfg: dict) -> "SurgicalPipelineConfig":
        model_cfg = cfg.get("model", {})
        mapping = cfg.get("hidden_state_mapping", cfg.get("mapping", {}))
        surgical = cfg.get("mrre_drift", cfg.get("surgical", {}))
        ifl = cfg.get("ifl_validation", cfg.get("ifl", {}))
        dsl = ifl.get("dsl", {})
        templates = dict(_DEFAULT_FORCING)
        templates.update(cfg.get("forcing_prompt_templates", {}))
        return cls(
            model_id=model_cfg.get("model_id", model_cfg.get("name", cls.model_id)),
            device=model_cfg.get("device", "cuda:0"),
            dtype=model_cfg.get("dtype", model_cfg.get("torch_dtype", "float16")),
            load_in_8bit=model_cfg.get("load_in_8bit", False),
            target_languages=cfg.get("target_languages", mapping.get("target_languages", ["th", "my"])),
            output_dir=cfg.get("output_dir", "results/surgical"),
            seed=cfg.get("seed", cfg.get("project", {}).get("seed", 42)),
            n_probe_texts=mapping.get("n_probe_texts", 16),
            top_k=mapping.get("logit_lens", {}).get("top_k", 10),
            collapse_threshold=mapping.get("logit_lens", {}).get("collapse_threshold", 0.40),
            vision_fusion_fraction=model_cfg.get(
                "vision_fusion_fraction", surgical.get("vision_fusion_fraction", 0.0)
            ),
            anchoring_tail_fraction=mapping.get("collapse_detector", {}).get(
                "anchoring_tail_fraction", 0.25
            ),
            use_craf=mapping.get("use_craf", True),
            enhancement_fractions=surgical.get("enhancement_fractions", [0.40, 0.55, 0.65]),
            anchoring_fractions=surgical.get("anchoring_fractions", [0.75, 0.875]),
            alpha_enhancement=surgical.get("alpha_enhancement", 0.3),
            alpha_anchoring=surgical.get("alpha_anchoring", 0.6),
            eta=surgical.get("eta", 0.05),
            n_prompt_pairs=surgical.get("n_prompt_pairs", 32),
            n_forcing_pairs=surgical.get("n_forcing_pairs", 32),
            forcing_templates=templates,
            n_samples_per_language=ifl.get("n_samples_per_language", 100),
            max_new_tokens=ifl.get("max_new_tokens", 128),
            sfr_threshold=ifl.get("sfr_threshold", 0.5),
            ifl_probe_mode=ifl.get("probe_mode", "forcing"),
            dsl_enabled=dsl.get("enabled", True),
            length_bins=dsl.get("length_bins", [0, 10, 50, 200]),
        )


class SurgicalPipeline:
    """Staged Surgical MRRE pipeline with checkpoint/resume."""

    def __init__(self, config: SurgicalPipelineConfig | dict, resume: bool = False) -> None:
        self.config = (
            SurgicalPipelineConfig.from_dict(config) if isinstance(config, dict) else config
        )
        self.resume = resume
        set_seed(int(self.config.seed))

        self.timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(self.config.output_dir) / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        setup_logging("surgical_pipeline", self.run_dir, level=logging.INFO)
        logger.info("Surgical MRRE Pipeline initialized at: %s", self.run_dir)

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(self.config.output_dir) / "checkpoints",
            project_name="surgical",
        )
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Model + data helpers
    # ------------------------------------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return self._model, self._tokenizer
        from shared.model_loader import ModelLoadSpec, load_model_and_tokenizer

        spec = ModelLoadSpec(
            model_id=self.config.model_id,
            device=self.config.device,
            dtype=self.config.dtype,
            load_in_8bit=self.config.load_in_8bit,
            output_hidden_states=False,   # interpret/* use forward hooks, not return values
            trust_remote_code=True,
        )
        self._model, self._tokenizer = load_model_and_tokenizer(spec)
        return self._model, self._tokenizer

    def _lexicon(self):
        from latent_coordination.data.lexicon import ContrastiveLexicon
        return ContrastiveLexicon(cache_dir=Path(".cache/datasets"))

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _run_stage_a(self) -> "object":
        """Hidden-state mapping → a single representative CollapseProfile."""
        if self.resume and self.checkpoint_manager.exists("stage_a"):
            logger.info("Resuming Stage A from checkpoint.")
            return self.checkpoint_manager.load_latest("stage_a")

        from mrre_drift.interpret.logit_lens import LogitLens
        from mrre_drift.interpret.collapse import CollapseDetector
        from mrre_drift.models.layers import get_transformer_layers

        model, tokenizer = self._ensure_model()
        n_layers = len(get_transformer_layers(model))
        lens = LogitLens(model, tokenizer, top_k=self.config.top_k, device=self.config.device)

        lexicon = self._lexicon()
        probe_texts: List[str] = []
        for lang in self.config.target_languages:
            pairs = lexicon.get_pairs(lang, n_pairs=self.config.n_probe_texts)
            probe_texts.extend([tgt for _, tgt in pairs])
        if not probe_texts:
            raise RuntimeError("Stage A: no probe texts loaded from FLORES+.")

        logger.info("Stage A: scanning %d probe texts with Logit Lens.", len(probe_texts))
        scans = [lens.scan(t) for t in probe_texts]

        craf_profile = None
        if self.config.use_craf:
            from mrre_drift.interpret.craf import CRAF
            craf = CRAF(model, tokenizer, device=self.config.device)
            # Contrast English vs target on the first language for concept directions.
            first_lang = self.config.target_languages[0]
            pairs = lexicon.get_pairs(first_lang, n_pairs=self.config.n_probe_texts)
            en_texts = [en for en, _ in pairs]
            tgt_texts = [tgt for _, tgt in pairs]
            try:
                craf_profile = craf.profile(en_texts, tgt_texts)
            except Exception as exc:  # CRAF is auxiliary; mapping still works from Logit Lens
                logger.warning("CRAF profiling failed (%s); continuing with Logit Lens only.", exc)
                craf_profile = None

        detector = CollapseDetector(
            n_layers=n_layers,
            vision_fusion_fraction=self.config.vision_fusion_fraction,
            collapse_threshold=self.config.collapse_threshold,
            anchoring_tail_fraction=self.config.anchoring_tail_fraction,
        )
        profile = detector.detect_from_scans(scans, craf_profile=craf_profile)
        logger.info("Stage A complete. %s", profile.summary())
        self.checkpoint_manager.save(profile, "stage_a")
        return profile

    def _build_forcing_pairs(self, lexicon, lang: str) -> List[Tuple[str, str]]:
        en_tmpl = self.config.forcing_templates.get("english", _DEFAULT_FORCING["english"])
        tgt_tmpl = self.config.forcing_templates.get(lang)
        if tgt_tmpl is None:
            raise RuntimeError(
                f"No forcing template for language '{lang}'. Add it under "
                f"forcing_prompt_templates in the config."
            )
        pairs = lexicon.get_pairs(lang, n_pairs=self.config.n_forcing_pairs)
        return [(en_tmpl.format(query=en), tgt_tmpl.format(query=en)) for en, _ in pairs]

    def _run_stage_b(self, collapse) -> Dict:
        """Fit enhancement + anchoring vectors and persist them."""
        if self.resume and self.checkpoint_manager.exists("stage_b"):
            logger.info("Resuming Stage B from checkpoint.")
            return self.checkpoint_manager.load_latest("stage_b")

        from mrre_drift.mrre.surgical import SurgicalMRRE, SurgicalMRREConfig

        model, tokenizer = self._ensure_model()
        lexicon = self._lexicon()

        # Prompt pairs (Stage 1): real (English, target) parallel text.
        primary = self.config.target_languages[0]
        prompt_pairs = lexicon.get_pairs(primary, n_pairs=self.config.n_prompt_pairs)
        # Forcing pairs (Stage 2): English vs target-language instruction wrappers.
        forcing_pairs: List[Tuple[str, str]] = []
        for lang in self.config.target_languages:
            forcing_pairs.extend(self._build_forcing_pairs(lexicon, lang))

        cfg = SurgicalMRREConfig(
            vision_fusion_fraction=self.config.vision_fusion_fraction,
            enhancement_fractions=self.config.enhancement_fractions,
            anchoring_fractions=self.config.anchoring_fractions,
            alpha_enhancement=self.config.alpha_enhancement,
            alpha_anchoring=self.config.alpha_anchoring,
            eta=self.config.eta,
        )
        surgical = SurgicalMRRE(
            model, tokenizer, collapse=collapse, config=cfg, device=self.config.device
        )
        logger.info(
            "Stage B: fitting on %d prompt pairs, %d forcing pairs | %s",
            len(prompt_pairs), len(forcing_pairs), surgical,
        )
        surgical.fit(prompt_pairs, forcing_pairs)

        vec_dir = self.run_dir / "vectors"
        surgical.save(vec_dir)
        meta = {
            "vectors_dir": str(vec_dir),
            "enhancement_layer_ids": surgical.enhancement_layer_ids,
            "anchoring_layer_ids": surgical.anchoring_layer_ids,
            "enhancement_norms": surgical.enhancement_norms(),
            "anchoring_norms": surgical.anchoring_norms(),
        }
        self.checkpoint_manager.save(meta, "stage_b")
        logger.info("Stage B complete. Vectors saved to %s", vec_dir)
        return meta

    def _run_stage_c(self, collapse, stage_b_meta: Dict) -> Dict:
        """IFL evaluation: baseline vs surgical, with optional DSL correction."""
        if self.resume and self.checkpoint_manager.exists("stage_c"):
            logger.info("Resuming Stage C from checkpoint.")
            return self.checkpoint_manager.load_latest("stage_c")

        from mrre_drift.mrre.surgical import SurgicalMRRE, SurgicalMRREConfig
        from mrre_drift.eval.ifl import IFLValidator
        from mrre_drift.eval.dsl import DSLCorrector

        model, tokenizer = self._ensure_model()
        lexicon = self._lexicon()

        # IFL probes. Two modes:
        #   "forcing" (default): English source wrapped in a target-language instruction
        #       ("answer in Thai: <english>"). This is the canonical IFL scenario — it
        #       creates real drift pressure toward English, which Stage-2 anchoring suppresses.
        #   "native": raw target-language FLORES+ text (no drift pressure; baseline IFL ~ 0,
        #       so the intervention can only hurt — kept for diagnostics only).
        prompts_by_lang: Dict[str, List[str]] = {}
        for lang in self.config.target_languages:
            pairs = lexicon.get_pairs(lang, n_pairs=self.config.n_samples_per_language)
            if self.config.ifl_probe_mode == "native":
                prompts_by_lang[lang] = [tgt for _, tgt in pairs]
            else:
                tgt_tmpl = self.config.forcing_templates.get(lang)
                if tgt_tmpl is None:
                    raise RuntimeError(
                        f"No forcing template for language '{lang}'. Add it under "
                        f"forcing_prompt_templates, or set ifl_probe_mode: native."
                    )
                prompts_by_lang[lang] = [tgt_tmpl.format(query=en) for en, _ in pairs]

        validator = IFLValidator(
            model, tokenizer, device=self.config.device,
            sfr_threshold=self.config.sfr_threshold,
            max_new_tokens=self.config.max_new_tokens,
        )

        # Reload the fitted intervention from the saved vectors.
        cfg = SurgicalMRREConfig(
            vision_fusion_fraction=self.config.vision_fusion_fraction,
            enhancement_fractions=self.config.enhancement_fractions,
            anchoring_fractions=self.config.anchoring_fractions,
            alpha_enhancement=self.config.alpha_enhancement,
            alpha_anchoring=self.config.alpha_anchoring,
            eta=self.config.eta,
        )
        surgical = SurgicalMRRE(
            model, tokenizer, collapse=collapse, config=cfg, device=self.config.device
        )
        surgical.load(stage_b_meta["vectors_dir"])

        # Per-(model, lang, condition) caching so changing --languages (or a crash) reuses
        # already-evaluated languages. Probe mode is part of the key (different probe → recompute).
        from mrre_drift.eval.ifl import IFLReport, IFLLanguageResult
        model_slug = "".join(
            c if (c.isalnum() or c in "-_.") else "_" for c in str(self.config.model_id)
        )

        def _eval_condition(condition: str, intervention=None) -> IFLReport:
            report = IFLReport(condition=condition)
            for lang, prompts in prompts_by_lang.items():
                key = (f"surg::{model_slug}::{self.config.ifl_probe_mode}::"
                       f"n{self.config.n_samples_per_language}::lang::{lang}::cond::{condition}")
                if self.checkpoint_manager.has_result(key):
                    report.by_language[lang] = IFLLanguageResult(**self.checkpoint_manager.get_result(key))
                    logger.info("IFL [%s] lang=%s loaded from cache.", condition, lang)
                    continue
                sub = validator.evaluate({lang: prompts}, condition=condition, intervention=intervention)
                res = sub.by_language[lang]
                report.by_language[lang] = res
                self.checkpoint_manager.cache_result(key, _dc_asdict(res))
            return report

        baseline = _eval_condition("baseline")
        steered = _eval_condition("mrre_drift", intervention=surgical)

        result = {
            "baseline": baseline.to_dict(),
            "mrre_drift": steered.to_dict(),
            "ifl_reduction_macro": baseline.macro_ifl_rate - steered.macro_ifl_rate,
        }

        if self.config.dsl_enabled:
            corrector = DSLCorrector(self.config.length_bins)
            dsl_out = {}
            for lang in self.config.target_languages:
                b = baseline.by_language.get(lang)
                s = steered.by_language.get(lang)
                if b is None or s is None:
                    continue
                b_flags = [1.0 if v < self.config.sfr_threshold else 0.0 for v in b.sfr_values]
                s_flags = [1.0 if v < self.config.sfr_threshold else 0.0 for v in s.sfr_values]
                b_dsl = corrector.correct(b.output_lengths, b_flags)
                # Debias the steered condition against the baseline length distribution.
                s_dsl = corrector.correct(
                    s.output_lengths, s_flags, reference_weights=b_dsl.reference_weights
                )
                dsl_out[lang] = {
                    "baseline": b_dsl.to_dict(),
                    "mrre_drift": s_dsl.to_dict(),
                }
            result["dsl_correction"] = dsl_out

        self.checkpoint_manager.save(result, "stage_c")
        logger.info(
            "Stage C complete. Macro IFL: baseline=%.3f → steered=%.3f (Δ=%.3f)",
            baseline.macro_ifl_rate, steered.macro_ifl_rate,
            result["ifl_reduction_macro"],
        )
        return result

    def _run_stage_d(self, collapse, stage_b_meta: Dict, eval_result: Dict) -> Dict:
        """Persist a consolidated JSON report."""
        report = {
            "pipeline": "mrre_drift",
            "version": __version__,
            "timestamp_utc": self.timestamp,
            "model_id": self.config.model_id,
            "target_languages": self.config.target_languages,
            "collapse_profile": {
                "n_layers": collapse.n_layers,
                "vision_fusion_cutoff": collapse.vision_fusion_cutoff,
                "collapse_onset_layer": collapse.collapse_onset_layer,
                "peak_collapse_layer": collapse.peak_collapse_layer,
                "safe_enhancement_layers": collapse.safe_enhancement_layers,
                "safe_anchoring_layers": collapse.safe_anchoring_layers,
            },
            "fit": stage_b_meta,
            "evaluation": eval_result,
        }
        report_path = self.run_dir / "final_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Stage D complete. Report → %s", report_path)
        return report

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self, stages: Optional[List[str]] = None) -> Dict:
        stages = stages or ALL_STAGES
        logger.info("Running Surgical MRRE pipeline. Stages: %s", stages)

        collapse = None
        stage_b_meta: Dict = {}
        eval_result: Dict = {}

        if "A" in stages:
            collapse = self._run_stage_a()
        elif self.checkpoint_manager.exists("stage_a"):
            collapse = self.checkpoint_manager.load_latest("stage_a")

        if "B" in stages:
            if collapse is None:
                raise RuntimeError("Stage B requires a CollapseProfile; run Stage A first.")
            stage_b_meta = self._run_stage_b(collapse)
        elif self.checkpoint_manager.exists("stage_b"):
            stage_b_meta = self.checkpoint_manager.load_latest("stage_b")

        if "C" in stages:
            if collapse is None or not stage_b_meta:
                raise RuntimeError("Stage C requires Stages A and B outputs.")
            eval_result = self._run_stage_c(collapse, stage_b_meta)
        elif self.checkpoint_manager.exists("stage_c"):
            eval_result = self.checkpoint_manager.load_latest("stage_c")

        report: Dict = {}
        if "D" in stages:
            if collapse is None:
                raise RuntimeError("Stage D requires earlier stage outputs.")
            report = self._run_stage_d(collapse, stage_b_meta, eval_result)

        return report
