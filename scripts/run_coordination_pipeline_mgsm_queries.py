"""Run the latent_coordination pipeline with MGSM queries for training stages.

This is intentionally separate from scripts/run_coordination_pipeline.py so the
shared pipeline remains untouched. It swaps the Stage-B/C/D query source from
gated FLORES+ to the enabled MGSM benchmark questions in the provided config.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import run_coordination_pipeline as base
from latent_coordination.latent_space.universal_space import UniversalLatentHub
from latent_coordination.pipeline.coordination_pipeline import CoordinationPipeline
from latent_coordination.topology.cvae_prior import CVAETopologyPrior, TrainingConfig


logger = logging.getLogger(__name__)


def _disable_auto_device_map_for_agent_pinning(scan_path: str) -> None:
    """Keep per-agent YAML devices authoritative for this alternate runner.

    The shared model loader reads compute_scan.json and turns any 2-GPU machine
    into device_map="auto". That is useful for one huge model, but this MGSM
    setup already assigns separate agents to cuda:0/cuda:1. Auto-sharding one
    quantized agent across GPU/CPU can leave tokenizer inputs and embedding
    weights on different devices, causing CUDA/CPU index_select failures.
    """
    path = Path(scan_path)
    if not path.exists():
        return
    try:
        scan = json.loads(path.read_text())
        if int(scan.get("device_count", 0)) <= 1:
            return
        scan["actual_device_count"] = scan.get("device_count")
        scan["actual_devices"] = scan.get("devices", [])
        scan["device_count"] = 1
        scan["devices"] = (scan.get("devices") or [])[:1]
        path.write_text(json.dumps(scan, indent=2))
        logger.info(
            "Pinned agent placement by disabling shared-loader device_map='auto' in %s.",
            path,
        )
    except Exception as exc:
        logger.warning("Could not pin agent device placement via %s: %s", path, exc)


def _pin_quantized_model_loads_to_agent_devices() -> None:
    """Patch the shared loader so quantized agents stay on their YAML device.

    This alternate runner is intentionally a multi-agent placement experiment:
    translation on cuda:0, reasoning/safety on cuda:1. The shared loader's
    compute-scan path is designed for single-model sharding and can set
    device_map="auto" before loading. Passing an explicit map prevents that
    path and keeps each quantized model internally consistent.
    """
    try:
        import shared.model_loader as model_loader
    except Exception as exc:
        logger.warning("Could not import shared.model_loader for placement patch: %s", exc)
        return

    original = model_loader.load_model_and_tokenizer

    if getattr(original, "_mgsm_query_pinned", False):
        return

    def pinned_load_model_and_tokenizer(spec):
        if (getattr(spec, "load_in_8bit", False) or getattr(spec, "load_in_4bit", False)) and getattr(spec, "device_map", None) is None:
            spec.device_map = {"": getattr(spec, "device", None) or "cuda:0"}
            logger.info(
                "Pinned quantized model '%s' to device_map=%s for MGSM-query runner.",
                getattr(spec, "model_id", "<unknown>"),
                spec.device_map,
            )
        return original(spec)

    pinned_load_model_and_tokenizer._mgsm_query_pinned = True
    model_loader.load_model_and_tokenizer = pinned_load_model_and_tokenizer


def _patch_cvae_router_inputs_to_prior_device() -> None:
    """Make CVAE topology routing device-safe for the alternate runner."""
    try:
        import latent_coordination.orchestration.router as router_mod
    except Exception as exc:
        logger.warning("Could not import router for CVAE device patch: %s", exc)
        return

    cls = router_mod.AdaptiveOrchestrator
    original = cls._sample_topology_for
    if getattr(original, "_mgsm_query_device_safe", False):
        return

    def sample_topology_for_device_safe(self, task):
        if self.topology_prior is None or self.topology_query_encoder is None:
            return original(self, task)

        # CheckpointManager restores modules with map_location="cpu", but
        # CVAETopologyPrior also serializes its internal `_device` field. If the
        # parameters are CPU while `_device` still says cuda, sample_topology()
        # creates z on cuda and q_emb on CPU, then torch.cat explodes. Keep both
        # the module parameters and `_device` synchronized to the router device.
        target_device = self.device
        param_device = next(self.topology_prior.parameters()).device
        internal_device = getattr(self.topology_prior, "_device", param_device)
        if param_device != target_device or internal_device != target_device:
            if hasattr(self.topology_prior, "to_device"):
                self.topology_prior.to_device(str(target_device))
            else:
                self.topology_prior.to(target_device)
                self.topology_prior._device = torch.device(target_device)
        prior_device = target_device
        Q = self.topology_query_encoder(task.query)
        if Q.dim() == 1:
            Q = Q.unsqueeze(0)
        Q = Q.long().to(prior_device)

        geo = None
        if getattr(self.topology_prior.config, "geo_dim", 0) > 0:
            if self.geo_profile is None:
                raise RuntimeError(
                    "The topology prior conditions on Geo_L (geo_dim>0) but no "
                    "GeoProfile is attached to the router."
                )
            geo = self.geo_profile.vector(task.target_language or "en").unsqueeze(0).to(prior_device)

        adj = self.topology_prior.sample_topology(Q, n_samples=1, geo=geo)
        return adj[0].detach().cpu()

    sample_topology_for_device_safe._mgsm_query_device_safe = True
    cls._sample_topology_for = sample_topology_for_device_safe


def _patch_mgsm_reasoning_protected_eval() -> None:
    """Keep adaptive CVAE topology, but make MGSM answer-bearing.

    CVAE may still decide whether translation/safety participate. For MGSM,
    however, the scored task output must come from the reasoning agent, not from
    a downstream translator. This preserves adaptive topology as a collaboration
    prior while preventing the math-QA failure mode where translation is last
    and the benchmark scores a rephrased problem as the answer.
    """
    try:
        import latent_coordination.eval.benchmark_runner as bench_mod
        from latent_coordination.agents.base_agent import AgentTask
        from latent_coordination.eval.scoring import is_safety_response
        from latent_coordination.orchestration.router import RoutingPlan
    except Exception as exc:
        logger.warning("Could not patch MGSM reasoning-protected eval: %s", exc)
        return

    cls = bench_mod.MultiAgentBenchmarkRunner
    if getattr(cls, "_mgsm_reasoning_protected", False):
        return

    original_single = cls._process_task_single_agent
    original_token = cls._process_task_token_based
    original_latent = cls._process_task_latent

    def is_math_reasoning_task(task) -> bool:
        return ((getattr(task, "metadata", None) or {}).get("benchmark") in {"mgsm", "afrimgsm"})

    def role_of(router, aid: str) -> str:
        return getattr(router.agents[aid].config, "role", "")

    def reasoning_agent_id(router):
        for aid, agent in router.agents.items():
            if getattr(agent.config, "role", None) == "reasoning":
                return aid
        return None

    def reasoning_protected_plan(router, task):
        plan = router.route(task)
        if not is_math_reasoning_task(task):
            return plan

        selected = list(dict.fromkeys(plan.execution_order))
        rid = reasoning_agent_id(router)
        if rid is not None and rid not in selected:
            selected.append(rid)

        # Preserve CVAE-selected participants, but make the answer-bearing math
        # solver run after translators and before safety gates. Unknown roles
        # remain before reasoning so their information can feed the solver.
        translations = [aid for aid in selected if role_of(router, aid) == "translation"]
        reasoning = [aid for aid in selected if role_of(router, aid) == "reasoning"]
        safety = [aid for aid in selected if role_of(router, aid) == "safety"]
        other = [
            aid for aid in selected
            if role_of(router, aid) not in {"translation", "reasoning", "safety"}
        ]
        ordered = translations + other + reasoning + safety
        logger.info(
            "MGSM reasoning-protected route for %s: CVAE=%s protected=%s",
            task.task_id,
            plan.execution_order,
            ordered,
        )
        return RoutingPlan(
            task_id=plan.task_id,
            selected_agents=list(ordered),
            execution_order=list(ordered),
            estimated_cost=plan.estimated_cost,
            routing_confidence=plan.routing_confidence,
        )

    def select_reasoning_answer(router, task, responses):
        if not responses:
            return None
        if is_math_reasoning_task(task):
            for resp in reversed(responses):
                aid = getattr(resp, "agent_id", "")
                meta = getattr(resp, "metadata", None) or {}
                if meta.get("role") == "reasoning" or (
                    aid in router.agents and role_of(router, aid) == "reasoning"
                ):
                    return resp
        for resp in reversed(responses):
            if not is_safety_response(resp):
                return resp
        return responses[-1]

    def process_task_single_agent_reasoning_baseline(self, router, task):
        if not is_math_reasoning_task(task):
            return original_single(self, router, task)

        rid = reasoning_agent_id(router)
        if rid is None:
            return original_single(self, router, task)

        agent = router.agents[rid]
        resp = agent.process(task)
        cost = self._count_tokens(resp.output_text, agent)
        safety = [resp] if is_safety_response(resp) else []
        logger.info(
            "Math single-agent baseline for %s uses reasoning agent %s.",
            task.task_id,
            rid,
        )
        return resp, safety, cost

    def process_task_token_based_reasoning_protected(self, router, task):
        if not is_math_reasoning_task(task):
            return original_token(self, router, task)

        plan = reasoning_protected_plan(router, task)
        context = task.context or ""
        step_responses = []
        token_cost = 0.0
        for aid in plan.execution_order:
            agent = router.agents[aid]
            text_task = AgentTask(
                task_id=f"{task.task_id}_token_{aid}",
                query=task.query,
                context=context,
                latent_state=None,
                target_language=task.target_language,
            )
            resp = agent.process(text_task)
            context = resp.output_text
            token_cost += self._count_tokens(resp.output_text, agent)
            step_responses.append(resp)
        answer = select_reasoning_answer(router, task, step_responses)
        safety = [r for r in step_responses if is_safety_response(r)]
        return answer, safety, token_cost

    def process_task_latent_reasoning_protected(self, router, task, universal_space):
        if not is_math_reasoning_task(task):
            return original_latent(self, router, task, universal_space)

        plan = reasoning_protected_plan(router, task)
        orch_result = router.execute(task, plan, universal_space)
        chain = orch_result.agent_responses
        if not chain:
            return None, [], 0.0
        answer = select_reasoning_answer(router, task, chain)
        safety = [r for r in chain if is_safety_response(r)]
        return answer, safety, 0.0

    cls._process_task_single_agent = process_task_single_agent_reasoning_baseline
    cls._process_task_token_based = process_task_token_based_reasoning_protected
    cls._process_task_latent = process_task_latent_reasoning_protected
    cls._mgsm_reasoning_protected = True


class MGSMQueryCoordinationPipeline(CoordinationPipeline):
    """CoordinationPipeline variant that avoids gated FLORES+ dependencies."""

    def _load_math_query_corpus(
        self, max_per_language: Optional[int] = None
    ) -> Tuple[List[str], List[str]]:
        """Load enabled MGSM/AfriMGSM questions and language labels."""
        benchmarks = self._raw_config.get("benchmarks", {}) or {}
        mgsm_cfg = benchmarks.get("mgsm", {}) or {}
        afrimgsm_cfg = benchmarks.get("afrimgsm", {}) or {}
        if not mgsm_cfg.get("enabled") and not afrimgsm_cfg.get("enabled"):
            raise RuntimeError(
                "MGSM-query runner requires benchmarks.mgsm.enabled=true or "
                "benchmarks.afrimgsm.enabled=true in the config."
            )

        from latent_coordination.eval.correctness import (
            AFRIMGSM_SUPPORTED_LANGUAGES,
            MGSM_SUPPORTED_LANGUAGES,
            load_afrimgsm_tasks,
            load_mgsm_tasks,
        )

        queries: List[str] = []
        query_langs: List[str] = []

        def resolve_n(cfg: Dict) -> Optional[int]:
            configured_n = cfg.get("n_samples")
            if configured_n is not None and max_per_language is not None:
                return min(int(configured_n), int(max_per_language))
            if configured_n is not None:
                return int(configured_n)
            return max_per_language

        if mgsm_cfg.get("enabled"):
            langs = list(mgsm_cfg.get("languages") or self.config.target_languages or ["en"])
            unknown = [lang for lang in langs if lang not in MGSM_SUPPORTED_LANGUAGES]
            if unknown:
                raise ValueError(
                    f"MGSM does not cover {unknown}; supported: "
                    f"{sorted(MGSM_SUPPORTED_LANGUAGES)}"
                )
            n = resolve_n(mgsm_cfg)
            for lang in langs:
                items = load_mgsm_tasks(language=lang, n=n)
                queries.extend(item["question"] for item in items)
                query_langs.extend([lang] * len(items))
                logger.info("Loaded %d MGSM training queries for '%s'.", len(items), lang)

        if afrimgsm_cfg.get("enabled"):
            langs = list(afrimgsm_cfg.get("languages") or self.config.target_languages or ["sw"])
            unknown = [lang for lang in langs if lang not in AFRIMGSM_SUPPORTED_LANGUAGES]
            if unknown:
                raise ValueError(
                    f"AfriMGSM does not cover {unknown}; supported: "
                    f"{sorted(AFRIMGSM_SUPPORTED_LANGUAGES)}"
                )
            n = resolve_n(afrimgsm_cfg)
            for lang in langs:
                items = load_afrimgsm_tasks(language=lang, n=n)
                queries.extend(item["question"] for item in items)
                query_langs.extend([lang] * len(items))
                logger.info("Loaded %d AfriMGSM training queries for '%s'.", len(items), lang)

        if not queries:
            raise RuntimeError("Math query corpus is empty; check benchmarks config.")
        return queries, query_langs

    def _load_mgsm_query_corpus(
        self, max_per_language: Optional[int] = None
    ) -> Tuple[List[str], List[str]]:
        """Backward-compatible alias for older call sites."""
        return self._load_math_query_corpus(max_per_language=max_per_language)

    def _run_stage_b(self) -> CVAETopologyPrior:
        """Stage B: train CVAE topology prior on enabled math questions."""
        if self.resume and self.checkpoint_manager.exists("stage_b"):
            logger.info("Resuming Stage B from checkpoints.")
            return self.checkpoint_manager.load_latest("stage_b")

        logger.info("Running Stage B: Training CVAE topology prior on math queries.")
        cvae_cfg = self._raw_config.get("cvae", {}) or {}
        train_cfg = cvae_cfg.get("training", {}) or {}

        geo_profile = None
        if cvae_cfg.get("condition_on_geometry"):
            from latent_coordination.topology.geo_profile import GeoProfile

            geo_path = cvae_cfg.get("geo_profile_path")
            if not geo_path:
                raise ValueError(
                    "cvae.condition_on_geometry=true requires cvae.geo_profile_path."
                )
            geo_profile = GeoProfile(geo_path)
            self._geo_profile = geo_profile

        t_config = TrainingConfig(
            z_dim=self.config.cvae_latent_dim,
            query_dim=int(cvae_cfg.get("query_dim", 64)),
            geo_dim=geo_profile.geo_dim if geo_profile is not None else 0,
            max_n_agents=3,
        )
        cvae_prior = CVAETopologyPrior(config=t_config).to(self.config.device)

        real_queries, query_langs = self._load_math_query_corpus()
        logger.info("Using %d math queries for CVAE training.", len(real_queries))

        self._cvae_query_vocab_size = t_config.query_vocab_size
        query_tensors = torch.stack([self._encode_cvae_query(q) for q in real_queries])
        target_graphs = torch.stack([
            self._topology_target(q, lang)
            for q, lang in zip(real_queries, query_langs)
        ])
        geo_all = geo_profile.batch(query_langs) if geo_profile is not None else None

        n_epochs = int(train_cfg.get("n_epochs", 20))
        lr = float(train_cfg.get("lr", 1e-3))
        batch_size = int(train_cfg.get("batch_size", 8))
        optimizer = torch.optim.Adam(cvae_prior.parameters(), lr=lr)
        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad()
            idx = torch.randperm(len(query_tensors))[:batch_size]
            batch_queries = query_tensors[idx].to(self.config.device)
            batch_g = target_graphs[idx].to(self.config.device)
            batch_geo = geo_all[idx].to(self.config.device) if geo_all is not None else None

            recon_g, mu, logvar = cvae_prior(batch_g, batch_queries, geo=batch_geo)
            loss, _ = cvae_prior.compute_loss(recon_g, batch_g, mu, logvar)
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0:
                logger.info(
                    "CVAE Epoch %d/%d | math-query ELBO loss=%.4f",
                    epoch, n_epochs, loss.item(),
                )

        self._cvae_query_tokens = query_tensors
        self._real_queries = real_queries

        self.checkpoint_manager.save(cvae_prior, "stage_b")
        return cvae_prior

    def _train_stage_c_adapters(
        self, universal_space: UniversalLatentHub, router, at_cfg: Dict
    ):
        """Collect hidden states on math questions and train hub adapters."""
        n_samples = int(at_cfg.get("n_samples", 64))
        texts, _ = self._load_math_query_corpus(max_per_language=n_samples)
        texts = texts[:n_samples]
        if not texts:
            raise RuntimeError("No math questions available for adapter training.")

        states_by_agent: Dict[str, torch.Tensor] = {}
        for aid, agent in router.agents.items():
            if not universal_space.is_registered(aid):
                continue
            rows = []
            for text in texts:
                hs = agent.extract_hidden_states(text, layer_ids=[-1])[-1]
                rows.append(hs.float().mean(dim=1).squeeze(0).cpu())
            states_by_agent[aid] = torch.stack(rows)

        logger.info(
            "Collected row-aligned hidden states for %d agents on %d math prompts; "
            "training adapters (n_epochs=%s, lr=%s, batch_size=%s).",
            len(states_by_agent), len(texts),
            at_cfg.get("n_epochs", 50), at_cfg.get("lr", 1e-3),
            at_cfg.get("batch_size", 32),
        )
        losses = universal_space.fit_adapters(
            states_by_agent,
            n_epochs=int(at_cfg.get("n_epochs", 50)),
            lr=float(at_cfg.get("lr", 1e-3)),
            batch_size=int(at_cfg.get("batch_size", 32)),
            dae_sigma=float(at_cfg.get("dae_sigma", 0.1)),
            mu_cka=float(at_cfg.get("mu_cka", 1.0)),
            gamma_dae=float(at_cfg.get("gamma_dae", 1.0)),
        )
        logger.info("Adapter training complete | final losses: %s", losses)
        return states_by_agent, texts

    def _run_stage_d(self, router) -> Dict:
        """Stage D: fit intent centroids on math questions."""
        if self.resume and self.checkpoint_manager.exists("stage_d"):
            logger.info("Resuming Stage D from checkpoints.")
            restored = self.checkpoint_manager.load_latest("stage_d")
            if isinstance(restored, dict) and "centroids" in restored:
                self._apply_stage_d_state(router, restored)
                return restored
            logger.info("Legacy Stage D checkpoint found (no centroid state); recomputing.")

        logger.info("Running Stage D: Intent centroids mapping on math queries.")
        from latent_coordination.orchestration.router import encode_query_bow

        if getattr(self, "_real_queries", None):
            real_queries = self._real_queries
        else:
            real_queries, _ = self._load_math_query_corpus()
            self._real_queries = real_queries

        historical_embeddings = torch.stack([encode_query_bow(q) for q in real_queries])
        self._intent_query_embeddings = historical_embeddings

        n_clusters = int(
            self._raw_config.get("orchestration", {}).get("n_intent_centroids", 8)
        )
        router.fit_centroids(historical_embeddings, n_clusters=n_clusters)
        state = {"centroids": router.centroids, "centroid_roles": router.centroid_roles}
        self.checkpoint_manager.save(state, "stage_d")
        return state


def main() -> int:
    args = base.parse_args()
    scan_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "compute_scan.json"
    )
    base.run_compute_scan(scan_path)
    _disable_auto_device_map_for_agent_pinning(scan_path)
    _pin_quantized_model_loads_to_agent_devices()
    _patch_cvae_router_inputs_to_prior_device()
    _patch_mgsm_reasoning_protected_eval()

    cfg = base.load_config(args.config)
    log_cfg = cfg.get("logging", {})
    base._bootstrap_logging(
        log_dir=log_cfg.get("log_dir"),
        level=log_cfg.get("level", "INFO"),
    )
    run_logger = logging.getLogger(__name__)
    run_logger.info("=" * 70)
    run_logger.info("MGSM-query Latent Coordination Pipeline Runner")
    run_logger.info("Config : %s", args.config.resolve())
    run_logger.info("=" * 70)

    try:
        cfg = base.apply_overrides(cfg, args)
        stages = base.resolve_stages(args.stages, args.skip_cvae_training)
    except ValueError as exc:
        run_logger.error("Config/stage error: %s", exc)
        return 1

    if args.languages is not None:
        langs = [l.strip() for l in args.languages.split(",") if l.strip()]
        cfg.setdefault("benchmarks", {}).setdefault("mgsm", {})["languages"] = langs
        run_logger.info("Override benchmarks.mgsm.languages -> %s", langs)

    output_name = Path(cfg.get("project", {}).get("output_dir", "results/coordination")).name
    checkpoint_dir = Path(".cache/checkpoints/bench_suite") / output_name
    cfg.setdefault("checkpointing", {})["checkpoint_dir"] = str(checkpoint_dir)
    run_logger.info("Override checkpointing.checkpoint_dir -> %s", checkpoint_dir)

    run_logger.info("Stages to run: %s", stages)
    run_logger.info("Stage descriptions: %s", {s: base.STAGE_MAP[s] for s in stages})

    if args.dry_run:
        run_logger.info("[DRY-RUN] Imports OK. Exiting without running pipeline.")
        return 0

    try:
        from latent_coordination.utils.logging_utils import setup_logging

        setup_logging(cfg)
        run_logger = logging.getLogger(__name__)
    except ImportError:
        run_logger.warning(
            "latent_coordination.utils.logging_utils not found; using bootstrap logger."
        )

    output_dir = cfg.get("project", {}).get("output_dir", "results/coordination")
    start_time = time.time()
    success = False
    error_msg: Optional[str] = None
    try:
        pipeline = MGSMQueryCoordinationPipeline(config=cfg, resume=args.resume)
        run_logger.info("MGSM-query pipeline instantiated. Starting execution...")
        pipeline.run(stages=stages)
        success = True
        run_logger.info("Pipeline completed successfully.")
    except KeyboardInterrupt:
        error_msg = "Interrupted by user (KeyboardInterrupt)."
        run_logger.warning(error_msg)
    except Exception as exc:  # noqa: BLE001
        error_msg = f"{type(exc).__name__}: {exc}"
        run_logger.error("Pipeline failed: %s", error_msg)
        run_logger.debug(traceback.format_exc())
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        run_logger.info("Total elapsed time: %.2f s (%.1f min)", elapsed, elapsed / 60)
        summary = base._build_run_summary(
            cfg=cfg,
            stages=stages,
            start_time=start_time,
            end_time=end_time,
            success=success,
            error=error_msg,
        )
        try:
            base.save_run_summary(summary, output_dir)
        except Exception as summ_exc:  # noqa: BLE001
            run_logger.warning("Could not save run summary: %s", summ_exc)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
