"""Latent Coordination multi-agent orchestration and coordination research pipeline."""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from latent_coordination.agents.base_agent import AgentConfig, AgentTask
from latent_coordination.agents.specialized_agents import TranslationAgent, ReasoningAgent, SafetyAgent
from latent_coordination.latent_space.universal_space import UniversalLatentHub
from latent_coordination.latent_space.adapter import AdapterConfig, LatentAdapter
from latent_coordination.topology.cvae_prior import CVAETopologyPrior, TrainingConfig
from latent_coordination.topology.graph_utils import GraphUtils
from latent_coordination.orchestration.router import AdaptiveOrchestrator, TOPOLOGY_ROLE_INDEX
from latent_coordination.orchestration.task_decomposer import TaskDecomposer
from latent_coordination.eval.efficiency_metrics import EfficiencyAnalyzer
from latent_coordination.eval.benchmark_runner import MultiAgentBenchmarkRunner
from latent_coordination.viz.topology_plots import TopologyPlotter
from latent_coordination.viz.efficiency_plots import EfficiencyPlotter
from latent_coordination.viz.latent_space_plots import LatentSpacePlotter
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
class CoordinationPipelineConfig:
    """Configuration orchestrating CVAE prior training and orchestration ablations."""
    cvae_latent_dim: int = 16
    universal_space_dim: int = 128
    target_languages: List[str] = field(default_factory=lambda: ["th", "my", "km"])
    output_dir: str = "results/coordination"
    device: str = "cpu"
    checkpoint_interval: int = 1

    def to_dict(self) -> Dict:
        return asdict(self)


def derive_headline_framing(
    results_by_mode: Dict[str, Dict[str, float]],
    agent_model_ids: List[str],
) -> Dict:
    """Narrative defensive gating (LRL-MRRE-MAS strategy.md §7.2).

    The strategy requires this as *a conditional in the report generator, not
    something decided manually after the fact*: if the strong single-agent
    baseline matches or exceeds the latent multi-agent system on end-task
    accuracy, the headline framing pivots from accuracy to (a) inter-agent
    token-overhead reduction, (b) communication-bandwidth savings, and (c) the
    heterogeneous cross-architecture regime a single-LLM simulation cannot
    capture (KV-cache cannot be shared across architectures).

    Returns a dict always containing ``framing`` (``"accuracy_headline"`` |
    ``"efficiency_fallback"`` | ``"undetermined"``) plus the measured evidence
    behind the decision. Never raises on missing modes — an incomplete run
    yields ``"undetermined"`` with the reason recorded.
    """
    baseline = results_by_mode.get("single_agent_baseline") or {}
    ours = results_by_mode.get("latent_based_mas_ours") or {}
    token_mas = results_by_mode.get("token_based_mas") or {}

    if "accuracy" not in baseline or "accuracy" not in ours:
        missing = [
            m for m in ("single_agent_baseline", "latent_based_mas_ours")
            if "accuracy" not in (results_by_mode.get(m) or {})
        ]
        return {
            "framing": "undetermined",
            "reason": f"missing accuracy for mode(s): {missing}",
        }

    distinct_models = list(dict.fromkeys(agent_model_ids))
    heterogeneous_pool = len(distinct_models) > 1
    framing: Dict = {
        "baseline_accuracy": float(baseline["accuracy"]),
        "ours_accuracy": float(ours["accuracy"]),
        "agent_pool_models": distinct_models,
        "heterogeneous_pool": heterogeneous_pool,
    }

    if baseline["accuracy"] >= ours["accuracy"]:
        # Fallback claim structure. Token-overhead reduction is measured
        # against the token-communication MAS (the like-for-like multi-agent
        # comparator), not the single-agent baseline (which has no inter-agent
        # communication to reduce).
        token_cost = token_mas.get("token_cost")
        ours_cost = ours.get("token_cost", 0.0)
        reduction = None
        if token_cost:
            reduction = 1.0 - float(ours_cost) / float(token_cost)
        framing.update({
            "framing": "efficiency_fallback",
            "headline_claims": [
                "inter_agent_token_overhead_reduction",
                "communication_bandwidth_savings",
                "heterogeneous_cross_architecture_regime",
            ],
            "token_overhead_reduction_vs_token_mas": reduction,
            "token_cost_token_mas": token_cost,
            "token_cost_ours": ours_cost,
            "single_llm_simulation_caveat": (
                "A single-LLM role-play baseline cannot capture the "
                "heterogeneous cross-architecture regime (e.g. a llama-family "
                "safety agent coordinating with a qwen2-family reasoning "
                "agent): KV-cache reuse does not transfer across architectures, "
                "whereas the universal latent hub is architecture-agnostic."
            ),
            "current_run_exercises_heterogeneous_regime": heterogeneous_pool,
        })
    else:
        framing["framing"] = "accuracy_headline"
    return framing


class CoordinationPipeline:
    """Orchestrates CVAE training, Universal space adapters mapping, intent routing, and benchmark reporting."""

    def __init__(self, config: CoordinationPipelineConfig | dict, resume: bool = False) -> None:
        self._raw_config = config if isinstance(config, dict) else {}
        if isinstance(config, dict):
            self.config = CoordinationPipelineConfig(
                cvae_latent_dim=config.get("cvae", {}).get("latent_dim", 16),
                universal_space_dim=config.get("universal_latent_space", {}).get("universal_dim", 128),
                target_languages=config.get("target_languages", ["th", "my", "km"]),
                output_dir=config.get("project", {}).get("output_dir", "results/coordination"),
                device=config.get("agents", [{"device": "cpu"}])[0].get("device", "cpu"),
                checkpoint_interval=config.get("checkpointing", {}).get("interval_stages", 1)
            )
            # Read model_id from the first named agent in YAML
            self._agent_model_id = config.get("agents", [{"model_id": "Qwen/Qwen3.5-9B"}])[0].get(
                "model_id", "Qwen/Qwen3.5-9B"
            )
        else:
            self.config = config
            self._agent_model_id = "Qwen/Qwen3.5-9B"
        self.resume = resume

        # Reproducibility: seed all RNGs from project.seed (default 42).
        from shared.seeding import set_seed
        set_seed(int(self._raw_config.get("project", {}).get("seed", 42)))

        self.timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(self.config.output_dir) / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        setup_logging("coordination_pipeline", self.run_dir, level=logging.INFO)
        logger.info("Latent Coordination Pipeline initialized at directory: %s", self.run_dir)

        # Checkpoint manager. checkpointing.checkpoint_dir was silently ignored
        # (dev_doc.md §9 gap 6) — every run checkpointed under
        # {output_dir}/checkpoints regardless of config; honor it, keeping the
        # historical location as the fallback when the knob is absent.
        ckpt_cfg = self._raw_config.get("checkpointing", {}) or {}
        ckpt_dir = ckpt_cfg.get("checkpoint_dir") or (Path(self.config.output_dir) / "checkpoints")
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=ckpt_dir,
            project_name="coordination"
        )

    def _flores_cap(self) -> Optional[int]:
        """Per-language FLORES+ task cap from config.

        Reads ``benchmarks.flores_plus.n_samples_per_language``, falling back to
        ``benchmarks.sea_vision.n_samples_per_language``. ``None`` (or a non-positive
        value) means use the full devtest split (1012/language).
        """
        bench = self._raw_config.get("benchmarks", {})
        cap = bench.get("flores_plus", {}).get("n_samples_per_language")
        if cap is None:
            cap = bench.get("sea_vision", {}).get("n_samples_per_language")
        if cap is None or int(cap) <= 0:
            return None
        return int(cap)

    @staticmethod
    def _resolve_agent_hidden_dim(model_id: str) -> int:
        """Return the hidden_size for a HuggingFace model without loading weights."""
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            # VLMs (Qwen3.5, Gemma4, LLaVA) store language hidden_size under text_config
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is not None:
                dim = getattr(text_cfg, "hidden_size", None)
                if dim:
                    return int(dim)
            return int(getattr(cfg, "hidden_size", getattr(cfg, "d_model", 4096)))
        except Exception as exc:
            # Known hidden_size values for the curated sweep, used only when the HF Hub
            # is unreachable. An UNKNOWN model must NOT silently default to a guessed
            # dimension (that would corrupt every adapter downstream) — raise instead.
            known = {
                "SeaLLMs/SeaLLMs-v3-7B-Chat": 3584,
                "aisingapore/Llama-SEA-LION-v3-8B-IT": 4096,
                "aisingapore/Gemma-SEA-LION-v3-9B-IT": 3584,
                "sail/Sailor2-8B-Chat": 3584,
                "scb10x/llama-3-typhoon-v1.5-8b-instruct": 4096,
                "meta-llama/Llama-3.1-8B-Instruct": 4096,
            }
            if model_id in known:
                logger.warning(
                    "Could not fetch AutoConfig for %s (%s); using known hidden_dim=%d.",
                    model_id, exc, known[model_id],
                )
                return known[model_id]
            raise RuntimeError(
                f"Could not resolve hidden_size for '{model_id}' (AutoConfig failed: {exc}) "
                f"and it is not in the known-dimensions table. Ensure the model is "
                f"reachable on the HF Hub or add it to the known map."
            ) from exc

    _STAGE_LETTERS = ("A", "B", "C", "D", "E", "F", "G")

    def run(self, stages: Optional[List[str]] = None) -> Dict:
        """Executes the pipeline stage-by-stage with resume support.

        ``stages`` restricts execution to a subset of A-G (see ``_STAGE_LETTERS``:
        A=setup, B=CVAE training, C=adapter pretraining, D=intent centroids,
        E=multi-agent execution & ablation, F=visualization, G=report). This lettering
        matches scripts/run_coordination_pipeline.py's STAGE_MAP one-to-one — the two
        registries were reconciled when the CLI's stale 8-letter scheme was removed;
        keep them in sync when adding stages.
        Stages A-E produce objects later stages depend on: any stage not in ``stages``
        must have a prior checkpoint to load from, or this raises -- it does not
        silently skip and leave the dependency unset.
        """
        requested = set(s.upper() for s in stages) if stages else set(self._STAGE_LETTERS)
        unknown = requested - set(self._STAGE_LETTERS)
        if unknown:
            raise ValueError(
                f"Unknown stage letter(s) {sorted(unknown)}. Valid: {self._STAGE_LETTERS}."
            )
        logger.info(
            "Executing Latent Coordination Multi-Agent Pipeline (stages=%s). Configuration: %s",
            sorted(requested), self.config,
        )

        def _ensure(letter: str, checkpoint_key: str, compute_fn):
            if letter in requested:
                return compute_fn()
            if self.checkpoint_manager.exists(checkpoint_key):
                logger.info(
                    "Stage %s not requested; loading '%s' from checkpoint instead of recomputing.",
                    letter, checkpoint_key,
                )
                return self.checkpoint_manager.load_latest(checkpoint_key)
            raise RuntimeError(
                f"Stage {letter} was not requested via --stages and no checkpoint "
                f"'{checkpoint_key}' exists to load its output from (needed by a later "
                f"stage). Either include stage {letter} in --stages, or run once without "
                f"--stages first to create the checkpoint, then re-run with --resume."
            )

        # Stage A: System Setup
        router, universal_space = _ensure("A", "stage_a", self._run_stage_a)

        # Stage B: CVAE Topology Training
        cvae_prior = _ensure("B", "stage_b", self._run_stage_b)

        # Stage C: Adapter Pre-training. When loaded from checkpoint, the restored
        # hub REPLACES the Stage-A one — the registrations/adapters live on the hub
        # object itself, and the old bool-sentinel checkpoint lost them (Stage E then
        # ran on a hub with no adapters registered by this stage). Module C/E
        # components (recursive core, drift probe) ride in the same state dict and
        # are re-attached to the router here.
        stage_c_out = _ensure("C", "stage_c", lambda: self._run_stage_c(universal_space, router))
        stage_c_state = self._coerce_stage_c_state(stage_c_out)
        if stage_c_state is not None:
            universal_space = stage_c_state["hub"]
            self._apply_stage_c_state(router, stage_c_state)

        # Stage D: Intent Centroid Mapping. Same pattern: restore fitted centroids
        # onto the router when the stage itself is skipped.
        stage_d_out = _ensure("D", "stage_d", lambda: self._run_stage_d(router))
        if isinstance(stage_d_out, dict) and "centroids" in stage_d_out:
            self._apply_stage_d_state(router, stage_d_out)

        # Stage E: Multi-Agent Execution & Ablation. Checkpoint key is "stage_e";
        # older runs saved it under "stage_f" (a copy-paste key collision with the
        # visualization stage's letter), so fall back to that for existing caches.
        def _ensure_stage_e():
            if "E" in requested:
                return self._run_stage_e(router, universal_space, cvae_prior=cvae_prior)
            for key in ("stage_e", "stage_f"):
                if self.checkpoint_manager.exists(key):
                    logger.info(
                        "Stage E not requested; loading benchmark report from checkpoint '%s'.", key
                    )
                    return self.checkpoint_manager.load_latest(key)
            raise RuntimeError(
                "Stage E was not requested via --stages and no 'stage_e' (or legacy "
                "'stage_f') checkpoint exists to load the benchmark report from. "
                "Include stage E, or run once without --stages first."
            )

        benchmark_report = _ensure_stage_e()

        # Stage F: Visualizations (best-effort, no downstream dependency -- just skip if unrequested)
        if "F" in requested:
            self._run_stage_f(router, universal_space, benchmark_report)

        # Stage G: Report compilation
        if "G" in requested:
            final_report = self._run_stage_g(benchmark_report)
        else:
            logger.info("Stage G not requested; returning the raw benchmark report instead of a compiled final_report.")
            final_report = benchmark_report

        return final_report

    def _remap_agent_devices(self, router: "AdaptiveOrchestrator") -> None:
        """Remap agent devices after checkpoint restore to match current config."""
        agent_devices = {
            a.get("role"): a.get("device", self.config.device)
            for a in self._raw_config.get("agents", [])
        }
        for agent in router.agents.values():
            role = agent.config.role
            if role in agent_devices:
                old = agent.config.device
                new_device = agent_devices[role]
                agent.config.device = new_device
                # _device is set at __init__ time from config.device; must stay in sync
                # so that tokenizer/tensor .to(self._device) calls use the correct GPU.
                agent._device = torch.device(new_device)
                if old != new_device:
                    logger.info(
                        "Remapped agent '%s' device %s → %s",
                        agent.config.agent_id, old, new_device,
                    )

    def _run_stage_a(self) -> Tuple[AdaptiveOrchestrator, UniversalLatentHub]:
        """Stage A: Setup orchestrators, databases, and adapters registry."""
        if self.resume and self.checkpoint_manager.exists("stage_a"):
            logger.info("Resuming Stage A from checkpoints.")
            router, universal_space = self.checkpoint_manager.load_latest("stage_a")
            self._remap_agent_devices(router)
            return router, universal_space

        logger.info("Running Stage A: Launching agent registry and universal space mappings.")
        # orchestration.routing_strategy was silently ignored (dev_doc.md §9
        # gap 6); it now selects the router implementation. Unknown values fail
        # loudly instead of quietly running the default.
        orch_cfg = self._raw_config.get("orchestration", {}) or {}
        # orchestration.parallel_agents promised concurrent agent execution but
        # was never implemented — and cannot be for the latent chain, whose
        # whole design is sequential (each agent consumes the previous agent's
        # transferred hidden state; see AdaptiveOrchestrator.execute). Per the
        # zero-tolerance policy on config that silently lies, `true` fails
        # loudly instead of quietly running sequentially.
        if orch_cfg.get("parallel_agents"):
            raise ValueError(
                "orchestration.parallel_agents=true is not supported: the latent "
                "coordination chain is sequential by design (each agent consumes "
                "the previous agent's transferred latent state, so there is "
                "nothing to parallelize within one task). Remove the knob or set "
                "it to false. For throughput, parallelize across languages/"
                "instances (see scripts/build_experimental_report.py's 2-instance "
                "8-GPU split) instead."
            )
        strategy = orch_cfg.get("routing_strategy", "attention")
        strategy_to_router = {
            "attention": "attention",
            "latent_centroid": "kmeans",
            "cvae_topology": "cvae",
        }
        if strategy not in strategy_to_router:
            raise ValueError(
                f"Unknown orchestration.routing_strategy '{strategy}'. "
                f"Valid: {sorted(strategy_to_router)}."
            )
        router = AdaptiveOrchestrator(
            device=self.config.device, router_type=strategy_to_router[strategy]
        )
        universal_space = UniversalLatentHub(universal_dim=self.config.universal_space_dim)

        # Register specialized agents — hidden_dim must match each agent's OWN model's
        # hidden_size. Each agent in configs/*.yaml carries its own model_id (that's the
        # whole point of configs/latent_coordination_heterogeneous.yaml's mixed
        # llama/qwen2/cohere pool) -- read it per-role here instead of collapsing every
        # agent onto self._agent_model_id (which is only ever the *first* config entry,
        # i.e. the orchestrator's model). Falls back to self._agent_model_id for any
        # role missing from the config, matching the historical homogeneous default.
        agent_model_ids = {
            agent.get("role"): agent.get("model_id", self._agent_model_id)
            for agent in self._raw_config.get("agents", [])
        }
        agent_devices = {
            agent.get("role"): agent.get("device", self.config.device)
            for agent in self._raw_config.get("agents", [])
        }
        agent_8bit = {
            agent.get("role"): agent.get("load_in_8bit", False)
            for agent in self._raw_config.get("agents", [])
        }

        t_model = agent_model_ids.get("translation", self._agent_model_id)
        r_model = agent_model_ids.get("reasoning", self._agent_model_id)
        s_model = agent_model_ids.get("safety", self._agent_model_id)

        t_device = agent_devices.get("translation", self.config.device)
        r_device = agent_devices.get("reasoning", self.config.device)
        s_device = agent_devices.get("safety", self.config.device)

        agent_tokens = {
            agent.get("role"): agent.get("max_new_tokens", 512)
            for agent in self._raw_config.get("agents", [])
        }
        agent_dtype = {
            agent.get("role"): agent.get("torch_dtype", "float16")
            for agent in self._raw_config.get("agents", [])
        }

        t_8bit = agent_8bit.get("translation", False)
        r_8bit = agent_8bit.get("reasoning", False)
        s_8bit = agent_8bit.get("safety", False)

        t_toks = agent_tokens.get("translation", 512)
        r_toks = agent_tokens.get("reasoning", 512)
        s_toks = agent_tokens.get("safety", 512)

        t_dt = agent_dtype.get("translation", "float16")
        r_dt = agent_dtype.get("reasoning", "float16")
        s_dt = agent_dtype.get("safety", "float16")

        t_hidden = self._resolve_agent_hidden_dim(t_model)
        r_hidden = self._resolve_agent_hidden_dim(r_model)
        s_hidden = self._resolve_agent_hidden_dim(s_model)

        # communication.latent_transfer_layer: which decoder layer's
        # generation-time hidden states are captured for the latent hand-off
        # (was silently ignored — dev_doc.md §9 gap 6).
        transfer_layer = int(
            self._raw_config.get("communication", {}).get("latent_transfer_layer", -1)
        )

        # orchestration.timeout_per_agent_s: wall-clock cap on a single agent's
        # generate() call (was silently ignored — dev_doc.md §9 gap 6). Wired
        # through AgentConfig.max_time_s to transformers' `max_time` stopping
        # criterion, so a runaway decode ends cleanly at the budget instead of
        # hanging the whole chain.
        raw_timeout = orch_cfg.get("timeout_per_agent_s")
        agent_timeout = float(raw_timeout) if raw_timeout else None

        t_conf = AgentConfig(agent_id="agent_trans", model_id=t_model, role="translation", device=t_device, hidden_dim=t_hidden, load_in_8bit=t_8bit, max_new_tokens=t_toks, dtype=t_dt, latent_transfer_layer=transfer_layer, max_time_s=agent_timeout)
        r_conf = AgentConfig(agent_id="agent_reason", model_id=r_model, role="reasoning", device=r_device, hidden_dim=r_hidden, load_in_8bit=r_8bit, max_new_tokens=r_toks, dtype=r_dt, latent_transfer_layer=transfer_layer, max_time_s=agent_timeout)
        s_conf = AgentConfig(agent_id="agent_safety", model_id=s_model, role="safety", device=s_device, hidden_dim=s_hidden, load_in_8bit=s_8bit, max_new_tokens=s_toks, dtype=s_dt, latent_transfer_layer=transfer_layer, max_time_s=agent_timeout)

        router.register_agent(TranslationAgent(t_conf))
        router.register_agent(ReasoningAgent(r_conf))
        router.register_agent(SafetyAgent(s_conf))

        self.checkpoint_manager.save((router, universal_space), "stage_a")
        return router, universal_space

    # Canonical role→adjacency-index mapping for 3-agent topologies. Single
    # source of truth lives in orchestration.router.TOPOLOGY_ROLE_INDEX so
    # Stage-B training targets and route(topology=...) can never drift apart.
    _TOPOLOGY_ROLE_INDEX = TOPOLOGY_ROLE_INDEX

    def _topology_target(self, query: str, target_language: str) -> torch.Tensor:
        """Derive a real per-query topology target from the TaskDecomposer DAG.

        The previous training loop used a constant all-ones (fully-connected) target
        for every query, which trains the CVAE prior to ignore its conditioning
        entirely and always emit the same graph — making Module D (query-conditioned
        topology generation) vacuous. Here the target adjacency reflects the actual
        dependency structure the decomposer assigns to the query (edge i→j when the
        sub-task with role j depends on the sub-task with role i).
        """
        if not hasattr(self, "_topology_decomposer"):
            self._topology_decomposer = TaskDecomposer()
        sub_tasks = self._topology_decomposer.decompose(query, target_language)
        n = len(self._TOPOLOGY_ROLE_INDEX)
        G = torch.zeros(n, n)
        role_of = {
            st.sub_task_id: self._TOPOLOGY_ROLE_INDEX.get(st.required_roles[0])
            for st in sub_tasks if st.required_roles
        }
        for st in sub_tasks:
            j = role_of.get(st.sub_task_id)
            if j is None:
                continue
            for dep in st.dependencies:
                i = role_of.get(dep)
                if i is not None:
                    G[i, j] = 1.0
        return G

    def _encode_cvae_query(self, text: str) -> torch.Tensor:
        """Deterministic hash-based tokenization for the CVAE query encoder.

        Shared by Stage B training and route-time topology sampling
        (router.topology_query_encoder) so the prior is queried with the same
        token distribution it was trained on.
        """
        import hashlib
        vocab_size = getattr(self, "_cvae_query_vocab_size", 30_522)
        tokens = []
        for word in text.lower().split()[:32]:
            h = (int(hashlib.md5(word.encode()).hexdigest(), 16) % (vocab_size - 1)) + 1
            tokens.append(h)
        while len(tokens) < 32:
            tokens.append(0)
        return torch.tensor(tokens, dtype=torch.long)

    def _run_stage_b(self) -> CVAETopologyPrior:
        """Stage B: Train the CVAE topology prior on real multi-agent task data."""
        if self.resume and self.checkpoint_manager.exists("stage_b"):
            logger.info("Resuming Stage B from checkpoints.")
            return self.checkpoint_manager.load_latest("stage_b")

        logger.info("Running Stage B: Training CVAE topology prior on adjacency matrices.")
        cvae_cfg = self._raw_config.get("cvae", {})
        train_cfg = cvae_cfg.get("training", {}) or {}

        # Module D geometry conditioning (strategy.md §4.2): x = [q ‖ Geo_L].
        # Opt-in via cvae.condition_on_geometry + a precomputed artifact from
        # scripts/export_geo_profiles.py; the GeoProfile loader enforces the
        # compressed 3–8-dim bound and the zero-fallback policy.
        geo_profile = None
        if cvae_cfg.get("condition_on_geometry"):
            from latent_coordination.topology.geo_profile import GeoProfile
            geo_path = cvae_cfg.get("geo_profile_path")
            if not geo_path:
                raise ValueError(
                    "cvae.condition_on_geometry=true requires cvae.geo_profile_path "
                    "pointing at a precomputed Geo_L artifact."
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

        # Load real query data from FLORES-200 to drive CVAE training
        logger.info("Loading real FLORES-200 queries for CVAE training.")
        try:
            from datasets import load_dataset  # type: ignore
            en_ds = load_dataset("openlanguagedata/flores_plus", name="eng_Latn", split="devtest")
            real_queries = [en_ds[i]["text"] for i in range(len(en_ds))]
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load FLORES-200 for CVAE training: {exc}. "
                "Install 'datasets' with: pip install datasets"
            ) from exc

        # Encode queries via a lightweight tokenizer-based bag-of-words embedding.
        # The SAME encoding must be used at route time (router.topology_query_encoder)
        # — hence the shared method rather than a local closure.
        self._cvae_query_vocab_size = t_config.query_vocab_size
        query_tensors = torch.stack([self._encode_cvae_query(q) for q in real_queries])

        # Per-query topology targets from the real task decomposition DAG (see
        # _topology_target — a constant all-ones target made the prior degenerate).
        # Cycle target languages across the configured benchmark set so both the
        # translation-inclusive and math-flavoured decomposition variants appear.
        langs = self.config.target_languages or ["th"]
        query_langs = [langs[i % len(langs)] for i in range(len(real_queries))]
        target_graphs = torch.stack([
            self._topology_target(q, lang)
            for q, lang in zip(real_queries, query_langs)
        ])
        # Per-query Geo_L vectors matching the cycled target languages.
        geo_all = geo_profile.batch(query_langs) if geo_profile is not None else None

        # Honor configs/*.yaml cvae.training (these were hardcoded to 20/1e-3/8,
        # silently ignoring the config's n_epochs/lr/batch_size).
        n_epochs = int(train_cfg.get("n_epochs", 20))
        lr = float(train_cfg.get("lr", 1e-3))
        batch_size = int(train_cfg.get("batch_size", 8))
        optimizer = torch.optim.Adam(cvae_prior.parameters(), lr=lr)
        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad()
            # Sample a batch of (query, decomposer-derived topology) pairs
            idx = torch.randperm(len(query_tensors))[:batch_size]
            batch_queries = query_tensors[idx].to(self.config.device)
            batch_G = target_graphs[idx].to(self.config.device)
            batch_geo = geo_all[idx].to(self.config.device) if geo_all is not None else None

            recon_G, mu, logvar = cvae_prior(batch_G, batch_queries, geo=batch_geo)
            loss, _ = cvae_prior.compute_loss(recon_G, batch_G, mu, logvar)
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0:
                logger.info("CVAE Epoch %d/%d | ELBO loss=%.4f", epoch, n_epochs, loss.item())

        # Keep the CVAE token-ids (long) separate from the float embeddings Stage D
        # builds for centroid clustering: a prior bug stored these long token-ids in
        # the shared `_real_query_tensors` slot, so Stage D ran k-means on hashed
        # token IDs instead of query embeddings whenever Stage B ran first.
        self._cvae_query_tokens = query_tensors
        self._real_queries = real_queries

        self.checkpoint_manager.save(cvae_prior, "stage_b")
        return cvae_prior

    def _run_stage_c(
        self, universal_space: UniversalLatentHub, router=None
    ) -> Dict:
        """Stage C: Latent adapters matching dimensions training.

        Returns (and checkpoints) ``{"hub": ..., "recursive_core": ...,
        "drift_probe": ...}`` — the hub plus the Module C/E components that live
        on the router. The previous version checkpointed the literal ``True``
        and relied on in-memory mutation of ``universal_space`` — so any later
        run that loaded Stage A from checkpoint but skipped Stage C
        (``--stages E``) received a hub with NO registered adapters at all,
        because the registrations lived only in the process that originally ran
        Stage C. The same held for Module C/E state, which is why they ride in
        this stage's checkpoint too.
        """
        if self.resume and self.checkpoint_manager.exists("stage_c"):
            logger.info("Resuming Stage C from checkpoints.")
            restored = self.checkpoint_manager.load_latest("stage_c")
            state = self._coerce_stage_c_state(restored)
            if state is not None:
                if router is not None:
                    self._apply_stage_c_state(router, state)
                return state
            # Legacy checkpoint (bool sentinel): fall through and recompute.
            logger.info("Legacy Stage C checkpoint found (no hub state); recomputing.")

        logger.info("Running Stage C: Adapters optimization mapping dimensions.")
        # Per-agent model_id (see _run_stage_a's fix note) -- registering every agent
        # with a single uniform hidden_dim silently discarded this stage's trained
        # adapters the moment Stage E re-registered each agent with its actual (possibly
        # different) hidden_dim from a real forward pass, for any agent whose model
        # doesn't happen to share the orchestrator's hidden_size.
        agent_model_ids = {
            agent.get("role"): agent.get("model_id", self._agent_model_id)
            for agent in self._raw_config.get("agents", [])
        }
        role_by_agent_id = {"agent_trans": "translation", "agent_reason": "reasoning", "agent_safety": "safety"}
        for aid, role in role_by_agent_id.items():
            model_id = agent_model_ids.get(role, self._agent_model_id)
            hidden_dim = self._resolve_agent_hidden_dim(model_id)
            universal_space.register_agent(aid, hidden_dim=hidden_dim)

        # Module A+B: actually TRAIN the adapters (L_recon + γ·L_DAE + μ·L_CKA with
        # unbiased-HSIC CKA — see UniversalLatentHub.fit_adapters). Before this,
        # "Adapter Pre-training" only *registered* random-init adapters and silently
        # ignored the configs' adapter_training block entirely, so every
        # latent_based_mas_ours number was produced by transferring hidden states
        # through untrained random MLPs.
        at_cfg = (
            self._raw_config.get("universal_latent_space", {}).get("adapter_training", {})
            or {}
        )
        states_by_agent: Dict[str, torch.Tensor] = {}
        adapter_texts: List[str] = []
        if at_cfg.get("enabled"):
            if router is None:
                raise RuntimeError(
                    "adapter_training.enabled=true requires the router (agent models) "
                    "to extract real hidden states; Stage C was called without one."
                )
            states_by_agent, adapter_texts = self._train_stage_c_adapters(
                universal_space, router, at_cfg
            )
        else:
            logger.warning(
                "Adapter training is DISABLED (universal_latent_space.adapter_training"
                ".enabled is false/absent): latent transfers will run through "
                "randomly-initialised adapters. That is fine for smoke tests but "
                "scientifically meaningless for benchmark results — enable it (and "
                "budget the model forward passes) for any run you intend to report."
            )

        # Module C: recursive latent refinement core operating in hub space
        # (dev_doc.md §9 gap 3 — the class existed but was never in the
        # execution path). Zero-init residual → identity until trained.
        recursive_core = None
        lr_cfg = self._raw_config.get("latent_reasoning", {}) or {}
        if lr_cfg.get("enabled"):
            from latent_coordination.latent_space.recursive_core import RecursiveLatentCore
            recursive_core = RecursiveLatentCore(
                hub_dim=self.config.universal_space_dim,
                max_steps=int(lr_cfg.get("max_steps", 10)),
                tau_exit=float(lr_cfg.get("tau_exit", 0.8)),
            )
            logger.info(
                "Module C enabled: RecursiveLatentCore(hub_dim=%d, max_steps=%s, tau_exit=%s).",
                self.config.universal_space_dim, lr_cfg.get("max_steps", 10),
                lr_cfg.get("tau_exit", 0.8),
            )

        # Module E: drift probe, fitted on REAL (hub-encoded state, query
        # embedding) pairs from the adapter-training corpus. The probe itself
        # refuses to gate on an untrained decoder, and we refuse to fabricate
        # training pairs — hence the hard adapter_training requirement.
        drift_probe = None
        ver_cfg = self._raw_config.get("verification", {}) or {}
        if ver_cfg.get("enabled"):
            if not states_by_agent:
                raise RuntimeError(
                    "verification.enabled=true requires real hidden states to fit the "
                    "drift-probe decoder; enable universal_latent_space.adapter_training "
                    "as well (the probe trains on the same collected corpus)."
                )
            drift_probe = self._fit_drift_probe(
                universal_space, states_by_agent, adapter_texts, ver_cfg
            )

        state = {
            "hub": universal_space,
            "recursive_core": recursive_core,
            "drift_probe": drift_probe,
        }
        if router is not None:
            self._apply_stage_c_state(router, state)
        self.checkpoint_manager.save(state, "stage_c")
        return state

    @staticmethod
    def _coerce_stage_c_state(restored) -> Optional[Dict]:
        """Normalize a Stage C checkpoint to the dict form.

        Accepts the current ``{"hub", "recursive_core", "drift_probe"}`` dict,
        a legacy bare ``UniversalLatentHub``, or anything else (→ None,
        signalling the caller to recompute).
        """
        if isinstance(restored, dict) and "hub" in restored:
            return restored
        if isinstance(restored, UniversalLatentHub):
            return {"hub": restored, "recursive_core": None, "drift_probe": None}
        return None

    @staticmethod
    def _apply_stage_c_state(router, state: Dict) -> None:
        """Attach Module C/E components from a Stage C state dict to the router."""
        router.recursive_core = state.get("recursive_core")
        router.drift_probe = state.get("drift_probe")

    def _fit_drift_probe(
        self,
        universal_space: UniversalLatentHub,
        states_by_agent: Dict[str, torch.Tensor],
        texts: List[str],
        ver_cfg: Dict,
    ):
        """Fit the Module E reconstruction probe on real hub-encoded states.

        For every (agent, prompt) pair from the adapter-training corpus, the
        probe learns to reconstruct the prompt's bag-of-words embedding from
        the agent's hub-encoded hidden state. Drift at test time = failure to
        reconstruct the query from the hub state.
        """
        from latent_coordination.eval.verification_probe import QueryReconstructionProbe
        from latent_coordination.orchestration.router import QUERY_EMBED_DIM, encode_query_bow

        # verification.probe_arch: 'linear' (default) | 'mlp' — the strategy's
        # §4.4 linear-vs-shallow-MLP comparison (ablation row 7e).
        probe = QueryReconstructionProbe(
            hub_dim=self.config.universal_space_dim,
            query_dim=QUERY_EMBED_DIM,
            tau_drift=float(ver_cfg.get("tau_drift", 0.5)),
            arch=str(ver_cfg.get("probe_arch", "linear")),
            mlp_hidden_dim=int(ver_cfg.get("mlp_hidden_dim", 256)),
        )
        hub_rows, q_rows = [], []
        q_embeds = [encode_query_bow(t) for t in texts]
        for aid, states in states_by_agent.items():
            with torch.no_grad():
                z = universal_space.encode(aid, states)  # (N, U)
            hub_rows.append(z.cpu())
            q_rows.extend(q_embeds)
        hub_states = torch.cat(hub_rows, dim=0)
        query_embeddings = torch.stack(q_rows)
        loss = probe.fit_decoder(
            hub_states, query_embeddings,
            n_epochs=int(ver_cfg.get("n_epochs", 100)),
            lr=float(ver_cfg.get("lr", 1e-3)),
        )
        logger.info(
            "Module E drift probe fitted on %d (hub-state, query) pairs | "
            "final cosine-reconstruction loss=%.4f | tau_drift=%s",
            hub_states.shape[0], loss, ver_cfg.get("tau_drift", 0.5),
        )
        return probe

    def _train_stage_c_adapters(
        self, universal_space: UniversalLatentHub, router, at_cfg: Dict
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """Collect real per-agent hidden states on FLORES+ English prompts and fit
        the hub adapters with the Module A+B objective.

        Returns the collected ``(states_by_agent, prompt_texts)`` so Stage C can
        reuse the same real corpus to fit the Module E drift probe."""
        n_samples = int(at_cfg.get("n_samples", 64))
        try:
            from datasets import load_dataset  # type: ignore
            en_ds = load_dataset("openlanguagedata/flores_plus", name="eng_Latn", split="devtest")
            texts = [en_ds[i]["text"] for i in range(min(n_samples, len(en_ds)))]
        except Exception as exc:
            raise RuntimeError(
                f"Adapter training needs real FLORES+ prompts, but loading failed: {exc}"
            ) from exc

        states_by_agent: Dict[str, torch.Tensor] = {}
        for aid, agent in router.agents.items():
            if not universal_space.is_registered(aid):
                continue
            rows = []
            for text in texts:
                hs = agent.extract_hidden_states(text, layer_ids=[-1])[-1]  # (1, L, D)
                rows.append(hs.float().mean(dim=1).squeeze(0).cpu())        # mean-pool → (D,)
            states_by_agent[aid] = torch.stack(rows)
        logger.info(
            "Collected row-aligned hidden states for %d agents on %d prompts; "
            "training adapters (n_epochs=%s, lr=%s, batch_size=%s).",
            len(states_by_agent), len(texts),
            at_cfg.get("n_epochs", 50), at_cfg.get("lr", 1e-3), at_cfg.get("batch_size", 32),
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
        """Stage D: K-means intent centroid clustering on real FLORES-200 query embeddings.

        Returns (and checkpoints) ``{"centroids": ..., "centroid_roles": ...}`` so a
        later run that skips Stage D can restore the fitted centroids onto the
        router — the previous ``True`` sentinel checkpoint lost them entirely.
        """
        if self.resume and self.checkpoint_manager.exists("stage_d"):
            logger.info("Resuming Stage D from checkpoints.")
            restored = self.checkpoint_manager.load_latest("stage_d")
            if isinstance(restored, dict) and "centroids" in restored:
                self._apply_stage_d_state(router, restored)
                return restored
            logger.info("Legacy Stage D checkpoint found (no centroid state); recomputing.")

        logger.info("Running Stage D: Intent centroids mapping on real FLORES-200 query embeddings.")
        from latent_coordination.orchestration.router import encode_query_bow

        # Always embed queries with the shared route-time encoder (encode_query_bow)
        # so centroids live in the SAME space route() queries against. The previous
        # version reused Stage B's tensors when available — but those are the CVAE's
        # long token-ids, not float embeddings, so k-means silently clustered hashed
        # token IDs whenever Stage B had run in the same process.
        if getattr(self, "_real_queries", None):
            real_queries = self._real_queries
        else:
            logger.info("Loading real FLORES-200 queries for centroid fitting.")
            try:
                from datasets import load_dataset  # type: ignore
                en_ds = load_dataset("openlanguagedata/flores_plus", name="eng_Latn", split="devtest")
                real_queries = [en_ds[i]["text"] for i in range(len(en_ds))]
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load FLORES+ for centroid fitting: {exc}"
                ) from exc
            self._real_queries = real_queries

        historical_embeddings = torch.stack([encode_query_bow(q) for q in real_queries])
        self._intent_query_embeddings = historical_embeddings

        # Honor orchestration.n_intent_centroids (was hardcoded to 3, ignoring the
        # config's 8).
        n_clusters = int(
            self._raw_config.get("orchestration", {}).get("n_intent_centroids", 8)
        )
        router.fit_centroids(historical_embeddings, n_clusters=n_clusters)
        state = {"centroids": router.centroids, "centroid_roles": router.centroid_roles}
        self.checkpoint_manager.save(state, "stage_d")
        return state

    @staticmethod
    def _apply_stage_d_state(router, state: Dict) -> None:
        """Restore fitted intent centroids onto a router from a Stage D checkpoint."""
        router.centroids = state["centroids"]
        router.centroid_roles = state.get("centroid_roles", {})

    def _run_stage_e(
        self,
        router: AdaptiveOrchestrator,
        universal_space: UniversalLatentHub,
        cvae_prior: Optional[CVAETopologyPrior] = None,
    ) -> Dict:
        """Stage E: Query execution evaluations and ablations."""
        for _key in ("stage_e", "stage_f"):  # "stage_f" = legacy key for this stage
            if self.resume and self.checkpoint_manager.exists(_key):
                logger.info("Resuming Stage E from checkpoint '%s'.", _key)
                return self.checkpoint_manager.load_latest(_key)

        logger.info("Running Stage E: Running orchestration task queries and ablations.")

        # Module D → execution: when routing_strategy is cvae_topology, the
        # trained Stage-B prior actually drives agent selection/order at route
        # time (dev_doc.md §9 gap 2 — sampled topologies used to be ignored).
        if router.router_type == "cvae":
            if cvae_prior is None:
                raise RuntimeError(
                    "orchestration.routing_strategy='cvae_topology' requires the "
                    "trained Stage-B CVAE prior, but none is available (run/resume "
                    "stage B first)."
                )
            router.topology_prior = cvae_prior
            router.topology_query_encoder = self._encode_cvae_query
            if getattr(cvae_prior.config, "geo_dim", 0) > 0:
                geo_profile = getattr(self, "_geo_profile", None)
                if geo_profile is None:
                    geo_path = self._raw_config.get("cvae", {}).get("geo_profile_path")
                    if not geo_path:
                        raise ValueError(
                            "The restored CVAE prior conditions on Geo_L but "
                            "cvae.geo_profile_path is not set in the config."
                        )
                    from latent_coordination.topology.geo_profile import GeoProfile
                    geo_profile = GeoProfile(geo_path)
                router.geo_profile = geo_profile

        decomposer = TaskDecomposer()

        # Load real FLORES+ tasks for decomposer demo (first task of the benchmark set)
        cap = self._flores_cap()
        logger.info(
            "FLORES+ per-language cap: %s", cap if cap is not None else "all (full devtest)"
        )
        benchmark_runner = MultiAgentBenchmarkRunner(
            output_dir=self.run_dir,
            max_samples_per_language=cap,
            languages=self.config.target_languages or None,
            translation_metrics=self._raw_config.get("benchmarks", {})
            .get("flores_plus", {})
            .get("translation_metrics"),
            # Without this the runner never saw the YAML's benchmarks section, so
            # mgsm/belebele/sea_vision/sea_safeguardbench blocks were silently
            # ignored in the pipeline path (only the standalone runner honored
            # them) and every Stage-E run was FLORES+-only.
            benchmarks=self._raw_config.get("benchmarks"),
        )
        real_tasks = benchmark_runner._load_real_tasks()
        if not real_tasks:
            raise RuntimeError(
                "Stage E loaded zero tasks. Ensure 'datasets' is installed, at "
                "least one benchmarks.* block is enabled, and the enabled "
                "benchmarks' datasets are accessible."
            )
        demo_query = real_tasks[0].query
        sub_tasks = decomposer.decompose(demo_query, real_tasks[0].target_language or "th")
        dep_graph = decomposer.build_dependency_graph(sub_tasks)
        decomposer.topological_sort(dep_graph)

        import hashlib
        comm_cfg = self._raw_config.get("communication", {})
        modes = comm_cfg.get("eval_modes")      # None → all benchmark modes
        backend_name = comm_cfg.get("backend", "auto")

        def _slug(s: str) -> str:
            return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s))

        # Cache key must reflect EVERY agent's model, not just the first (the
        # orchestrator's): with only the first entry, swapping the translation or
        # reasoning model in a heterogeneous config silently reused the previous
        # models' cached results. dict.fromkeys dedups while preserving order, so
        # homogeneous configs keep their historical single-model cache keys.
        agent_models = [
            a.get("model_id", self._agent_model_id)
            for a in self._raw_config.get("agents", [])
        ] or [self._agent_model_id]
        model_slug = "+".join(dict.fromkeys(_slug(m) for m in agent_models))
        # Scope: a cached per-mode result is valid only for the same languages +
        # FLORES cap + benchmark selection. The benchmarks fingerprint matters:
        # without it, an MGSM run and a Belebele run sharing a checkpoint dir
        # would silently reuse each other's cached mode results.
        bench_fingerprint = json.dumps(
            self._raw_config.get("benchmarks", {}) or {}, sort_keys=True, default=str
        )
        scope = "|".join([
            ",".join(sorted(self.config.target_languages or [])),
            f"cap={cap if cap is not None else 'all'}",
            f"bench={bench_fingerprint}",
        ])
        scope_hash = hashlib.md5(scope.encode()).hexdigest()[:8]
        report = benchmark_runner.run_eval(
            router, real_tasks, universal_space,
            modes=modes,
            backend_name=backend_name,
            checkpoint_manager=self.checkpoint_manager,
            cache_prefix=f"coord::{model_slug}::{scope_hash}",
        )

        report_dict = report.to_dict()
        self.checkpoint_manager.save(report_dict, "stage_e")
        return report_dict

    def _run_stage_f(
        self,
        router: AdaptiveOrchestrator,
        universal_space: UniversalLatentHub,
        benchmark_report: Dict,
    ) -> None:
        """Stage F: Visualizing topology layouts, scaling properties, and convergence curves."""
        logger.info("Running Stage F: Visualizing multi-agent layouts and convergence metrics.")

        viz_dir = self.run_dir / "plots"
        viz_dir.mkdir(parents=True, exist_ok=True)

        top_plotter = TopologyPlotter()
        eff_plotter = EfficiencyPlotter()
        latent_plotter = LatentSpacePlotter()

        # 1. Agent collaboration graph — adjacency from registered agents
        n_agents = len(router.agents)
        adj = torch.zeros(n_agents, n_agents)
        agent_names = [agent.config.role.title() for agent in router.agents.values()]
        # Safety supervises all; Reasoning -> Translation is typical flow
        if n_agents == 3:
            adj[1, 2] = 1.0   # Reasoning -> Translation
            adj[0, 1] = 1.0   # Safety -> Reasoning (oversight)
            adj[0, 2] = 1.0   # Safety -> Translation (oversight)
        try:
            top_plotter.plot_agent_topology(adj, agent_names, viz_dir / "collaboration_topology.png")
        except Exception as exc:  # noqa: BLE001 — viz is non-critical
            logger.warning("Skipping agent-topology plot: %s", exc)

        # 2. CVAE prior latent space — sample real latent codes from the trained prior.
        # The CVAE QueryEncoder consumes long token-ids (Stage B encoding); a downstream
        # viz error must not discard the completed benchmark, so this plot is best-effort.
        cvae_tokens = getattr(self, "_cvae_query_tokens", None)
        if cvae_tokens is not None:
            try:
                cvae_prior = self.checkpoint_manager.load_latest("stage_b")
                if cvae_prior is not None:
                    probe_queries = cvae_tokens[:20].long().to(self.config.device)
                    # encode(G, Q): adjacency batch (B, N, N) first, query tokens second.
                    # The previous call passed (Q, adj.view(-1)) — swapped order AND a
                    # flattened graph — so the query encoder embedding-looked-up float
                    # adjacencies and this plot crashed (silently, as best-effort) on
                    # every run.
                    probe_adj = adj.unsqueeze(0).repeat(probe_queries.size(0), 1, 1).to(self.config.device)
                    with torch.no_grad():
                        mu, logvar = cvae_prior.encode(probe_adj, probe_queries)
                    query_labels = getattr(self, "_real_queries", [])[:20]
                    top_plotter.plot_cvae_latent_space(
                        mu.cpu().numpy(), logvar.cpu().numpy(), query_labels,
                        viz_dir / "cvae_latent_space.png",
                    )
            except Exception as exc:  # noqa: BLE001 — viz is non-critical
                logger.warning("Skipping CVAE latent-space plot: %s", exc)

        # 3. Latency + accuracy tradeoff — from real benchmark_report
        results_by_mode = benchmark_report.get("results_by_mode", {}) if benchmark_report else {}
        if results_by_mode:
            try:
                ablation_data = {
                    "metrics_by_mode": {
                        mode: {"avg_latency_ms": metrics.get("latency_ms", 0.0)}
                        for mode, metrics in results_by_mode.items()
                    }
                }
                eff_plotter.plot_token_vs_latent_cost(ablation_data, viz_dir / "token_vs_latent_latency.png")
            except Exception as exc:  # noqa: BLE001 — viz is non-critical
                logger.warning("Skipping token-vs-latent plot: %s", exc)

            # 4. Accuracy-vs-latency tradeoff scatter — real measured values per mode
            try:
                tradeoff_points = [
                    {
                        "name": mode.replace("_", " ").title(),
                        "accuracy": metrics.get("accuracy", 0.0),
                        "latency_ms": metrics.get("latency_ms", 0.0),
                    }
                    for mode, metrics in results_by_mode.items()
                ]
                eff_plotter.plot_accuracy_vs_latency_tradeoff(tradeoff_points, viz_dir / "accuracy_vs_latency.png")
            except Exception as exc:  # noqa: BLE001 — viz is non-critical
                logger.warning("Skipping accuracy-vs-latency plot: %s", exc)

        # 5. Scalability — theoretical O(N) vs O(N²) communication cost
        try:
            n_agents_list = [2, 4, 8, 16, 32]
            costs = {
                "token_peer_to_peer": [c ** 2 for c in n_agents_list],
                "latent_hub_and_spoke": [c for c in n_agents_list],
            }
            eff_plotter.plot_scalability(n_agents_list, costs, viz_dir / "scalability_scaling.png")
        except Exception as exc:  # noqa: BLE001 — viz is non-critical
            logger.warning("Skipping scalability plot: %s", exc)

        # 6. Intent Centroid voronoi-style plot — from real fitted centroids and the
        # embeddings the centroids were fit on. Best-effort: skip on any viz error.
        intent_embeds = getattr(self, "_intent_query_embeddings", None)
        if router.centroids is not None and intent_embeds is not None:
            try:
                latent_plotter.plot_intent_centroids(
                    router.centroids,
                    intent_embeds[:20],
                    [],
                    viz_dir / "intent_centroids.png"
                )
            except Exception as exc:  # noqa: BLE001 — viz is non-critical
                logger.warning("Skipping intent-centroid plot: %s", exc)

        logger.info("All Multi-Agent Latent Coordination plots saved to %s", viz_dir)

    def _run_stage_g(self, benchmark_report: Dict) -> Dict:
        """Stage G: Final Latent Coordination report consolidation."""
        logger.info("Running Stage G: Compiling Latent Coordination final coordination report.")
        agent_model_ids = [
            a.get("model_id", self._agent_model_id)
            for a in self._raw_config.get("agents", [])
        ] or [self._agent_model_id]
        final_report = {
            "timestamp": self.timestamp,
            "config": self.config.to_dict(),
            "results": benchmark_report,
            "headline_framing": derive_headline_framing(
                (benchmark_report or {}).get("results_by_mode", {}) or {},
                agent_model_ids,
            ),
            "plots_directory": str(self.run_dir / "plots"),
            "status": "completed",
        }

        from shared.serialization import to_json_safe
        report_path = self.run_dir / "final_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(to_json_safe(final_report), f, indent=2, ensure_ascii=False)

        logger.info("Latent Coordination final report compiled at %s", report_path)
        return final_report
