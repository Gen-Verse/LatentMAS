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
from latent_coordination.latent_space.universal_space import UniversalLatentSpace
from latent_coordination.latent_space.adapter import AdapterConfig, LatentAdapter
from latent_coordination.topology.cvae_prior import CVAETopologyPrior, TrainingConfig
from latent_coordination.topology.graph_utils import GraphUtils
from latent_coordination.orchestration.router import AdaptiveOrchestrator
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

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(self.config.output_dir) / "checkpoints",
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

    def run(self, stages: Optional[List[str]] = None) -> Dict:
        """Executes the pipeline stage-by-stage with resume support."""
        logger.info("Executing Latent Coordination Multi-Agent Pipeline. Configuration: %s", self.config)

        # Stage A: System Setup
        router, universal_space = self._run_stage_a()

        # Stage B: CVAE Topology Training
        cvae_prior = self._run_stage_b()

        # Stage C: Adapter Pre-training
        self._run_stage_c(universal_space)

        # Stage D: Intent Centroid Mapping
        self._run_stage_d(router)

        # Stage E: Multi-Agent Execution & Ablation
        benchmark_report = self._run_stage_e(router, universal_space)

        # Stage F: Visualizations
        self._run_stage_f(router, universal_space, benchmark_report)

        # Stage G: Report compilation
        final_report = self._run_stage_g(benchmark_report)

        return final_report

    def _run_stage_a(self) -> Tuple[AdaptiveOrchestrator, UniversalLatentSpace]:
        """Stage A: Setup orchestrators, databases, and adapters registry."""
        if self.resume and self.checkpoint_manager.exists("stage_a"):
            logger.info("Resuming Stage A from checkpoints.")
            return self.checkpoint_manager.load_latest("stage_a")

        logger.info("Running Stage A: Launching agent registry and universal space mappings.")
        router = AdaptiveOrchestrator(device=self.config.device)
        universal_space = UniversalLatentSpace(universal_dim=self.config.universal_space_dim)

        # Register specialized agents — hidden_dim must match the model's hidden_size.
        # Qwen1.5-0.5B-Chat: hidden_size=1024; Qwen3.5-9B: hidden_size=4096 (text backbone).
        model_id = self._agent_model_id
        hidden_dim = self._resolve_agent_hidden_dim(model_id)
        agent_devices = {
            agent.get("role"): agent.get("device", self.config.device)
            for agent in self._raw_config.get("agents", [])
        }
        agent_8bit = {
            agent.get("role"): agent.get("load_in_8bit", False)
            for agent in self._raw_config.get("agents", [])
        }
        
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

        t_conf = AgentConfig(agent_id="agent_trans", model_id=model_id, role="translation", device=t_device, hidden_dim=hidden_dim, load_in_8bit=t_8bit, max_new_tokens=t_toks, dtype=t_dt)
        r_conf = AgentConfig(agent_id="agent_reason", model_id=model_id, role="reasoning", device=r_device, hidden_dim=hidden_dim, load_in_8bit=r_8bit, max_new_tokens=r_toks, dtype=r_dt)
        s_conf = AgentConfig(agent_id="agent_safety", model_id=model_id, role="safety", device=s_device, hidden_dim=hidden_dim, load_in_8bit=s_8bit, max_new_tokens=s_toks, dtype=s_dt)

        router.register_agent(TranslationAgent(t_conf))
        router.register_agent(ReasoningAgent(r_conf))
        router.register_agent(SafetyAgent(s_conf))

        self.checkpoint_manager.save((router, universal_space), "stage_a")
        return router, universal_space

    def _run_stage_b(self) -> CVAETopologyPrior:
        """Stage B: Train the CVAE topology prior on real multi-agent task data."""
        if self.resume and self.checkpoint_manager.exists("stage_b"):
            logger.info("Resuming Stage B from checkpoints.")
            return self.checkpoint_manager.load_latest("stage_b")

        logger.info("Running Stage B: Training CVAE topology prior on adjacency matrices.")
        t_config = TrainingConfig(
            z_dim=self.config.cvae_latent_dim,
            query_dim=64,
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

        # Encode queries via a lightweight tokenizer-based bag-of-words embedding
        import hashlib
        query_dim = t_config.query_dim

        def _encode_query(text: str) -> torch.Tensor:
            """Deterministic hash-based tokenization."""
            vocab_size = t_config.query_vocab_size
            tokens = []
            for word in text.lower().split()[:32]:
                h = (int(hashlib.md5(word.encode()).hexdigest(), 16) % (vocab_size - 1)) + 1
                tokens.append(h)
            while len(tokens) < 32:
                tokens.append(0)
            return torch.tensor(tokens, dtype=torch.long)

        query_tensors = torch.stack([_encode_query(q) for q in real_queries])

        # Generate adjacency matrices from the CVAE prior
        optimizer = torch.optim.Adam(cvae_prior.parameters(), lr=1e-3)
        n_epochs = 20
        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad()
            # Sample a batch of queries
            batch_queries = query_tensors[torch.randperm(len(query_tensors))[:8]]
            batch_queries = batch_queries.to(self.config.device)
            # Form a default fully-connected target topology G for the real queries
            B = batch_queries.size(0)
            N = t_config.max_n_agents
            batch_G = torch.ones(B, N, N, device=self.config.device)
            
            recon_G, mu, logvar = cvae_prior(batch_G, batch_queries)
            loss, _ = cvae_prior.compute_loss(recon_G, batch_G, mu, logvar)
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0:
                logger.info("CVAE Epoch %d/%d | ELBO loss=%.4f", epoch, n_epochs, loss.item())

        # Store real query tensors for use in Stage F visualization.
        # Keep the CVAE token-ids (long) separate from whatever Stage D builds for
        # centroid clustering, so the CVAE encoder always receives long token-ids.
        self._real_query_tensors = query_tensors
        self._cvae_query_tokens = query_tensors
        self._real_queries = real_queries

        self.checkpoint_manager.save(cvae_prior, "stage_b")
        return cvae_prior

    def _run_stage_c(self, universal_space: UniversalLatentSpace) -> None:
        """Stage C: Latent adapters matching dimensions training."""
        if self.resume and self.checkpoint_manager.exists("stage_c"):
            logger.info("Resuming Stage C from checkpoints.")
            return

        logger.info("Running Stage C: Adapters optimization mapping dimensions.")
        hidden_dim = self._resolve_agent_hidden_dim(self._agent_model_id)
        for aid in ["agent_trans", "agent_reason", "agent_safety"]:
            universal_space.register_agent(aid, hidden_dim=hidden_dim)

        self.checkpoint_manager.save(True, "stage_c")

    def _run_stage_d(self, router) -> None:
        """Stage D: K-means intent centroid clustering on real FLORES-200 query embeddings."""
        if self.resume and self.checkpoint_manager.exists("stage_d"):
            logger.info("Resuming Stage D from checkpoints.")
            return

        logger.info("Running Stage D: Intent centroids mapping on real FLORES-200 query embeddings.")

        # Load real query embeddings (reuse from Stage B if available)
        if hasattr(self, "_real_query_tensors") and self._real_query_tensors is not None:
            historical_embeddings = self._real_query_tensors
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

            import hashlib
            query_dim = 64  # match CVAE query_dim from TrainingConfig

            def _encode_query(text: str) -> torch.Tensor:
                vec = torch.zeros(query_dim)
                for word in text.lower().split():
                    h = int(hashlib.md5(word.encode()).hexdigest(), 16) % query_dim
                    vec[h] += 1.0
                norm = vec.norm()
                return vec / norm.clamp(min=1e-9)

            historical_embeddings = torch.stack([_encode_query(q) for q in real_queries])
            self._real_query_tensors = historical_embeddings

        router.fit_centroids(historical_embeddings, n_clusters=3)
        self.checkpoint_manager.save(True, "stage_d")

    def _run_stage_e(self, router: AdaptiveOrchestrator, universal_space: UniversalLatentSpace) -> Dict:
        """Stage E: Query execution evaluations and ablations."""
        if self.resume and self.checkpoint_manager.exists("stage_f"):
            logger.info("Resuming Stage E from checkpoints.")
            return self.checkpoint_manager.load_latest("stage_f")

        logger.info("Running Stage E: Running orchestration task queries and ablations.")
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
        )
        real_tasks = benchmark_runner._load_real_tasks()
        if not real_tasks:
            raise RuntimeError(
                "Stage E requires real FLORES+ tasks. Ensure 'datasets' is installed "
                "and openlanguagedata/flores_plus is accessible."
            )
        demo_query = real_tasks[0].query
        sub_tasks = decomposer.decompose(demo_query, real_tasks[0].target_language or "th")
        dep_graph = decomposer.build_dependency_graph(sub_tasks)
        decomposer.topological_sort(dep_graph)

        import hashlib
        comm_cfg = self._raw_config.get("communication", {})
        modes = comm_cfg.get("eval_modes")      # None → all benchmark modes
        backend_name = comm_cfg.get("backend", "auto")
        model_slug = "".join(
            c if (c.isalnum() or c in "-_.") else "_" for c in str(self._agent_model_id)
        )
        # Scope: a cached per-mode result is valid only for the same languages + FLORES cap.
        scope = "|".join([
            ",".join(sorted(self.config.target_languages or [])),
            f"cap={cap if cap is not None else 'all'}",
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
        self.checkpoint_manager.save(report_dict, "stage_f")
        return report_dict

    def _run_stage_f(
        self,
        router: AdaptiveOrchestrator,
        universal_space: UniversalLatentSpace,
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
                    probe_adj = adj.view(-1).unsqueeze(0).repeat(probe_queries.size(0), 1).to(self.config.device)
                    with torch.no_grad():
                        mu, logvar = cvae_prior.encode(probe_queries, probe_adj)
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
        if router.centroids is not None and hasattr(self, "_real_query_tensors") and self._real_query_tensors is not None:
            try:
                latent_plotter.plot_intent_centroids(
                    router.centroids,
                    self._real_query_tensors[:20],
                    [],
                    viz_dir / "intent_centroids.png"
                )
            except Exception as exc:  # noqa: BLE001 — viz is non-critical
                logger.warning("Skipping intent-centroid plot: %s", exc)

        logger.info("All Multi-Agent Latent Coordination plots saved to %s", viz_dir)

    def _run_stage_g(self, benchmark_report: Dict) -> Dict:
        """Stage G: Final Latent Coordination report consolidation."""
        logger.info("Running Stage G: Compiling Latent Coordination final coordination report.")
        final_report = {
            "timestamp": self.timestamp,
            "config": self.config.to_dict(),
            "results": benchmark_report,
            "plots_directory": str(self.run_dir / "plots"),
            "status": "completed",
        }

        from shared.serialization import to_json_safe
        report_path = self.run_dir / "final_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(to_json_safe(final_report), f, indent=2, ensure_ascii=False)

        logger.info("Latent Coordination final report compiled at %s", report_path)
        return final_report
