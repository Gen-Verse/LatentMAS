"""Adaptive router and latent intent orchestrator for heterogeneous agents."""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from latent_coordination.agents.base_agent import BaseAgent, AgentResponse, AgentTask
from latent_coordination.topology.cvae_prior import CVAETopologyPrior
from latent_coordination.latent_space.universal_space import UniversalLatentHub

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


logger = logging.getLogger(__name__)

# Canonical execution precedence for agent roles (strategy.md §4.5: ordered role
# sequence translation → reasoning → safety). Iterating a *set* of role strings
# (the previous behaviour) is order-randomized per process by PYTHONHASHSEED, which
# made the multi-agent execution order — and therefore which response select_answer()
# scores — non-reproducible across runs.
CANONICAL_ROLE_ORDER = ("translation", "reasoning", "safety", "orchestrator")

# Query embedding dimensionality shared by route(), fit_centroids() callers, and the
# CVAE query encoder config (configs/*.yaml cvae.query_dim). route() used to hash into
# 32 dims while Stage D fit centroids on 64-dim embeddings, so the k-means routing
# path crashed (or silently compared garbage) on any real run.
QUERY_EMBED_DIM = 64

# Canonical role ↔ adjacency-index mapping for 3-agent topologies. Single source
# of truth shared by CVAE training targets (Stage B), sampled-topology routing
# (route(topology=...)), and any consumer interpreting an adjacency matrix.
TOPOLOGY_ROLE_ORDER = ("translation", "reasoning", "safety")
TOPOLOGY_ROLE_INDEX = {role: i for i, role in enumerate(TOPOLOGY_ROLE_ORDER)}

# Role → descriptor vocabulary used to seed each AttentionRouter key. No stage of
# the pipeline trains the attention router (Stage D only fits the k-means path),
# so its previous randomly-initialised keys produced ~0.01-magnitude logits and
# an exactly-uniform softmax: every one of the ~14k routing decisions in the
# 20260705 bench_suite runs logged confidence in [0.333, 0.341] and dispatched
# all three roles — the "adaptive" router was a constant. Seeding each key with
# the hashed-BoW embedding of its role's vocabulary (the SAME encode_query_bow
# space route() embeds queries into) gives the untrained router a real,
# deterministic routing signal. A query with no lexical overlap with any
# prototype (e.g. non-Latin script) still degrades to near-uniform weights and
# full-chain dispatch — the safe fallback.
ROLE_KEY_PROTOTYPES: Dict[str, str] = {
    "translation": (
        "translate translation language english meaning word phrase sentence "
        "multilingual foreign script text say written"
    ),
    "reasoning": (
        "question answer which what why how passage according following correct "
        "reason solve explain calculate conclusion best statement true"
    ),
    "safety": (
        "safe unsafe harmful violence hate threat weapon attack kill dangerous "
        "illegal explicit harassment moderation risk toxic"
    ),
    "orchestrator": (
        "plan coordinate orchestrate schedule assign delegate manage route"
    ),
}


def encode_query_bow(text: str, dim: int = QUERY_EMBED_DIM) -> Tensor:
    """Deterministic hashed bag-of-words query embedding (L2-normalised float)."""
    import hashlib

    vec = torch.zeros(dim)
    for word in text.lower().split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
        vec[h] += 1.0
    return vec / vec.norm().clamp(min=1e-9)


def canonical_role_sort(roles) -> List[str]:
    """Sort roles into the canonical execution order, unknown roles last (stable)."""
    order = {r: i for i, r in enumerate(CANONICAL_ROLE_ORDER)}
    return sorted(roles, key=lambda r: (order.get(r, len(order)), r))


@dataclass
class LatentIntentCentroid:
    """Represents a clustered query intent centroid in latent space."""
    centroid_id: int
    vector: Tensor
    associated_roles: List[str]


@dataclass
class RoutingPlan:
    """Contains selected agents, execution constraints, and routing costs."""
    task_id: str
    selected_agents: List[str]
    execution_order: List[str]
    estimated_cost: float
    routing_confidence: float = 1.0


@dataclass
class OrchestrationResult:
    """Consolidated summary of a multi-agent orchestrated run."""
    task_id: str
    final_output: str
    agent_responses: List[AgentResponse]
    total_elapsed_ms: float
    communication_cost_tokens: int
    communication_cost_latent: float


class AttentionRouter(nn.Module):
    """Learned attention-based intent router.

    Replaces k-means centroid assignment with scaled dot-product attention over
    a learned key matrix (one key per role).  This avoids the need to choose k
    a priori and handles non-convex, non-spherical intent clusters.

    Args:
        query_dim: Dimensionality of incoming query embeddings.
        roles: Ordered list of agent role names the router can dispatch to.
        temperature: Softmax temperature for the attention distribution.
    """

    def __init__(
        self,
        query_dim: int,
        roles: List[str],
        temperature: float = 0.15,
    ) -> None:
        super().__init__()
        self.roles = roles
        self.temperature = temperature
        # Keys are seeded from the role-descriptor prototypes so the router
        # discriminates without a training loop (none exists in the pipeline;
        # random keys made the softmax exactly uniform — see ROLE_KEY_PROTOTYPES).
        # Roles without a prototype fall back to hashing the role name itself,
        # which is still deterministic across processes.
        self.keys = nn.Parameter(torch.stack([
            encode_query_bow(ROLE_KEY_PROTOTYPES.get(role, role), dim=query_dim)
            for role in roles
        ]))
        # Identity init (not the default Kaiming-uniform random matrix): at
        # init the score must be exactly cos(query, prototype). A random
        # projection scrambled the query before it ever met the keys, which is
        # half of what flattened the old logits to ~0.01. Kept as a Linear so a
        # future training stage can still learn a better projection.
        self.query_proj = nn.Linear(query_dim, query_dim, bias=False)
        with torch.no_grad():
            self.query_proj.weight.copy_(torch.eye(query_dim))

    def forward(self, query_emb: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute soft role distribution and confidence.

        Args:
            query_emb: Query embedding, shape (1, query_dim) or (query_dim,).

        Returns:
            Tuple of:
                weights: Soft role weights, shape (n_roles,).
                confidence: Max weight value (routing sharpness indicator).
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        # Cosine similarity over temperature. The old scaled-dot-product form
        # additionally divided by sqrt(query_dim) — correct for unnormalised
        # attention logits, but here BOTH sides are unit vectors, so scores
        # already live in [-1, 1] and the extra /8 damping (query_dim=64)
        # guaranteed a uniform softmax no matter what the keys contained.
        q = F.normalize(self.query_proj(query_emb.float()), dim=-1)  # (1, D)
        k = F.normalize(self.keys, dim=-1)                           # (R, D)
        scores = (q @ k.T) / self.temperature  # (1, R)
        weights = F.softmax(scores, dim=-1).squeeze(0)  # (R,)
        confidence = float(weights.max().item())
        return weights, torch.tensor(confidence)

    def dispatch(
        self,
        query_emb: Tensor,
        threshold: float = 0.1,
    ) -> Tuple[List[str], float]:
        """Return roles with weight above threshold (hard dispatch from soft weights).

        Args:
            query_emb: Query embedding.
            threshold: Minimum weight to include a role (default selects top roles).

        Returns:
            Tuple of (selected_roles list, confidence float).
        """
        with torch.no_grad():
            weights, conf = self(query_emb)
        selected = [self.roles[i] for i, w in enumerate(weights) if w.item() >= threshold]
        if not selected:
            selected = [self.roles[int(weights.argmax().item())]]
        return selected, float(conf.item())


class AdaptiveOrchestrator:
    """Orchestrates multi-agent planning and routing using latent intent centroids."""

    def __init__(self, device: str = "cpu", router_type: str = "attention") -> None:
        self.device = torch.device(device)
        self.agents: Dict[str, BaseAgent] = {}
        self.router_type = router_type
        # K-means centroids (kept for ablation when router_type='kmeans')
        self.centroids: Optional[Tensor] = None
        self.centroid_roles: Dict[int, List[str]] = {}
        # Attention router (initialized lazily once agents are registered)
        self._attention_router: Optional[AttentionRouter] = None
        # Module C (recursive latent refinement) and Module E (drift probe):
        # attached by the pipeline (Stage C) when latent_reasoning.enabled /
        # verification.enabled. None = plain encode->decode transfer.
        self.recursive_core = None
        self.drift_probe = None
        # Module D: trained CVAE topology prior + its query tokenizer, attached
        # by Stage E when orchestration.routing_strategy == 'cvae_topology'.
        self.topology_prior = None
        self.topology_query_encoder = None
        self.geo_profile = None
        logger.info("AdaptiveOrchestrator initialized on device: %s, router_type=%s", device, router_type)

    def __setstate__(self, state: dict) -> None:
        """Backward-compatible unpickling for checkpoints missing newer attributes."""
        state.setdefault("router_type", "kmeans")
        # Always rebuild the attention router instead of restoring the pickled
        # one: nothing ever trains it, so dropping it loses no learned state,
        # while a stage_a checkpoint written before the prototype-key fix would
        # resurrect the old random-key module (uniform softmax) — or worse, run
        # random keys through the new sharper temperature and route arbitrarily.
        state["_attention_router"] = None
        state.setdefault("recursive_core", None)
        state.setdefault("drift_probe", None)
        state.setdefault("topology_prior", None)
        state.setdefault("topology_query_encoder", None)
        state.setdefault("geo_profile", None)
        self.__dict__.update(state)

    def register_agent(self, agent: BaseAgent) -> None:
        """Add an agent to the router's registry."""
        self.agents[agent.config.agent_id] = agent
        self._attention_router = None  # reset; rebuilt on next route() call
        logger.info("Registered agent: %s (%s)", agent.config.agent_id, agent.config.role)

    def _get_attention_router(self, query_dim: int) -> AttentionRouter:
        """Lazily build/rebuild the AttentionRouter based on registered agents."""
        if self._attention_router is None:
            # Deterministic canonical order — a raw set() here randomized the role
            # (and thus agent execution) order across processes.
            roles = canonical_role_sort({a.config.role for a in self.agents.values()})
            self._attention_router = AttentionRouter(
                query_dim=query_dim, roles=roles
            ).to(self.device)
            logger.debug("AttentionRouter built with roles: %s", roles)
        return self._attention_router

    def fit_centroids(self, task_embeddings: Tensor, n_clusters: int = 5) -> None:
        """Compute intent centroids using simple PyTorch-based k-means clustering."""
        X = task_embeddings.float().to(self.device)
        n_samples, embed_dim = X.shape

        if n_samples < n_clusters:
            # Degrade gracefully by replicating samples
            n_clusters = n_samples

        # Initialize centroids randomly from dataset
        indices = torch.randperm(n_samples)[:n_clusters]
        centroids = X[indices].clone()

        # Run 10 iterations of k-means
        for _ in range(10):
            # Compute pairwise distances
            # (n_samples, 1, embed_dim) - (1, n_centroids, embed_dim)
            dists = torch.cdist(X, centroids)  # (n_samples, n_centroids)
            assignments = torch.argmin(dists, dim=1)  # (n_samples,)

            # Update centroids
            for k in range(n_clusters):
                mask = assignments == k
                if mask.sum() > 0:
                    centroids[k] = X[mask].mean(dim=0)

        self.centroids = centroids.cpu()

        # Map each centroid to a default agent role sequence
        default_sequences = [
            ["reasoning", "translation", "safety"],
            ["reasoning", "safety"],
            ["translation", "safety"],
            ["reasoning", "translation"],
            ["safety"]
        ]
        for k in range(n_clusters):
            self.centroid_roles[k] = default_sequences[k % len(default_sequences)]

        logger.info("Fitted %d latent intent centroids.", n_clusters)

    def assign_centroid(self, task_embedding: Tensor) -> int:
        """Map a task embedding to its nearest intent centroid."""
        if self.centroids is None:
            logger.warning("Centroids not fitted. Defaulting assignment to random cluster to enforce differentiation.")
            return torch.randint(0, 5, (1,)).item()

        emb = task_embedding.float().cpu().view(1, -1)
        dists = torch.cdist(emb, self.centroids)
        return int(torch.argmin(dists, dim=1).item())

    def _sample_topology_for(self, task: AgentTask) -> Optional[Tensor]:
        """Sample a collaboration topology from the trained CVAE prior (Module D).

        Requires ``self.topology_prior`` (trained CVAETopologyPrior) and
        ``self.topology_query_encoder`` (str → token-id tensor, the SAME
        encoding Stage B trained with). When the prior conditions on geometry
        (geo_dim > 0), the task's target-language Geo_L is looked up via
        ``self.geo_profile`` — a missing profile raises rather than silently
        substituting zeros.
        """
        if self.topology_prior is None or self.topology_query_encoder is None:
            raise RuntimeError(
                "router_type='cvae' requires a trained topology prior and its query "
                "encoder attached (Stage E does this when orchestration."
                "routing_strategy is 'cvae_topology'); got none."
            )
        Q = self.topology_query_encoder(task.query)
        if Q.dim() == 1:
            Q = Q.unsqueeze(0)
        geo = None
        if getattr(self.topology_prior.config, "geo_dim", 0) > 0:
            if self.geo_profile is None:
                raise RuntimeError(
                    "The topology prior conditions on Geo_L (geo_dim>0) but no "
                    "GeoProfile is attached to the router."
                )
            geo = self.geo_profile.vector(task.target_language or "en").unsqueeze(0)
        adj = self.topology_prior.sample_topology(Q, n_samples=1, geo=geo)  # (1, N, N)
        return adj[0]

    def _plan_from_topology(self, task: AgentTask, topology: Tensor) -> Optional[RoutingPlan]:
        """Build a RoutingPlan from a sampled adjacency matrix (Module D output).

        Interpretation (canonical TOPOLOGY_ROLE_INDEX): ``adj[i, j] = 1`` means
        role i's output feeds role j — role j depends on role i. A role
        participates if it has any incident edge (or self-loop). Execution
        order is the topological order of the dependency DAG; on a cycle the
        active roles fall back to the canonical role order. Returns None when
        the topology selects no roles at all, letting the caller fall through
        to the standard router instead of executing an empty plan.
        """
        adj = topology.squeeze(0) if topology.dim() == 3 else topology
        n = min(adj.shape[0], len(TOPOLOGY_ROLE_ORDER))
        adj = (adj[:n, :n] > 0.5).float()

        active = [
            i for i in range(n)
            if adj[i].any() or adj[:, i].any()
        ]
        if not active:
            return None

        # Kahn's algorithm on the active sub-graph (self-loops ignored).
        sub = adj.clone()
        sub.fill_diagonal_(0.0)
        in_deg = {i: int(sub[:, i][active].sum().item()) for i in active}
        order: List[int] = []
        ready = sorted(i for i in active if in_deg[i] == 0)
        while ready:
            node = ready.pop(0)
            order.append(node)
            for j in active:
                if sub[node, j] > 0 and j not in order and j not in ready:
                    in_deg[j] -= 1
                    if in_deg[j] == 0:
                        ready.append(j)
            ready.sort()
        if len(order) < len(active):
            # Cycle: fall back to canonical role order over the active roles.
            logger.warning(
                "Sampled topology for task %s contains a cycle; falling back to "
                "canonical role order for the active roles.", task.task_id,
            )
            ordered_roles = canonical_role_sort(
                TOPOLOGY_ROLE_ORDER[i] for i in active
            )
        else:
            ordered_roles = [TOPOLOGY_ROLE_ORDER[i] for i in order]

        selected_agents = []
        for role in ordered_roles:
            for aid, agent in self.agents.items():
                if agent.config.role == role:
                    selected_agents.append(aid)
                    break
        if not selected_agents:
            return None

        logger.info(
            "Routed Task %s via sampled topology to sequence: %s",
            task.task_id, selected_agents,
        )
        return RoutingPlan(
            task_id=task.task_id,
            selected_agents=selected_agents,
            execution_order=list(selected_agents),
            estimated_cost=len(selected_agents) * 1.5,
            routing_confidence=1.0,
        )

    def route(self, task: AgentTask, topology: Optional[Tensor] = None) -> RoutingPlan:
        """Select specialized agents based on task query intent and graph topology.

        When ``topology`` is given (or ``router_type == 'cvae'``, which samples
        one from the trained Module D prior), the sampled adjacency actually
        determines agent selection AND execution order — previously the
        argument was accepted and ignored, so sampled topologies never
        influenced execution (dev_doc.md §9 gap 2). Otherwise encodes the task
        query into a bag-of-words embedding and routes via the configured
        router (attention or k-means). Uses the shared :func:`encode_query_bow`
        at :data:`QUERY_EMBED_DIM` so route-time embeddings live in the same
        space as the centroids Stage D fits (they used to differ, 32-dim here
        vs 64-dim in centroid fitting, breaking the k-means path).
        """
        if topology is None and self.router_type == "cvae":
            topology = self._sample_topology_for(task)
        if topology is not None:
            plan = self._plan_from_topology(task, topology)
            if plan is not None:
                return plan
            logger.warning(
                "Topology for task %s selected no roles; falling back to standard routing.",
                task.task_id,
            )

        query_embedding = encode_query_bow(task.query).unsqueeze(0)  # (1, QUERY_EMBED_DIM)

        routing_confidence = 1.0

        if self.router_type == "attention":
            attn_router = self._get_attention_router(QUERY_EMBED_DIM)
            required_roles, routing_confidence = attn_router.dispatch(
                query_embedding.to(self.device)
            )
        else:
            # K-means path (ablation)
            centroid_id = self.assign_centroid(query_embedding)
            required_roles = self.centroid_roles.get(
                centroid_id, ["reasoning", "translation", "safety"]
            )

        # Map roles to specific registered agent IDs, in canonical execution order
        # (translation → reasoning → safety) so the pipeline is reproducible and the
        # safety verdict comes after — not before — the substantive answer.
        selected_agents = []
        for role in canonical_role_sort(required_roles):
            for aid, agent in self.agents.items():
                if agent.config.role == role:
                    selected_agents.append(aid)
                    break

        if not selected_agents:
            logger.warning("Router selected no matching agents. Falling back to full heterogeneous MAS broadcast.")
            selected_agents = list(self.agents.keys())

        execution_order = list(selected_agents)
        estimated_cost = len(selected_agents) * 1.5

        logger.info(
            "Routed Task %s to sequence: %s | router=%s | confidence=%.3f",
            task.task_id,
            execution_order,
            self.router_type,
            routing_confidence,
        )

        return RoutingPlan(
            task_id=task.task_id,
            selected_agents=selected_agents,
            execution_order=execution_order,
            estimated_cost=estimated_cost,
            routing_confidence=routing_confidence,
        )

    def _hub_transfer(
        self,
        universal_space: UniversalLatentHub,
        sender_id: str,
        receiver_id: str,
        hidden_states: Tensor,
        query: str,
        step_meta: Dict,
    ) -> Tensor:
        """Encode → (Module C refine) → (Module E verify) → decode.

        This is the full latent channel between two agents. Modules C and E
        operate INSIDE the universal hub space — between the sender's encoder
        and the receiver's decoder — which is why this replaces the older
        opaque ``universal_space.transfer`` call in the execution path
        (dev_doc.md §9 gap 3: both modules existed but were dead code).

        Module E per strategy.md §4.4: on drift the repair hop is capped at
        one retry (fall back to the unrefined hub state); a still-drifting
        sample is flagged in ``step_meta`` and execution continues — a
        benchmark run must record the failure, not die on it.
        """
        universal = universal_space.encode(sender_id, hidden_states)
        refined = universal

        if self.recursive_core is not None:
            refined = self.recursive_core(universal.float())
            step_meta["n_recursive_steps"] = self.recursive_core.last_n_steps

        if self.drift_probe is not None:
            # probe.query_dim is the canonical attribute (the mlp probe_arch's
            # Sequential decoder has no out_features); fall back for probes
            # restored from checkpoints written before the MLP variant existed.
            query_dim = getattr(
                self.drift_probe, "query_dim", None
            ) or self.drift_probe.decoder.out_features
            q_emb = encode_query_bow(query, dim=query_dim).unsqueeze(0)

            def _drift_of(z: Tensor) -> float:
                z_pooled = z.mean(dim=1) if z.dim() == 3 else z
                score = self.drift_probe(
                    z_pooled, q_emb.to(z_pooled.device), raise_on_drift=False
                )
                return float(score.max().item())

            drift = _drift_of(refined)
            step_meta["drift_score"] = drift
            if drift > self.drift_probe.tau_drift:
                # Repair hop (one retry): drop the refinement, go back to the
                # raw hub encoding of the sender's states.
                refined = universal
                drift_after = _drift_of(refined)
                step_meta["drift_score_after_repair"] = drift_after
                step_meta["drift_repaired"] = drift_after <= self.drift_probe.tau_drift
                if not step_meta["drift_repaired"]:
                    logger.warning(
                        "Latent drift persists after repair hop (%.3f > tau=%.3f) "
                        "for transfer %s→%s; flagged in metadata.",
                        drift_after, self.drift_probe.tau_drift, sender_id, receiver_id,
                    )

        return universal_space.decode(receiver_id, refined)

    def execute(
        self,
        task: AgentTask,
        routing_plan: RoutingPlan,
        universal_space: UniversalLatentHub,
    ) -> OrchestrationResult:
        """Execute the routing plan sequentially using text-free latent state transfers."""
        start_time = time.time()
        agent_responses = []
        current_state = task.latent_state
        last_output = task.context
        total_tokens = 0
        total_latent_bytes = 0.0

        for idx, agent_id in enumerate(routing_plan.execution_order):
            agent = self.agents[agent_id]
            logger.info("Executing Agent: %s on task.", agent_id)
            step_meta: Dict = {}

            # Transfer state to the receiving agent
            if current_state is not None:
                # Map sender/receiver agent IDs for latent adapter transfer.
                # IMPORTANT: `current_state.shape[-1]` is the SENDER's outgoing
                # dimension at this point in the loop -- it must only be used to
                # register the sender. Registering the *receiver* with it too (a
                # prior bug) silently swapped in a wrong-shaped adapter for any
                # cross-architecture pair (confirmed live: registered agent_reason,
                # a 4096-dim Llama-3.1 agent, at hidden_dim=3584 -- the *sender*
                # agent_trans/Sailor2's dimension -- which then crashed
                # inject_latent_and_generate with a tensor-size mismatch). The
                # receiver must be registered with its own true hidden_dim from its
                # AgentConfig. Also guard both registrations with `is_registered`:
                # register_agent() unconditionally overwrites with a fresh,
                # untrained adapter on every call (see its docstring/warning), so
                # re-registering on every hand-off was discarding Stage C's trained
                # adapters and replacing them with random-init ones each time.
                sender_id = routing_plan.execution_order[idx-1] if idx > 0 else "source"
                if not universal_space.is_registered(sender_id):
                    universal_space.register_agent(sender_id, current_state.shape[-1])
                if not universal_space.is_registered(agent_id):
                    universal_space.register_agent(agent_id, agent.config.hidden_dim)

                # Transfer via the universal hub with Modules C (recursive
                # refinement) and E (drift verification) applied in hub space.
                current_state = self._hub_transfer(
                    universal_space, sender_id, agent_id, current_state,
                    task.query, step_meta,
                )
                total_latent_bytes += current_state.numel() * 4  # Float32 size in bytes

            # Text-free channel: inter-agent communication is the latent tensor ONLY.
            # Passing the previous agent's decoded text as `context` (the previous
            # behaviour) opened a hidden token side-channel inside the "latent" mode,
            # while the benchmark simultaneously reported token_cost=0 for it — the
            # headline 0-token claim was false. Each agent receives the original
            # task context plus the transferred latent state, nothing else.
            sub_task = AgentTask(
                task_id=f"{task.task_id}_step_{idx}",
                query=task.query,
                context=task.context,
                latent_state=current_state,
                target_language=task.target_language,
            )

            # Run agent processing
            resp = agent.process(sub_task)
            if step_meta:
                resp.metadata.update(step_meta)  # per-transfer drift / step counts
            agent_responses.append(resp)

            # Collect results
            last_output = resp.output_text
            current_state = resp.latent_state
            # No decoded text crosses the inter-agent boundary in this mode, so the
            # communication token count is genuinely 0 (was: a fabricated
            # words*2.0 "estimated token cost" of text that is no longer passed).

        elapsed_ms = (time.time() - start_time) * 1000.0

        return OrchestrationResult(
            task_id=task.task_id,
            final_output=last_output,
            agent_responses=agent_responses,
            total_elapsed_ms=elapsed_ms,
            communication_cost_tokens=int(total_tokens),
            communication_cost_latent=total_latent_bytes
        )

    def compare_communication_modes(
        self,
        task: AgentTask,
        agents: List[BaseAgent],
        universal_space: UniversalLatentHub,
    ) -> Dict[str, float]:
        """Measure real communication overhead: Token-based vs Latent-based modes.

        Runs the same task through both communication strategies and measures
        actual wall-clock latency and token counts.
        """
        import time

        # --- Token mode: pass decoded text between agents ---
        t_start = time.perf_counter()
        token_cost = 0
        context = task.context or ""
        for agent in agents:
            sub_task = AgentTask(
                task_id=f"{task.task_id}_token",
                query=task.query,
                context=context,
                latent_state=None,
                target_language=task.target_language,
            )
            resp = agent.process(sub_task)
            context = resp.output_text
            token_cost += len(resp.output_text.split())
        token_latency_ms = (time.perf_counter() - t_start) * 1000.0

        # --- Latent mode: pass hidden states between agents ---
        t_start = time.perf_counter()
        current_state = task.latent_state
        for idx, agent in enumerate(agents):
            sender_id = f"agent_{idx - 1}" if idx > 0 else "source"
            receiver_id = f"agent_{idx}"
            if current_state is not None:
                # Same rule as execute(): the sender is registered at the incoming
                # state's dimension, the RECEIVER at its own true hidden_dim (using
                # the sender's dim for both was the cross-architecture crash bug),
                # and neither registration may overwrite an existing/trained adapter.
                if not universal_space.is_registered(sender_id):
                    universal_space.register_agent(sender_id, current_state.shape[-1])
                if not universal_space.is_registered(receiver_id):
                    universal_space.register_agent(receiver_id, agent.config.hidden_dim)
                current_state = self._hub_transfer(
                    universal_space, sender_id, receiver_id, current_state,
                    task.query, {},
                )
            sub_task = AgentTask(
                task_id=f"{task.task_id}_latent_{idx}",
                query=task.query,
                context=None,
                latent_state=current_state,
                target_language=task.target_language,
            )
            resp = agent.process(sub_task)
            current_state = resp.latent_state
        latent_latency_ms = (time.perf_counter() - t_start) * 1000.0

        latent_bytes = (
            current_state.numel() * 4 if current_state is not None else 0.0
        )

        return {
            "token_mode_cost_tokens": float(token_cost),
            "token_mode_latency_ms": token_latency_ms,
            "latent_mode_cost_bytes": float(latent_bytes),
            "latent_mode_latency_ms": latent_latency_ms,
        }
