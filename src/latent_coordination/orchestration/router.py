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
from latent_coordination.latent_space.universal_space import UniversalLatentSpace

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
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.roles = roles
        self.temperature = temperature
        n_roles = len(roles)
        self.keys = nn.Parameter(torch.randn(n_roles, query_dim) * 0.02)
        self.query_proj = nn.Linear(query_dim, query_dim, bias=False)

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
        q = self.query_proj(query_emb.float())  # (1, D)
        k = F.normalize(self.keys, dim=-1)       # (R, D)
        scores = (q @ k.T) / (self.temperature * (q.shape[-1] ** 0.5))  # (1, R)
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
        logger.info("AdaptiveOrchestrator initialized on device: %s, router_type=%s", device, router_type)

    def __setstate__(self, state: dict) -> None:
        """Backward-compatible unpickling for checkpoints missing newer attributes."""
        state.setdefault("router_type", "kmeans")
        state.setdefault("_attention_router", None)
        self.__dict__.update(state)

    def register_agent(self, agent: BaseAgent) -> None:
        """Add an agent to the router's registry."""
        self.agents[agent.config.agent_id] = agent
        self._attention_router = None  # reset; rebuilt on next route() call
        logger.info("Registered agent: %s (%s)", agent.config.agent_id, agent.config.role)

    def _get_attention_router(self, query_dim: int) -> AttentionRouter:
        """Lazily build/rebuild the AttentionRouter based on registered agents."""
        if self._attention_router is None:
            roles = list({a.config.role for a in self.agents.values()})
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
            logger.warning("Centroids not fitted. Defaulting assignment to centroid 0.")
            return 0

        emb = task_embedding.float().cpu().view(1, -1)
        dists = torch.cdist(emb, self.centroids)
        return int(torch.argmin(dists, dim=1).item())

    def route(self, task: AgentTask, topology: Optional[Tensor] = None) -> RoutingPlan:
        """Select specialized agents based on task query intent and graph topology.

        Encodes the task query into a bag-of-words embedding then routes via
        the configured router (attention or k-means).
        """
        import hashlib
        query_dim = 32
        vec = torch.zeros(query_dim)
        for word in task.query.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16) % query_dim
            vec[h] += 1.0
        norm = vec.norm()
        query_embedding = (vec / norm.clamp(min=1e-9)).unsqueeze(0)  # (1, query_dim)

        routing_confidence = 1.0

        if self.router_type == "attention":
            attn_router = self._get_attention_router(query_dim)
            required_roles, routing_confidence = attn_router.dispatch(
                query_embedding.to(self.device)
            )
        else:
            # K-means path (ablation)
            centroid_id = self.assign_centroid(query_embedding)
            required_roles = self.centroid_roles.get(
                centroid_id, ["reasoning", "translation", "safety"]
            )

        # Map roles to specific registered agent IDs
        selected_agents = []
        for role in required_roles:
            for aid, agent in self.agents.items():
                if agent.config.role == role:
                    selected_agents.append(aid)
                    break

        if not selected_agents:
            selected_agents = list(self.agents.keys())[:1]

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

    def execute(
        self,
        task: AgentTask,
        routing_plan: RoutingPlan,
        universal_space: UniversalLatentSpace,
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

            # Transfer state to the receiving agent
            if current_state is not None:
                # Map sender/receiver agent IDs for latent adapter transfer
                sender_id = routing_plan.execution_order[idx-1] if idx > 0 else "source"
                universal_space.register_agent(sender_id, current_state.shape[-1])
                # Use actual hidden state dim (shape[-1]) as ground truth for receiver
                # to avoid VLM config mismatches (text_config.hidden_size vs top-level)
                universal_space.register_agent(agent_id, current_state.shape[-1])

                # Transfer state via universal hub
                current_state = universal_space.transfer(sender_id, agent_id, current_state)
                total_latent_bytes += current_state.numel() * 4  # Float32 size in bytes

            sub_task = AgentTask(
                task_id=f"{task.task_id}_step_{idx}",
                query=task.query,
                context=last_output,
                latent_state=current_state,
                target_language=task.target_language,
            )

            # Run agent processing
            resp = agent.process(sub_task)
            agent_responses.append(resp)

            # Collect results
            last_output = resp.output_text
            current_state = resp.latent_state
            total_tokens += len(resp.output_text.split()) * 2.0  # estimated token cost (words * avg tokens/word)

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
        universal_space: UniversalLatentSpace,
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
            actual_dim = current_state.shape[-1] if current_state is not None else agent.config.hidden_dim
            universal_space.register_agent(sender_id, actual_dim)
            universal_space.register_agent(receiver_id, actual_dim)
            if current_state is not None:
                current_state = universal_space.transfer(sender_id, receiver_id, current_state)
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
