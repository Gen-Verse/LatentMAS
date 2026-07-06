"""Task decomposer for dividing complex queries into sub-tasks with dependency resolution."""

import logging
from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

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
class SubTask:
    """A single granular sub-task in the decomposed execution graph."""
    sub_task_id: int
    description: str
    required_roles: List[str]
    dependencies: List[int]
    priority: int = 1


class TaskDecomposer:
    """Decomposes a user query into a Directed Acyclic Graph (DAG) of sub-tasks."""

    def __init__(self) -> None:
        logger.info("TaskDecomposer initialized.")

    def decompose(self, query: str, target_language: str) -> List[SubTask]:
        """Perform rule-based task decomposition based on query semantics."""
        sub_tasks: List[SubTask] = []

        # Step 1: Safety analysis
        sub_tasks.append(SubTask(
            sub_task_id=0,
            description="Evaluate query content against policy standards.",
            required_roles=["safety"],
            dependencies=[],
            priority=3
        ))

        # Step 2: Reasoning processing
        # Detect if math logic or factual reasoning is likely required
        query_lower = query.lower()
        if "calculate" in query_lower or "solve" in query_lower or "math" in query_lower or "reason" in query_lower:
            sub_tasks.append(SubTask(
                sub_task_id=1,
                description="Solve the reasoning/computation task step-by-step.",
                required_roles=["reasoning"],
                dependencies=[0],  # depends on safety validation
                priority=2
            ))
        else:
            sub_tasks.append(SubTask(
                sub_task_id=1,
                description="Process semantic representation of user request.",
                required_roles=["reasoning"],
                dependencies=[0],
                priority=1
            ))

        # Step 3: Translation / Localization
        # If target language is not english (en), translate output
        if target_language != "en":
            sub_tasks.append(SubTask(
                sub_task_id=2,
                description=f"Localize output and verify script constraints for target language '{target_language}'.",
                required_roles=["translation"],
                dependencies=[1],  # depends on reasoning solution
                priority=2
            ))

        logger.info("Decomposed query into %d sub-tasks.", len(sub_tasks))
        return sub_tasks

    def build_dependency_graph(self, sub_tasks: List[SubTask]) -> Tensor:
        """Create a DAG adjacency matrix representing sub-task dependencies.

        Returns
        -------
        Tensor
            Adjacency matrix of shape ``(n_tasks, n_tasks)``, where ``adj[i, j] = 1``
            indicates that task j depends on task i.
        """
        n = len(sub_tasks)
        adj = torch.zeros(n, n)
        # Create map from task ID to matrix index
        id_to_idx = {task.sub_task_id: idx for idx, task in enumerate(sub_tasks)}

        for idx, task in enumerate(sub_tasks):
            for dep in task.dependencies:
                if dep in id_to_idx:
                    dep_idx = id_to_idx[dep]
                    adj[dep_idx, idx] = 1.0  # dep_idx must complete before idx

        return adj

    def topological_sort(self, dep_graph: Tensor) -> List[List[int]]:
        """Sort tasks topologically into parallel execution batches.

        Tasks in the same batch have zero unresolved dependencies and can run concurrently.
        """
        n = dep_graph.shape[0]
        in_degree = dep_graph.sum(dim=0).tolist()
        adj_list = [[] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if dep_graph[i, j] > 0:
                    adj_list[i].append(j)

        batches: List[List[int]] = []
        visited = set()

        while len(visited) < n:
            current_batch = []
            for node in range(n):
                if node not in visited and in_degree[node] == 0:
                    current_batch.append(node)

            if not current_batch:
                # Detect cycle fallback (execute remaining sequentially)
                logger.warning("DAG cycle detected during topological sort! Executing fallback.")
                remaining = [node for node in range(n) if node not in visited]
                batches.append(remaining)
                break

            batches.append(current_batch)
            for node in current_batch:
                visited.add(node)
                # Decrement indegree for all neighbors
                for neighbor in adj_list[node]:
                    in_degree[neighbor] -= 1

        logger.info("Resolved topological ordering into %d parallel batches: %s", len(batches), batches)
        return batches
