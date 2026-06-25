"""Blackboard MAS Baseline: shared-memory text communication.

Implements a central blackboard (shared dict store) where all agents read from
and write to a single shared memory.  This achieves O(N) reads/writes rather
than O(N²) peer-to-peer links, making the efficiency comparison with hub-and-spoke
latent MAS fair.

The current paper's O(N²) claim compares against dense peer-to-peer text MAS.
This baseline shows that a well-designed text MAS is already O(N), and the
latent channel's advantage must be stated in terms of latency and bandwidth,
not just message count topology.

References:
    AutoGen GroupChat (O(N) central manager, single broadcast per step)
    Blackboard system (Hayes-Roth 1985) — classical shared-memory MARL pattern
"""

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BlackboardEntry:
    """A single entry in the shared blackboard store."""
    author_id: str
    content: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BlackboardMASBaseline:
    """Central shared-memory text MAS.

    All agents write their outputs to a shared blackboard and read from it.
    Each round: one broadcast write (O(1)) + N reads (O(N)) = O(N) total.

    Args:
        max_entries: Maximum blackboard entries to retain (circular buffer).
    """

    def __init__(self, max_entries: int = 128) -> None:
        self._board: List[BlackboardEntry] = []
        self.max_entries = max_entries
        self._read_counts: Dict[str, int] = {}
        self._write_counts: Dict[str, int] = {}
        logger.info("BlackboardMASBaseline initialized (max_entries=%d)", max_entries)

    def write(self, author_id: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Write an agent output to the shared blackboard.

        O(1) operation regardless of N.

        Args:
            author_id: Agent that produced the content.
            content: Text content to share.
            metadata: Optional metadata dict.
        """
        entry = BlackboardEntry(
            author_id=author_id,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self._board.append(entry)
        if len(self._board) > self.max_entries:
            self._board.pop(0)
        self._write_counts[author_id] = self._write_counts.get(author_id, 0) + 1

    def read_all(self, reader_id: str) -> List[BlackboardEntry]:
        """Return all current blackboard entries for a reader agent.

        O(N) operation where N = number of entries (proportional to n_agents).

        Args:
            reader_id: Agent ID requesting the board contents.

        Returns:
            List of all :class:`BlackboardEntry` items in write order.
        """
        self._read_counts[reader_id] = self._read_counts.get(reader_id, 0) + 1
        return list(self._board)

    def read_latest(self, reader_id: str, n: int = 1) -> List[BlackboardEntry]:
        """Return the most recent N entries.

        Args:
            reader_id: Requesting agent.
            n: Number of most recent entries to return.

        Returns:
            Up to n most recent entries.
        """
        self._read_counts[reader_id] = self._read_counts.get(reader_id, 0) + 1
        return self._board[-n:]

    def clear(self) -> None:
        """Clear all blackboard entries."""
        self._board.clear()

    def communication_stats(self, n_agents: int, n_rounds: int) -> Dict[str, Any]:
        """Report communication complexity statistics.

        Args:
            n_agents: Number of agents in the system.
            n_rounds: Number of coordination rounds completed.

        Returns:
            Dict comparing blackboard O(N) vs peer-to-peer O(N²) costs.
        """
        peer_to_peer_msgs = n_agents * (n_agents - 1) * n_rounds
        blackboard_ops = n_agents * n_rounds  # each agent writes once, reads once
        return {
            "n_agents": n_agents,
            "n_rounds": n_rounds,
            "peer_to_peer_msg_count": peer_to_peer_msgs,
            "blackboard_op_count": blackboard_ops,
            "reduction_factor": peer_to_peer_msgs / max(blackboard_ops, 1),
            "total_writes": sum(self._write_counts.values()),
            "total_reads": sum(self._read_counts.values()),
        }

    def run_round(
        self,
        agents: Dict[str, Callable[[str, str], str]],
        task_query: str,
    ) -> Dict[str, str]:
        """Execute one coordination round: each agent reads, thinks, writes.

        Args:
            agents: Dict mapping agent_id -> callable(query, context) -> response.
            task_query: Task prompt shared with all agents.

        Returns:
            Dict mapping agent_id -> response text.
        """
        responses: Dict[str, str] = {}
        for agent_id, agent_fn in agents.items():
            entries = self.read_all(agent_id)
            context = "\n".join(e.content for e in entries[-5:])  # last 5 entries
            response = agent_fn(task_query, context)
            self.write(agent_id, response)
            responses[agent_id] = response
        return responses
