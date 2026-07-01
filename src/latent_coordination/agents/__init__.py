"""Agent implementations: base agent, specialized agents (translation, reasoning, safety)."""


from latent_coordination.agents.base_agent import BaseAgent, AgentConfig, AgentTask, AgentResponse
from latent_coordination.agents.specialized_agents import (
    TranslationAgent,
    ReasoningAgent,
    SafetyAgent,
    SafetyVerdict,
)
from latent_coordination.agents.single_agent_baseline import SingleAgentOneFlow

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

__all__ = [
    "BaseAgent",
    "AgentConfig",
    "AgentTask",
    "AgentResponse",
    "TranslationAgent",
    "ReasoningAgent",
    "SafetyAgent",
    "SafetyVerdict",
]
