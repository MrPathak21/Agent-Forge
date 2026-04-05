from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_forge.core.agent import BaseAgent


class AgentFactory(ABC):
    """
    Abstract factory for creating and destroying agents.

    Each backend provides a concrete implementation. The manager
    depends only on this interface, staying backend-agnostic.
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    @abstractmethod
    async def create(self, role: str, name: str, *, system_message: str, **kwargs: Any) -> BaseAgent:
        """
        Instantiate and register a new agent.

        Implementations must add the agent to self._agents and return it.
        The system_message defines the agent's behaviour and is typically
        written by an Orchestrator for the specific task at hand.
        """

    # ------------------------------------------------------------------
    # Lifecycle (shared, override if needed)
    # ------------------------------------------------------------------

    async def kill(self, agent_id: str) -> None:
        """Close and deregister an agent by ID."""
        agent = self._agents.pop(agent_id, None)
        if agent is not None:
            await agent.close()

    async def kill_all(self) -> None:
        """Close and deregister every agent owned by this factory."""
        for agent in list(self._agents.values()):
            await agent.close()
        self._agents.clear()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, agent_id: str) -> BaseAgent:
        try:
            return self._agents[agent_id]
        except KeyError:
            raise KeyError(f"No agent with id={agent_id!r}")

    def list(self) -> list[BaseAgent]:
        return list(self._agents.values())

    def __len__(self) -> int:
        return len(self._agents)
