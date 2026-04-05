from __future__ import annotations

from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from agent_forge.core.agent import AgentStatus, BaseAgent


class AutoGenAgent(BaseAgent):
    """
    BaseAgent wrapper around autogen_agentchat.AssistantAgent.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        role: str,
        native: AssistantAgent,
    ) -> None:
        super().__init__(agent_id=agent_id, name=name, role=role)
        self._native = native

    async def run(self, task: str, **kwargs: Any) -> str:
        self.status = AgentStatus.RUNNING
        try:
            result = await self._native.run(
                task=TextMessage(content=task, source="user")
            )
            return str(result.messages[-1].content)
        finally:
            self.status = AgentStatus.IDLE

    async def close(self) -> None:
        await self._native.close()
        self.status = AgentStatus.CLOSED

    @property
    def native(self) -> AssistantAgent:
        """Expose the underlying AutoGen agent for advanced use."""
        return self._native
