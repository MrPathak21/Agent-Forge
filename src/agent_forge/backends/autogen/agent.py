from __future__ import annotations

from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from agent_forge.core.agent import AgentRunResult, AgentStatus, BaseAgent


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
        model: str = "",
    ) -> None:
        super().__init__(agent_id=agent_id, name=name, role=role)
        self._native = native
        self._model = model

    async def run(self, task: str, **kwargs: Any) -> AgentRunResult:
        self.status = AgentStatus.RUNNING
        try:
            result = await self._native.run(
                task=TextMessage(content=task, source="user")
            )
            input_tokens = 0
            output_tokens = 0
            for msg in result.messages:
                usage = getattr(msg, "models_usage", None)
                if usage is not None:
                    input_tokens += usage.prompt_tokens
                    output_tokens += usage.completion_tokens
            return AgentRunResult(
                content=str(result.messages[-1].content),
                model=self._model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        finally:
            self.status = AgentStatus.IDLE

    async def close(self) -> None:
        await self._native.close()
        self.status = AgentStatus.CLOSED

    @property
    def native(self) -> AssistantAgent:
        """Expose the underlying AutoGen agent for advanced use."""
        return self._native
