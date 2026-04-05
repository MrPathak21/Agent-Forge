from __future__ import annotations

# TODO: implement Anthropic agent wrapper
#
# Suggested approach:
#   - Use the `anthropic` SDK: client.messages.create(...)
#   - Store the client + system_message on the instance
#   - In run(), send the task as a user message and return the text response
#   - For tool use, pass tools= to messages.create and handle tool_use blocks
#
# Reference: https://docs.anthropic.com/en/api/messages

from typing import Any

from agent_forge.core.agent import AgentStatus, BaseAgent


class AnthropicAgent(BaseAgent):
    """
    BaseAgent wrapper around the Anthropic Messages API.

    STUB — not yet implemented.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        role: str,
        system_message: str,
    ) -> None:
        super().__init__(agent_id=agent_id, name=name, role=role)
        self._system_message = system_message
        # self._client = <anthropic.AsyncAnthropic goes here>

    async def run(self, task: str, **kwargs: Any) -> str:
        raise NotImplementedError(
            "AnthropicAgent.run() is not implemented yet. "
            "Call client.messages.create() with the task as a user message."
        )

    async def close(self) -> None:
        # HTTP clients are stateless; nothing to clean up by default.
        self.status = AgentStatus.CLOSED
