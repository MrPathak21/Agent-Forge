from __future__ import annotations

# TODO: implement Anthropic factory
#
# Suggested approach:
#   - In __init__(), create an anthropic.AsyncAnthropic(api_key=...) client
#   - In create(), build an AnthropicAgent and inject the shared client
#
# Reference: https://docs.anthropic.com/en/api/messages

import uuid
from typing import Any

from agent_forge.backends.anthropic.agent import AnthropicAgent
from agent_forge.config.settings import Settings
from agent_forge.core.factory import AgentFactory


class AnthropicFactory(AgentFactory):
    """
    Creates agents backed by the Anthropic Messages API.

    STUB — not yet implemented.
    """

    def __init__(self, *, model: str | None = None) -> None:
        super().__init__()
        self._config = Settings.for_provider("anthropic", model=model)
        # TODO: self._client = anthropic.AsyncAnthropic(api_key=self._config.api_key)

    async def create(
        self,
        role: str = "general_assistant",
        name: str = "agent",
        *,
        system_message: str | None = None,
        **kwargs: Any,
    ) -> AnthropicAgent:
        raise NotImplementedError(
            "AnthropicFactory.create() is not implemented yet. "
            "Instantiate AnthropicAgent with a shared AsyncAnthropic client."
        )
