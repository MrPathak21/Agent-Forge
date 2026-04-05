from __future__ import annotations

# TODO: implement LangGraph factory
#
# Suggested approach:
#   - Accept a ChatOpenAI / ChatAnthropic model in __init__
#   - In create(), build and compile a StateGraph, wrap it in LangGraphAgent
#
# Reference: https://langchain-ai.github.io/langgraph/

import uuid
from typing import Any

from agent_forge.backends.langgraph.agent import LangGraphAgent
from agent_forge.core.factory import AgentFactory


class LangGraphFactory(AgentFactory):
    """
    Creates agents backed by LangGraph StateGraphs.

    STUB — not yet implemented.
    """

    def __init__(self, provider: str = "openai", *, model: str | None = None) -> None:
        super().__init__()
        self._provider = provider
        self._model = model
        # TODO: initialise a LangChain model client here

    async def create(
        self,
        role: str = "general_assistant",
        name: str = "agent",
        **kwargs: Any,
    ) -> LangGraphAgent:
        raise NotImplementedError(
            "LangGraphFactory.create() is not implemented yet. "
            "Build and compile a StateGraph, then wrap it in LangGraphAgent."
        )
