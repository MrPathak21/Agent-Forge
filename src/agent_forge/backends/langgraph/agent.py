from __future__ import annotations

# TODO: implement LangGraph agent wrapper
#
# Suggested approach:
#   - Compile a StateGraph with a single node that calls an LLM
#   - Store the compiled graph as self._graph
#   - In run(), invoke the graph with {"messages": [HumanMessage(content=task)]}
#   - Return the last AIMessage content
#
# Reference: https://langchain-ai.github.io/langgraph/

from typing import Any

from agent_forge.core.agent import AgentStatus, BaseAgent


class LangGraphAgent(BaseAgent):
    """
    BaseAgent wrapper around a compiled LangGraph StateGraph.

    STUB — not yet implemented.
    """

    def __init__(self, agent_id: str, name: str, role: str) -> None:
        super().__init__(agent_id=agent_id, name=name, role=role)
        # self._graph = <compiled StateGraph goes here>

    async def run(self, task: str, **kwargs: Any) -> str:
        raise NotImplementedError(
            "LangGraphAgent.run() is not implemented yet. "
            "Wire up a compiled StateGraph and invoke it here."
        )

    async def close(self) -> None:
        # LangGraph graphs are stateless; nothing to clean up by default.
        self.status = AgentStatus.CLOSED
