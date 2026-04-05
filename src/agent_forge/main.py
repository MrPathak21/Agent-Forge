from __future__ import annotations

"""
agent-forge demo entrypoint.

Spins up two agents via the AutoGen backend, runs tasks on each,
then cleanly shuts everything down. Swap AutoGenFactory for
LangGraphFactory or AnthropicFactory once those backends are wired.

Run:
    python -m agent_forge
"""

import asyncio
import logging

from agent_forge.backends.autogen import AutoGenFactory
from agent_forge.core.manager import AgentManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


async def run() -> None:
    factory = AutoGenFactory(provider="openai")
    manager = AgentManager(factory)

    print("\n=== agent-forge demo ===\n")

    # Spawn two specialised agents
    analyst = await manager.spawn(role="research_analyst", name="analyst")
    strategist = await manager.spawn(role="market_strategist", name="strategist")

    print(f"Active agents: {[a.name for a in manager.list_agents()]}\n")

    # Run tasks
    answer1 = await manager.run_task(
        analyst.agent_id,
        "What are the key factors that drive inflation?",
    )
    print(f"[analyst] {answer1}\n")

    answer2 = await manager.run_task(
        strategist.agent_id,
        "Given rising inflation, what is the bullish and bearish case for equities?",
    )
    print(f"[strategist] {answer2}\n")

    # Release one agent mid-session
    await manager.release(analyst.agent_id)
    print(f"After releasing analyst — active: {[a.name for a in manager.list_agents()]}\n")

    # Shutdown everything
    await manager.shutdown()
    print("All agents shut down cleanly.")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
