from __future__ import annotations

"""
agent-forge demo entrypoint.

The Orchestrator decides what agents are needed for a given goal and writes
their system prompts. The manager then spawns, runs, and kills those agents.

Run:
    python -m agent_forge
"""

import asyncio
import logging

from agent_forge.backends.autogen import AutoGenFactory
from agent_forge.core.manager import AgentManager
from agent_forge.core.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


GOAL = (
    "Analyse the impact of rising US interest rates on emerging market equities "
    "and produce a concise investment brief."
)

# One task per agent — keyed by agent name as returned by the orchestrator.
TASKS: dict[str, str] = {
    # The orchestrator may return different agent names depending on the goal.
    # These tasks are matched to agents by name at runtime; unmatched agents
    # receive the goal itself as their task.
}


async def run() -> None:
    orchestrator = Orchestrator(provider="openai")
    factory = AutoGenFactory(provider="openai")
    manager = AgentManager(factory)

    print("\n=== agent-forge demo ===\n")
    print(f"Goal: {GOAL}\n")

    # --- Plan ---
    print("Orchestrator planning agents...\n")
    specs = await orchestrator.plan(GOAL)

    for spec in specs:
        print(f"  [{spec.name}] {spec.role_description}")
    print()

    # --- Spawn ---
    agents = []
    for spec in specs:
        agent = await manager.spawn(
            role=spec.role_description,
            name=spec.name,
            system_message=spec.system_prompt,
        )
        agents.append((spec, agent))

    print(f"Active agents: {[a.name for _, a in agents]}\n")

    # --- Run ---
    for spec, agent in agents:
        task = TASKS.get(spec.name, GOAL)
        result = await manager.run_task(agent.agent_id, task)
        print(f"[{spec.name}]\n{result}\n")

    # --- Kill ---
    await manager.shutdown()
    print("All agents shut down cleanly.")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
