from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from agent_forge.config.settings import ProviderConfig, Settings


class AgentSpec(BaseModel):
    """Specification for a single agent, written by the Orchestrator."""

    name: str
    role_description: str
    system_prompt: str
    tools: list[str] = []


class Orchestrator:
    """
    Uses an LLM to plan what agents are needed for a given goal and
    writes detailed system prompts for each one.

    Usage:
        orchestrator = Orchestrator(provider="openai")
        specs = await orchestrator.plan("Analyse AAPL earnings and write a short report.")

        for spec in specs:
            agent = await manager.spawn(
                name=spec.name,
                role=spec.role_description,
                system_message=spec.system_prompt,
            )
    """

    _SYSTEM = (
        "You are an agent orchestration planner. Given a high-level goal, decide what "
        "specialized AI agents are needed to accomplish it and write detailed system "
        "prompts for each one.\n\n"
        "Return a JSON object with an 'agents' array. Each element must have:\n"
        "  - name: short snake_case identifier (e.g. 'data_analyst')\n"
        "  - role_description: one sentence describing this agent's purpose\n"
        "  - system_prompt: a detailed, specific system prompt tailored exactly to the "
        "task — not generic. Include persona, constraints, output format expectations, "
        "and any domain knowledge the agent needs.\n"
        "  - tools: list of tool names this agent needs (empty list if none)\n\n"
        "Only create agents that are genuinely necessary. Prefer fewer, more capable "
        "agents over many narrow ones."
    )

    def __init__(self, provider: str = "openai", *, model: str | None = None) -> None:
        self._config: ProviderConfig = Settings.for_provider(provider, model=model)
        kwargs: dict[str, Any] = {"api_key": self._config.api_key}
        if self._config.base_url:
            kwargs["base_url"] = self._config.base_url
        self._client = AsyncOpenAI(**kwargs)

    async def plan(self, goal: str) -> list[AgentSpec]:
        """
        Given a goal, return a list of AgentSpecs needed to accomplish it.
        The LLM decides how many agents are needed and writes their system prompts.
        """
        response = await self._client.chat.completions.create(
            model=self._config.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": f"Goal: {goal}"},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        return [AgentSpec(**spec) for spec in data["agents"]]
