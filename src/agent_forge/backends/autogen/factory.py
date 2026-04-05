from __future__ import annotations

import uuid
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agent_forge.backends.autogen.agent import AutoGenAgent
from agent_forge.config.settings import ProviderConfig, Settings
from agent_forge.core.factory import AgentFactory

# Role → system message
_ROLE_PROMPTS: dict[str, str] = {
    "general_assistant": (
        "You are a concise, helpful AI assistant."
    ),
    "research_analyst": (
        "You are a research analyst. Gather facts carefully, separate signal from noise, "
        "and structure findings clearly."
    ),
    "market_strategist": (
        "You are a market strategist. Form views with explicit reasoning, identify market "
        "drivers, and compare bullish and bearish cases before concluding."
    ),
    "risk_reviewer": (
        "You are a risk reviewer. Focus on downside, edge cases, invalid assumptions, "
        "and failure modes. Challenge weak reasoning."
    ),
    "execution_planner": (
        "You are an execution planner. Convert intent into concrete next steps, sequence "
        "actions carefully, and keep plans realistic and measurable."
    ),
}

_DEFAULT_ROLE = "general_assistant"


class AutoGenFactory(AgentFactory):
    """
    Creates AutoGen AssistantAgents backed by an OpenAI-compatible model client.

    Args:
        provider: LLM provider name (only 'openai' is wired today).
        model:    Override the default model for this provider.
    """

    def __init__(self, provider: str = "openai", *, model: str | None = None) -> None:
        super().__init__()
        self._config: ProviderConfig = Settings.for_provider(provider, model=model)

    def _build_model_client(self) -> OpenAIChatCompletionClient:
        if self._config.provider != "openai":
            raise NotImplementedError(
                f"AutoGenFactory only supports 'openai' today, got {self._config.provider!r}."
            )
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "api_key": self._config.api_key,
        }
        if self._config.base_url:
            kwargs["base_url"] = self._config.base_url
        return OpenAIChatCompletionClient(**kwargs)

    async def create(
        self,
        role: str = _DEFAULT_ROLE,
        name: str = "agent",
        *,
        system_message: str | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> AutoGenAgent:
        agent_id = str(uuid.uuid4())
        resolved_system = system_message or _ROLE_PROMPTS.get(role, _ROLE_PROMPTS[_DEFAULT_ROLE])

        native = AssistantAgent(
            name=name,
            model_client=self._build_model_client(),
            system_message=resolved_system,
            tools=tools or [],
        )

        agent = AutoGenAgent(agent_id=agent_id, name=name, role=role, native=native)
        self._agents[agent_id] = agent
        return agent
