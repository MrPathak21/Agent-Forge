from __future__ import annotations

import uuid
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agent_forge.backends.autogen.agent import AutoGenAgent
from agent_forge.config.settings import ProviderConfig, Settings
from agent_forge.core.factory import AgentFactory

# autogen_ext only recognizes a fixed list of OpenAI model name strings and
# refuses to guess capabilities for anything outside it (raises ValueError).
# Settings.py's configured defaults (e.g. "gpt-5.4-mini") are newer than that
# list, so we infer a reasonable ModelInfo and retry rather than failing.
_FAMILY_BY_PREFIX: list[tuple[str, str]] = [
    ("gpt-5", ModelFamily.GPT_5),
    ("gpt-4.1", ModelFamily.GPT_41),
    ("gpt-4.5", ModelFamily.GPT_45),
    ("gpt-4o", ModelFamily.GPT_4O),
    ("gpt-4", ModelFamily.GPT_4),
    ("gpt-3.5", ModelFamily.GPT_35),
    ("o4", ModelFamily.O4),
    ("o3", ModelFamily.O3),
    ("o1", ModelFamily.O1),
]


def _infer_model_info(model: str) -> dict[str, Any]:
    """Best-effort ModelInfo for an OpenAI model name autogen_ext doesn't recognize."""
    family = next((f for prefix, f in _FAMILY_BY_PREFIX if model.startswith(prefix)), ModelFamily.UNKNOWN)
    return {
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
        "family": family,
        "multiple_system_messages": True,
    }


class AutoGenFactory(AgentFactory):
    """
    Creates AutoGen AssistantAgents backed by an OpenAI-compatible model client.

    Agents are fully defined by the system_message passed at creation time —
    typically written by an Orchestrator for the specific task at hand.

    Args:
        provider: LLM provider name (only 'openai' is wired today).
        model:    Override the default model for this provider.
    """

    def __init__(self, provider: str = "openai", *, model: str | None = None) -> None:
        super().__init__()
        self._config: ProviderConfig = Settings.for_provider(provider, model=model)

    def _build_model_client(self, model_override: str | None = None) -> OpenAIChatCompletionClient:
        if self._config.provider != "openai":
            raise NotImplementedError(
                f"AutoGenFactory only supports 'openai' today, got {self._config.provider!r}."
            )
        kwargs: dict[str, Any] = {
            "model": model_override or self._config.model,
            "api_key": self._config.api_key,
        }
        if self._config.base_url:
            kwargs["base_url"] = self._config.base_url
        try:
            return OpenAIChatCompletionClient(**kwargs)
        except ValueError as exc:
            if "model_info is required" not in str(exc):
                raise
            kwargs["model_info"] = _infer_model_info(kwargs["model"])
            return OpenAIChatCompletionClient(**kwargs)

    async def create(
        self,
        role: str = "agent",
        name: str = "agent",
        *,
        system_message: str,
        tools: list[str] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AutoGenAgent:
        from agent_forge.tools import get_tools

        agent_id = str(uuid.uuid4())
        resolved_tools = get_tools(tools) if tools else []
        resolved_model = model or self._config.model

        native = AssistantAgent(
            name=name,
            model_client=self._build_model_client(model_override=resolved_model),
            system_message=system_message,
            tools=resolved_tools,
        )

        agent = AutoGenAgent(agent_id=agent_id, name=name, role=role, native=native, model=resolved_model)
        self._agents[agent_id] = agent
        return agent
