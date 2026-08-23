from __future__ import annotations

"""
App Registry for agent-forge.

Apps declare a fixed agent workflow in a JSON/YAML config file under /apps.
When a task's goal semantically matches a registered app (see
Orchestrator.match_app), the app's workflow is used directly instead of
dynamic planning:

- framework_locked apps skip planning entirely — build_plan_from_app()
  constructs the AgentSpec/GraphSpec deterministically from the config,
  no LLM call involved.
- non-locked apps still need one LLM call (Orchestrator.plan_from_app) to
  pick a framework and write prompts, but the agent names/roles are fixed
  by the config rather than invented.

Zero registered apps is a no-op: AppRegistry.list_apps() returns [] and the
pipeline falls straight through to today's dynamic orchestration.
"""

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from agent_forge.core.orchestrator import AgentSpec, GraphEdge, GraphNode, GraphSpec

logger = logging.getLogger(__name__)


class WorkflowStep(BaseModel):
    agent: str
    role: str
    tools: list[str] = []


class AppGuardrails(BaseModel):
    skip_goal_clarity: bool = False
    skip_plan_validation: bool = False
    hallucination_check: bool = True


class AppConfig(BaseModel):
    app_id: str
    name: str
    version: str = "1.0"
    triggers: list[str] = []
    description: str = ""
    workflow: list[WorkflowStep]
    framework: Literal["autogen", "langgraph"]
    framework_locked: bool = False
    models: dict[str, str] = {}
    models_locked: bool = False
    guardrails: AppGuardrails = AppGuardrails()


class AppRegistry:
    """Loads and looks up registered apps. Loading is additive/idempotent per app_id."""

    def __init__(self) -> None:
        self._apps: dict[str, AppConfig] = {}

    def load(self, apps_dir: Path) -> None:
        """Load every *.json/*.yaml/*.yml file in apps_dir. Malformed files are
        skipped with a warning rather than aborting startup."""
        if not apps_dir.exists():
            logger.info("App registry directory %s does not exist — no apps loaded.", apps_dir)
            return
        for path in sorted(apps_dir.iterdir()):
            if path.suffix.lower() not in (".json", ".yaml", ".yml"):
                continue
            try:
                data = self._parse_file(path)
                app = AppConfig(**data)
            except Exception:
                logger.warning("Skipping malformed app config %s", path, exc_info=True)
                continue
            self.register(app)
            logger.info("Registered app app_id=%s name=%s from %s", app.app_id, app.name, path.name)

    @staticmethod
    def _parse_file(path: Path) -> dict:
        text = path.read_text()
        if path.suffix.lower() == ".json":
            return json.loads(text)
        import yaml
        return yaml.safe_load(text)

    def register(self, app: AppConfig) -> None:
        """Register (or replace) an app programmatically."""
        self._apps[app.app_id] = app

    def list_apps(self) -> list[AppConfig]:
        return list(self._apps.values())

    def get(self, app_id: str) -> AppConfig | None:
        return self._apps.get(app_id)


def _resolve_model(node_name: str, app: AppConfig) -> str | None:
    """Lenient model lookup: exact key match on the node/agent name, else a
    substring match against app.models keys, else None (provider default)."""
    if node_name in app.models:
        return app.models[node_name]
    for key, model in app.models.items():
        if key in node_name or node_name in key:
            return model
    return None


_APP_PROMPT_TEMPLATE = (
    "You are {name}, part of the \"{app_name}\" workflow.\n\n"
    "Your role: {role}\n\n"
    "Workflow description: {description}\n\n"
    "Produce your output directly and completely in a single response — you are not "
    "an interactive chatbot and cannot ask follow-up questions or wait for more input. "
    "All context you need is already in the task message.{tools_note}"
)

_TOOLS_NOTE_TEMPLATE = (
    "\n\nYou have access to these tools: {tools}. Use them when they would "
    "materially improve your answer — do not call them speculatively."
)


def build_plan_from_app(app: AppConfig) -> tuple[GraphSpec | list[AgentSpec], str]:
    """Deterministically build an execution plan from a locked app's workflow —
    no LLM call. System prompts come from a fixed template, not orchestrator-written
    prose; this is what makes LOCKED execution genuinely "no planning"."""
    if app.framework == "langgraph":
        nodes = [
            GraphNode(
                name=step.agent,
                role_description=step.role,
                system_prompt=_APP_PROMPT_TEMPLATE.format(
                    name=step.agent, app_name=app.name, role=step.role, description=app.description,
                    tools_note=_TOOLS_NOTE_TEMPLATE.format(tools=", ".join(step.tools)) if step.tools else "",
                ),
                task_prompt=step.role,
                tools=step.tools,
                model=_resolve_model(step.agent, app),
            )
            for step in app.workflow
        ]
        edges = [
            GraphEdge(from_node=nodes[i].name, to_node=nodes[i + 1].name)
            for i in range(len(nodes) - 1)
        ]
        return GraphSpec(nodes=nodes, edges=edges, entry=nodes[0].name), "langgraph"

    agents = [
        AgentSpec(
            name=step.agent,
            role_description=step.role,
            system_prompt=_APP_PROMPT_TEMPLATE.format(
                name=step.agent, app_name=app.name, role=step.role, description=app.description,
                tools_note=_TOOLS_NOTE_TEMPLATE.format(tools=", ".join(step.tools)) if step.tools else "",
            ),
            tools=step.tools,
            model=_resolve_model(step.agent, app),
        )
        for step in app.workflow
    ]
    return agents, "autogen"
