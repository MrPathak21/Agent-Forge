# agent-forge

Framework-agnostic AI agent orchestration scaffold.

A **manager + factory** pattern that lets you spin up, run, and tear down agents on demand — regardless of which AI framework powers them under the hood.

## Architecture

```
AgentManager          ← orchestrates lifecycle & task routing
  └── AgentFactory    ← abstract: create / kill agents
        ├── AutoGenFactory    ✅ working
        ├── LangGraphFactory  🔧 stub
        └── AnthropicFactory  🔧 stub

BaseAgent             ← common interface: run(task) / close()
  ├── AutoGenAgent
  ├── LangGraphAgent
  └── AnthropicAgent
```

The manager and core abstractions know nothing about the backend — swap factories freely.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
# add your OPENAI_API_KEY to .env
```

## Run the demo

```bash
python -m agent_forge
```

Spawns a `research_analyst` and a `market_strategist`, runs a task on each, releases the analyst mid-session, then shuts down cleanly.

## Adding a new agent role

Edit `src/agent_forge/backends/autogen/factory.py` and add an entry to `_ROLE_PROMPTS`:

```python
"your_role": "You are a ... your system prompt here."
```

Then spawn it:

```python
agent = await manager.spawn(role="your_role", name="my_agent")
```

## Implementing a stub backend

1. Open `src/agent_forge/backends/langgraph/` (or `anthropic/`)
2. Follow the `TODO` comments in `agent.py` and `factory.py`
3. Uncomment the relevant dependency in `pyproject.toml`
4. Re-install: `pip install -e .`

## Project structure

```
src/agent_forge/
├── core/
│   ├── agent.py       # BaseAgent ABC + AgentStatus
│   ├── factory.py     # AgentFactory ABC
│   └── manager.py     # AgentManager (orchestrator)
├── backends/
│   ├── autogen/       # ✅ AutoGen implementation
│   ├── langgraph/     # 🔧 LangGraph stub
│   └── anthropic/     # 🔧 Anthropic stub
├── config/
│   └── settings.py    # Provider config + env resolution
└── main.py            # Demo entrypoint
```
