from __future__ import annotations

"""
agent-forge FastAPI backend.

Run:
    uvicorn agent_forge.api.app:app --reload --port 8000
"""

import json
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_forge.backends.autogen import AutoGenFactory
from agent_forge.core.conversation import AgentConversation, ConversationMessage, StopSignal
from agent_forge.core.manager import AgentManager
from agent_forge.core.orchestrator import AgentSpec, Orchestrator, OrchestratorToolCall
from agent_forge.tools import list_tools

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="agent-forge", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    goal: str
    max_rounds: int = 3
    provider: str = "openai"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _event(type: str, **data) -> str:
    """Format a single SSE event."""
    return f"data: {json.dumps({'type': type, **data})}\n\n"


async def _run_stream(req: RunRequest) -> AsyncGenerator[str, None]:
    """Full agent-forge pipeline as an SSE stream."""
    try:
        orchestrator = Orchestrator(provider=req.provider)

        # ── Phase 1: Research + plan ──────────────────────────────────────────
        specs: list[AgentSpec] = []
        async for item in orchestrator.plan_stream(req.goal):
            if isinstance(item, OrchestratorToolCall):
                yield _event(
                    "orchestrator_tool_call",
                    tool=item.tool,
                    args=item.args,
                    result=item.result,
                )
            elif isinstance(item, list):
                specs = item
                yield _event(
                    "plan_ready",
                    specs=[s.model_dump() for s in specs],
                )
            else:
                yield _event("plan_chunk", text=item)

        # ── Phase 2: Spawn agents ─────────────────────────────────────────────
        factory = AutoGenFactory(provider=req.provider)
        manager = AgentManager(factory)
        agents = []
        for spec in specs:
            agent = await manager.spawn(
                role=spec.role_description,
                name=spec.name,
                system_message=spec.system_prompt,
                tools=spec.tools or None,
            )
            agents.append(agent)

        # ── Phase 3: Conversation ─────────────────────────────────────────────
        conversation = AgentConversation(
            manager=manager,
            agents=agents,
            orchestrator=orchestrator,
            max_rounds=req.max_rounds,
        )
        async for item in conversation.run_stream(req.goal):
            if isinstance(item, ConversationMessage):
                yield _event(
                    "agent_message",
                    agent=item.agent,
                    content=item.content,
                    round=item.round,
                )
            elif isinstance(item, StopSignal):
                yield _event(
                    "stop_signal",
                    reason=item.reason,
                    stopped_by=item.stopped_by,
                )

        # ── Phase 4: Synthesize ───────────────────────────────────────────────
        async for chunk in orchestrator.synthesize_stream(req.goal, conversation.to_context_text()):
            yield _event("synthesis_chunk", text=chunk)

        # ── Phase 5: Shutdown ─────────────────────────────────────────────────
        await manager.shutdown()
        yield _event("done")

    except Exception as exc:
        yield _event("error", message=str(exc))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tools")
def get_tools():
    return {"tools": list_tools()}


@app.post("/run/stream")
async def run_stream(req: RunRequest):
    return StreamingResponse(
        _run_stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
