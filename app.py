from __future__ import annotations

"""
Streamlit UI for agent-forge.

Run:
    streamlit run app.py
"""

import asyncio
import queue
import threading

import streamlit as st

from agent_forge.backends.autogen import AutoGenFactory
from agent_forge.core.manager import AgentManager
from agent_forge.core.orchestrator import AgentSpec, Orchestrator
from agent_forge.core.shared_thread import SharedThread

# ── Async helpers ─────────────────────────────────────────────────────────────

def run_async(coro):
    """Run an async coroutine synchronously in a dedicated thread."""
    result_box: list = [None]
    error_box: list = [None]

    def target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_box[0] = loop.run_until_complete(coro)
        except Exception as exc:
            error_box[0] = exc
        finally:
            loop.close()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join()
    if error_box[0]:
        raise error_box[0]
    return result_box[0]


def iter_async_stream(async_gen_factory):
    """
    Sync generator that bridges any async generator to Streamlit.
    Pass a zero-argument lambda that returns the async generator.
    """
    q: queue.Queue = queue.Queue()
    sentinel = object()

    async def drain():
        async for item in async_gen_factory():
            q.put(item)
        q.put(sentinel)

    def target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(drain())
        finally:
            loop.close()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item
    t.join()


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="agent-forge", page_icon="🔨", layout="wide")
st.title("🔨 agent-forge")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_chat, tab_orch, tab_agents = st.tabs(["💬 Chat", "🧠 Orchestrator", "🤖 Agent Activity"])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────

with tab_chat:
    DEFAULT_GOAL = (
        "Analyse the impact of rising US interest rates on emerging market equities "
        "and produce a concise investment brief."
    )
    goal = st.text_area("Goal", value=DEFAULT_GOAL, height=80)
    run_btn = st.button("▶ Run", type="primary", disabled=not goal.strip())
    st.divider()
    chat_status = st.empty()
    final_report = st.empty()

# ── Tab 2: Orchestrator ───────────────────────────────────────────────────────

with tab_orch:
    orch_status = st.empty()
    orch_raw = st.empty()
    orch_specs_area = st.container()

# ── Tab 3: Agent Activity ─────────────────────────────────────────────────────

with tab_agents:
    agents_cards_area = st.container()
    st.divider()
    st.markdown("#### 💬 Shared Thread")
    thread_display = st.empty()

# ── Stop here until Run is clicked ───────────────────────────────────────────

if not run_btn:
    st.stop()

# ── Step 1: Stream orchestrator planning → Tab 2 ─────────────────────────────

chat_status.info("⏳ Planning agents...")
orch_status.info("Planning agents for your goal...")

full_text = ""
specs: list[AgentSpec] = []

orchestrator = Orchestrator(provider="openai")

for item in iter_async_stream(lambda: orchestrator.plan_stream(goal)):
    if isinstance(item, list):
        specs = item
    else:
        full_text += item
        orch_raw.code(full_text + "▌", language="json")

orch_raw.code(full_text, language="json")
orch_status.success(f"Planned {len(specs)} agent(s)")

with orch_specs_area:
    st.markdown("#### Agent specs")
    for spec in specs:
        with st.expander(f"`{spec.name}` — {spec.role_description}"):
            st.markdown(f"**System prompt**\n\n{spec.system_prompt}")
            if spec.tools:
                st.markdown(f"**Tools:** {', '.join(spec.tools)}")

# ── Step 2: Create agent cards → Tab 3 ───────────────────────────────────────

status_slots: dict[str, st.delta_generator.DeltaGenerator] = {}
result_slots: dict[str, st.delta_generator.DeltaGenerator] = {}

with agents_cards_area:
    st.markdown("#### Agent outputs")
    for spec in specs:
        with st.container(border=True):
            st.markdown(f"**`{spec.name}`**  \n{spec.role_description}")
            status_slots[spec.name] = st.empty()
            result_slots[spec.name] = st.empty()
            status_slots[spec.name].info("Waiting...")

# ── Step 3: Spawn + run agents ────────────────────────────────────────────────

chat_status.info("⏳ Agents running...")

factory = AutoGenFactory(provider="openai")
manager = AgentManager(factory)
thread = SharedThread()

agents_map: dict[str, tuple[AgentSpec, object]] = {}
for spec in specs:
    agent = run_async(manager.spawn(
        role=spec.role_description,
        name=spec.name,
        system_message=spec.system_prompt,
    ))
    agents_map[spec.name] = (spec, agent)

for spec_name, (spec, agent) in agents_map.items():
    status_slots[spec_name].warning("⚙️ Running...")
    result = run_async(manager.run_task(agent.agent_id, goal, thread=thread))
    status_slots[spec_name].success("✅ Done")
    result_slots[spec_name].markdown(result)

    # Update shared thread display after each agent
    thread_display.markdown(
        "\n\n---\n\n".join(
            f"**`{msg.agent}`**\n\n{msg.content}"
            for msg in thread.messages()
        )
    )

# ── Step 4: Synthesize → Tab 1 ────────────────────────────────────────────────

chat_status.info("⏳ Synthesizing final report...")

synthesis_text = ""
for chunk in iter_async_stream(lambda: orchestrator.synthesize_stream(goal, thread)):
    synthesis_text += chunk
    final_report.markdown(synthesis_text + "▌")

final_report.markdown(synthesis_text)
chat_status.success("✅ Done")

# ── Step 5: Shutdown ──────────────────────────────────────────────────────────

run_async(manager.shutdown())
