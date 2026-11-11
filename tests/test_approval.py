"""Stage-6 tests: human approval before high-risk tools (scripted model, no network).

    python -m pytest tests/test_approval.py -v
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from app.agent import AgentRuntime, build_agent
from app.mcp_client import LoadReport
from app.policy import AgentPolicy
from test_agent import ScriptedModel

POLICY = AgentPolicy(max_model_calls=4, max_tool_calls=10, recursion_limit=40,
                     system_prompt="test", retry_initial_delay_seconds=0.0,
                     high_risk_tools=("risky",))
executed: list[str] = []


@tool
async def risky(x: int) -> str:
    """A high-risk action."""
    executed.append("risky")
    return "risky done"


@tool
async def safe(x: int) -> str:
    """A harmless action."""
    executed.append("safe")
    return "safe done"


def calls(*names: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": n, "args": {"x": 1}, "id": f"id-{i}-{n}"} for i, n in enumerate(names)])


def runtime(*script: AIMessage) -> AgentRuntime:
    graph = build_agent([risky, safe], POLICY, model=ScriptedModel(script=list(script)))
    return AgentRuntime(graph=graph, policy=POLICY, tool_report=LoadReport())


async def start(rt: AgentRuntime, thread: str = "t"):
    return await rt.graph.ainvoke({"messages": [("user", "go")]}, rt.run_config(thread))


@pytest.fixture(autouse=True)
def reset():
    executed.clear()


@pytest.mark.anyio
async def test_high_risk_tool_pauses_before_running():
    rt = runtime(calls("risky"), AIMessage(content="final"))
    await start(rt)
    pending = await rt.pending_approvals("t")
    assert [p["name"] for p in pending] == ["risky"]
    assert executed == []                                 # nothing ran yet


@pytest.mark.anyio
async def test_approve_runs_the_tool():
    rt = runtime(calls("risky"), AIMessage(content="final"))
    await start(rt)
    result = await rt.graph.ainvoke(rt.resume([{"type": "approve"}]), rt.run_config("t"))
    assert executed == ["risky"]
    assert result["messages"][-1].content == "final"
    assert await rt.pending_approvals("t") == []


@pytest.mark.anyio
async def test_reject_skips_the_tool_and_tells_the_model():
    rt = runtime(calls("risky"), AIMessage(content="answered without it"))
    await start(rt)
    result = await rt.graph.ainvoke(
        rt.resume([{"type": "reject", "message": "external call not allowed"}]), rt.run_config("t"))
    assert executed == []
    rejection = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert rejection.status == "error" and "external call not allowed" in rejection.content
    assert result["messages"][-1].content == "answered without it"


@pytest.mark.anyio
async def test_safe_tool_never_pauses():
    rt = runtime(calls("safe"), AIMessage(content="final"))
    result = await start(rt)
    assert executed == ["safe"]
    assert await rt.pending_approvals("t") == []
    assert result["messages"][-1].content == "final"


@pytest.mark.anyio
async def test_mixed_parallel_calls_only_ask_about_the_risky_one():
    rt = runtime(calls("safe", "risky"), AIMessage(content="final"))
    await start(rt)
    pending = await rt.pending_approvals("t")
    assert [p["name"] for p in pending] == ["risky"]
    await rt.graph.ainvoke(rt.resume([{"type": "approve"}]), rt.run_config("t"))
    assert sorted(executed) == ["risky", "safe"]


@pytest.mark.anyio
async def test_pause_is_per_conversation():
    rt = runtime(calls("risky"), AIMessage(content="final"))
    await start(rt, "user-a")
    assert await rt.pending_approvals("user-a")
    assert await rt.pending_approvals("user-b") == []
