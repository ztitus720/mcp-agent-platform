"""Stage-6 tests: retries, timeouts and model fallback (scripted models, no network).

    python -m pytest tests/test_resilience.py -v
"""

import asyncio
import time

import pytest
from langchain_core.exceptions import ModelNotFoundError
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from app.agent import build_agent
from app.policy import AgentPolicy
from test_agent import ScriptedModel

POLICY = AgentPolicy(max_model_calls=4, max_tool_calls=10, recursion_limit=40,
                     system_prompt="test", tool_timeout_seconds=0.5,
                     tool_max_retries=2, retry_initial_delay_seconds=0.0)
attempts: dict[str, int] = {}


@tool
async def flaky(x: int) -> int:
    """Fails twice with a connection error, then works."""
    attempts["flaky"] = attempts.get("flaky", 0) + 1
    if attempts["flaky"] < 3:
        raise ConnectionError("server restarting")
    return x


@tool
async def dead(x: int) -> int:
    """Always fails: its server is gone."""
    attempts["dead"] = attempts.get("dead", 0) + 1
    raise ConnectionError("server gone")


@tool
async def hung(x: int) -> int:
    """Never answers in time."""
    await asyncio.sleep(30)
    return x


@tool
async def bad_args(x: int) -> int:
    """Rejects its input: retrying cannot help."""
    attempts["bad_args"] = attempts.get("bad_args", 0) + 1
    raise ValueError("x must be positive")


TOOLS = [flaky, dead, hung, bad_args]


class BrokenModel(ScriptedModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        raise ConnectionError("model endpoint down")


class UnknownModel(ScriptedModel):
    """Like a real 404 'model does not exist': LangChain marks it NON-retryable."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        raise ModelNotFoundError("The model `no-such-model` does not exist")


def call(name: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": {"x": 1}, "id": f"id-{name}"}])


async def run(model, fallback=None):
    graph = build_agent(TOOLS, POLICY, model=model, fallback_model=fallback)
    config = {"configurable": {"thread_id": "t"}, "recursion_limit": POLICY.recursion_limit}
    return await graph.ainvoke({"messages": [("user", "go")]}, config)


def tool_result(result) -> ToolMessage:
    return next(m for m in result["messages"] if isinstance(m, ToolMessage))


@pytest.fixture(autouse=True)
def reset_attempts():
    attempts.clear()


@pytest.mark.anyio
async def test_transient_tool_failure_is_retried_until_success():
    result = await run(ScriptedModel(script=[call("flaky"), AIMessage(content="final")]))
    assert attempts["flaky"] == 3
    assert tool_result(result).status == "success" and tool_result(result).content == "1"


@pytest.mark.anyio
async def test_dead_tool_degrades_instead_of_crashing():
    result = await run(ScriptedModel(script=[call("dead"), AIMessage(content="answered without it")]))
    assert attempts["dead"] == 1 + POLICY.tool_max_retries
    assert tool_result(result).status == "error"
    assert "Tool unavailable after retries" in tool_result(result).content
    assert result["messages"][-1].content == "answered without it"


@pytest.mark.anyio
async def test_hung_tool_times_out():
    start = time.perf_counter()
    result = await run(ScriptedModel(script=[call("hung"), AIMessage(content="final")]))
    elapsed = time.perf_counter() - start
    assert "TimeoutError" in tool_result(result).content
    assert elapsed < POLICY.tool_timeout_seconds * (POLICY.tool_max_retries + 1) + 5


@pytest.mark.anyio
async def test_bad_arguments_are_not_retried():
    result = await run(ScriptedModel(script=[call("bad_args"), AIMessage(content="final")]))
    assert attempts["bad_args"] == 1
    assert tool_result(result).status == "error"
    assert "ValueError" in tool_result(result).content
    assert result["messages"][-1].content == "final"


@pytest.mark.anyio
async def test_broken_model_falls_back():
    primary = BrokenModel(script=[AIMessage(content="never")])
    backup = ScriptedModel(script=[AIMessage(content="answer from the fallback model")])
    result = await run(primary, fallback=backup)
    assert primary.calls >= 1
    assert result["messages"][-1].content == "answer from the fallback model"


@pytest.mark.anyio
async def test_broken_model_without_fallback_ends_politely():
    result = await run(BrokenModel(script=[AIMessage(content="never")]))   # must not raise
    assert "language model is unavailable" in result["messages"][-1].content


@pytest.mark.anyio
async def test_unknown_model_falls_back():
    backup = ScriptedModel(script=[AIMessage(content="answer from the fallback model")])
    result = await run(UnknownModel(script=[AIMessage(content="never")]), fallback=backup)
    assert result["messages"][-1].content == "answer from the fallback model"


@pytest.mark.anyio
async def test_non_retryable_model_error_still_ends_politely():
    result = await run(UnknownModel(script=[AIMessage(content="never")]))   # must not raise
    assert "language model is unavailable" in result["messages"][-1].content
    assert "ModelNotFoundError" in result["messages"][-1].content
