"""Stage-5 tests for the agent layer. A scripted model replaces the LLM: no network needed.

    python -m pytest tests/test_agent.py -v
"""

import re
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from app.agent import build_agent
from app.policy import AgentPolicy

POLICY = AgentPolicy(max_model_calls=4, max_tool_calls=10, recursion_limit=40,
                     system_prompt="test prompt")


class ScriptedModel(BaseChatModel):
    """Returns pre-written AI messages in order; repeats the last one forever."""

    script: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def tool_call(call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": call_id}])


def config(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}, "recursion_limit": POLICY.recursion_limit}


def test_agent_uses_tool_then_answers():
    model = ScriptedModel(script=[tool_call("c1"), AIMessage(content="The sum is 5.")])
    graph = build_agent([add], POLICY, model=model)
    result = graph.invoke({"messages": [("user", "2+3?")]}, config("t1"))

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.content for m in tool_messages] == ["5"]
    assert result["messages"][-1].content == "The sum is 5."


def test_runaway_tool_loop_is_stopped_gracefully():
    model = ScriptedModel(script=[tool_call(f"c{i}") for i in range(50)])
    graph = build_agent([add], POLICY, model=model)
    result = graph.invoke({"messages": [("user", "loop forever")]}, config("t2"))  # must not raise

    assert model.calls == POLICY.max_model_calls
    assert isinstance(result["messages"][-1], AIMessage)


def test_memory_is_kept_per_thread():
    model = ScriptedModel(script=[AIMessage(content="ok")])
    graph = build_agent([add], POLICY, model=model)
    graph.invoke({"messages": [("user", "my name is Titus")]}, config("a"))
    second = graph.invoke({"messages": [("user", "what is my name?")]}, config("a"))
    other = graph.invoke({"messages": [("user", "hello")]}, config("b"))

    assert any("Titus" in str(m.content) for m in second["messages"])
    assert not any("Titus" in str(m.content) for m in other["messages"])


def test_agent_module_names_no_tool():
    """The decoupling rule, enforced: agent.py must not mention any concrete tool."""
    source = (Path(__file__).resolve().parent.parent / "app" / "agent.py").read_text(encoding="utf-8")
    for name in ["calculate", "get_current_time", "search_knowledge", "list_documents",
                 "read_document", "web_search"]:
        assert not re.search(rf"\b{name}\b", source), f"agent.py mentions tool {name!r}"
