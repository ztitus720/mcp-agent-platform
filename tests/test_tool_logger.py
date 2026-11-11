"""Stage-7 tests: one JSONL line per logical tool call, with retries counted (no network).

    python -m pytest tests/test_tool_logger.py -v
"""

import json

import pytest
from langchain_core.messages import AIMessage

from app.agent import build_agent
from app.tool_logger import ToolCallLog
from test_resilience import POLICY, TOOLS, attempts, call
from test_agent import ScriptedModel


async def run_and_read(tmp_path, tool_name: str, thread: str = "t-log") -> list[dict]:
    attempts.clear()
    log_file = tmp_path / "logs" / "tool_calls.jsonl"
    graph = build_agent(TOOLS, POLICY, model=ScriptedModel(script=[call(tool_name), AIMessage(content="final")]),
                        tool_log=ToolCallLog(log_file))
    config = {"configurable": {"thread_id": thread}, "recursion_limit": POLICY.recursion_limit}
    await graph.ainvoke({"messages": [("user", "go")]}, config)
    return [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]


@pytest.mark.anyio
async def test_success_after_retries_is_one_line(tmp_path):
    [entry] = await run_and_read(tmp_path, "flaky")
    assert entry["tool"] == "flaky" and entry["args"] == {"x": 1}
    assert entry["ok"] is True and entry["attempts"] == 3
    assert entry["thread_id"] == "t-log" and entry["result"] == "1"
    assert entry["duration_ms"] >= 0 and entry["ts"].endswith("+00:00")


@pytest.mark.anyio
async def test_final_failure_is_logged_as_not_ok(tmp_path):
    [entry] = await run_and_read(tmp_path, "dead")
    assert entry["ok"] is False and entry["attempts"] == 1 + POLICY.tool_max_retries
    assert "Tool unavailable after retries" in entry["result"]


@pytest.mark.anyio
async def test_timeout_is_logged(tmp_path):
    [entry] = await run_and_read(tmp_path, "hung")
    assert entry["ok"] is False and "TimeoutError" in entry["result"]
    assert entry["duration_ms"] >= POLICY.tool_timeout_seconds * 1000


@pytest.mark.anyio
async def test_lines_append_across_calls(tmp_path):
    await run_and_read(tmp_path, "flaky", thread="a")
    entries = await run_and_read(tmp_path, "bad_args", thread="b")
    assert [e["thread_id"] for e in entries] == ["a", "b"]
    assert entries[1]["ok"] is False and entries[1]["attempts"] == 1
