"""Stage-1 tests for Server-B (utility).

Two layers:
  * plain-logic tests: call the Python functions directly; fast, no MCP involved.
  * protocol tests: spawn the real server over stdio and talk MCP to it,
    exactly the way the agent will in stage 4.

Run from the project root:
    python -m pytest tests -v
"""

import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters
from mcp.server.mcpserver.exceptions import ToolError

from servers.utility_server import evaluate_expression, format_now

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UTILITY_SERVER = StdioServerParameters(
    command=sys.executable,                 # the same .venv python that runs pytest
    args=["-m", "servers.utility_server"],
    cwd=str(PROJECT_ROOT),
)


def text_of(result) -> str:
    """Join the text blocks of a tool result into one string."""
    return "".join(getattr(block, "text", "") for block in result.content)


# ---------------------------------------------------------------------------
# Layer 1: plain logic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("expression", "expected"), [
    ("(3 + 5) * 12", "96"),
    ("2 ** 10 / 7", "146.285714286"),
    ("0.1 + 0.2", "0.3"),
    ("-(4 - 10) // 4", "1"),
    ("3×4", "12"),
    ("2^10", "1024"),
])
def test_evaluate_valid(expression, expected):
    assert evaluate_expression(expression) == expected


@pytest.mark.parametrize(("expression", "reason"), [
    ("1/0", "Division by zero"),
    ("abs(-3)", "Unsupported element 'Call'"),
    ("x + 1", "Unsupported element 'Name'"),
    ("__import__(chr(111)+chr(115)).getcwd()", "Unsupported element"),
    ("9 ** 9 ** 9", "too large"),
    ("", "empty"),
    ("1 +", "Could not parse"),
])
def test_evaluate_rejects(expression, reason):
    with pytest.raises(ToolError, match=reason):
        evaluate_expression(expression)


def test_format_now_rejects_unknown_timezone():
    with pytest.raises(ToolError, match="Unknown timezone"):
        format_now("Mars/Base")


# ---------------------------------------------------------------------------
# Layer 2: real MCP protocol over stdio
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_server_lists_tools_with_schema():
    async with Client(UTILITY_SERVER) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert set(tools) == {"calculate", "get_current_time"}
    assert tools["calculate"].input_schema["required"] == ["expression"]
    assert "ANY math" in tools["calculate"].description


@pytest.mark.anyio
async def test_server_calls_and_reports_errors():
    async with Client(UTILITY_SERVER) as client:
        ok = await client.call_tool("calculate", {"expression": "(3 + 5) * 12"})
        bad = await client.call_tool("calculate", {"expression": "1/0"})
        after = await client.call_tool("calculate", {"expression": "2 ** 10"})
        now = await client.call_tool("get_current_time", {})
        bad_tz = await client.call_tool("get_current_time", {"timezone": "Mars/Base"})

    assert not ok.is_error and text_of(ok) == "96"
    assert bad.is_error and "Division by zero" in text_of(bad)
    assert not after.is_error and text_of(after) == "1024"   # server survived the error
    assert not now.is_error and "Asia/Shanghai" in text_of(now)
    assert bad_tz.is_error and "Unknown timezone" in text_of(bad_tz)