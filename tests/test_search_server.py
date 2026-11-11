"""Stage-3 tests for Server-C (search).

The protocol test adapts to the current mode: with no TAVILY_API_KEY it checks mock
behaviour; with a key it checks real Tavily output (needs network access to Tavily).
    python -m pytest tests/test_search_server.py -v
"""

import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters
from mcp.server.mcpserver.exceptions import ToolError

from core.settings import get_settings
from servers.search_server import format_results, mock_results, validate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODE = get_settings().search_mode

SEARCH_SERVER = StdioServerParameters(
    command=sys.executable,
    args=["-m", "servers.search_server"],
    cwd=str(PROJECT_ROOT),
)


def text_of(result) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


# ---------------------------------------------------------------------------
# Layer 1: plain logic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("query", "max_results"), [
    ("", 5),
    ("   ", 5),
    ("x" * 401, 5),
    ("ok", 0),
    ("ok", 9),
])
def test_validate_rejects(query, max_results):
    with pytest.raises(ToolError):
        validate(query, max_results)


def test_mock_output_is_clearly_labelled():
    text = format_results("天气", mock_results("天气", 5), "mock")
    assert text.startswith("[MOCK MODE]")
    assert "NOT real information" in text
    assert "example.invalid" in text


def test_real_output_format_and_truncation():
    fake = [{"title": "T", "url": "https://a.example", "content": "c" * 900}]
    text = format_results("q", fake, "tavily")
    assert "[1] T" in text and "URL: https://a.example" in text
    assert "c" * 501 not in text


# ---------------------------------------------------------------------------
# Layer 2: real MCP protocol over stdio
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_search_server_end_to_end():
    async with Client(SEARCH_SERVER) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
        ok = await client.call_tool("web_search", {"query": "MCP protocol", "max_results": 3})
        bad = await client.call_tool("web_search", {"query": "", "max_results": 3})

    assert names == {"web_search"}
    assert bad.is_error
    assert not ok.is_error, text_of(ok)
    if MODE == "mock":
        assert text_of(ok).startswith("[MOCK MODE]")
    else:
        assert "source: Tavily" in text_of(ok)
