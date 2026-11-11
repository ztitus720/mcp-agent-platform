"""Stage-2 tests for Server-A (knowledge).

Needs the network (Bailian embedding) and a built index:
    python -m scripts.ingest
    python -m pytest tests/test_knowledge_server.py -v
"""

import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters
from mcp.server.mcpserver.exceptions import ToolError

from servers.knowledge_server import resolve_doc_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KNOWLEDGE_SERVER = StdioServerParameters(
    command=sys.executable,
    args=["-m", "servers.knowledge_server"],
    cwd=str(PROJECT_ROOT),
)


def text_of(result) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


# ---------------------------------------------------------------------------
# Layer 1: plain logic (no network)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "../.env",
    "..\\.env",
    "../../requirements.txt",
    "C:\\Windows\\win.ini",
    "missing.md",
    "",
])
def test_resolve_rejects_anything_outside_docs(name):
    with pytest.raises(ToolError, match="Unknown document"):
        resolve_doc_path(name)


def test_resolve_accepts_known_document():
    assert resolve_doc_path("leave_policy.md").name == "leave_policy.md"


# ---------------------------------------------------------------------------
# Layer 2: real MCP protocol over stdio (uses the network)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_knowledge_server_end_to_end():
    async with Client(KNOWLEDGE_SERVER) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
        taxi = await client.call_tool("search_knowledge", {"query": "西安本地打车每天能报多少钱"})
        docs = await client.call_tool("list_documents", {})
        full = await client.call_tool("read_document", {"filename": "leave_policy.md"})
        escape = await client.call_tool("read_document", {"filename": "../.env"})
        bad_k = await client.call_tool("search_knowledge", {"query": "年假", "top_k": 50})

    assert names == {"search_knowledge", "list_documents", "read_document"}

    assert not taxi.is_error
    first_hit = text_of(taxi).split("\n\n")[0]
    assert "expense_policy.md" in first_hit and "80" in first_hit

    assert text_of(docs).count(".md") == 4
    assert not full.is_error and "满 5 年不满 10 年" in text_of(full)

    assert escape.is_error and "API_KEY" not in text_of(escape)
    assert bad_k.is_error
