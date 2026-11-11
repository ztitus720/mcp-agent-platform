"""Stage-8 tests for Server-D (files): sandbox escape attempts and the real protocol.

    python -m pytest tests/test_file_server.py -v
"""

import sys

import pytest
from mcp import Client, StdioServerParameters
from mcp.server.mcpserver.exceptions import ToolError

import servers.file_server as fs
from core.settings import PROJECT_ROOT

FILE_SERVER = StdioServerParameters(command=sys.executable, args=["-m", "servers.file_server"],
                                    cwd=str(PROJECT_ROOT))
PROBE = "pytest_probe/hello.txt"


def text_of(result) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    monkeypatch.setattr(fs, "WORKSPACE_DIR", workspace)
    return workspace


# ---------------------------------------------------------------------------
# Layer 1: path rules (no MCP)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "../.env", "..\\.env", "../../requirements.txt", "C:\\Windows\\win.ini",
    "/etc/passwd.txt", "", ".", "notes/../../escape.txt",
])
def test_paths_outside_workspace_are_rejected(sandbox, path):
    with pytest.raises(ToolError):
        fs.resolve_in_workspace(path)


@pytest.mark.parametrize("path", ["run.py", "notes/app.exe", "data.db"])
def test_only_text_suffixes_are_allowed(sandbox, path):
    with pytest.raises(ToolError, match="Only"):
        fs.resolve_in_workspace(path)


def test_subfolders_inside_workspace_are_fine(sandbox):
    assert fs.resolve_in_workspace("notes/2026/plan.md") == sandbox / "notes" / "2026" / "plan.md"


# ---------------------------------------------------------------------------
# Layer 2: real MCP protocol over stdio (writes one probe file, then removes it)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_file_server_end_to_end():
    probe = fs.WORKSPACE_DIR / PROBE
    try:
        async with Client(FILE_SERVER) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            wrote = await client.call_tool("write_file", {"path": PROBE, "content": "西安 80 元"})
            again = await client.call_tool("write_file", {"path": PROBE, "content": "x"})
            listed = await client.call_tool("list_files", {})
            read = await client.call_tool("read_file", {"path": PROBE})
            escape = await client.call_tool("read_file", {"path": "../.env"})

        assert names == {"list_files", "read_file", "write_file"}
        assert not wrote.is_error and "Wrote" in text_of(wrote)
        assert again.is_error and "already exists" in text_of(again)      # no silent overwrite
        assert PROBE in text_of(listed)
        assert text_of(read) == "西安 80 元"
        assert escape.is_error and "API_KEY" not in text_of(escape)
    finally:
        probe.unlink(missing_ok=True)
        if probe.parent.exists() and not any(probe.parent.iterdir()):
            probe.parent.rmdir()
