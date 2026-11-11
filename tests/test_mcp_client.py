"""Stage-4 tests for the MCP client layer (app/mcp_client.py).

    python -m pytest tests/test_mcp_client.py -v
"""

import sys

import pytest

import app.mcp_client as mcp_client
from app.mcp_client import load_mcp_tools, read_server_specs
from core.settings import PROJECT_ROOT


def write_registry(tmp_path, body: str):
    path = tmp_path / "servers.yaml"
    path.write_text(body, encoding="utf-8")
    return path


UTILITY = """
  utility:
    command: ${PYTHON}
    args: ["-m", "servers.utility_server"]
"""

# ---------------------------------------------------------------------------
# Layer 1: parsing and validation of servers.yaml (no processes started)
# ---------------------------------------------------------------------------


def test_specs_expand_placeholders_and_skip_disabled(tmp_path):
    path = write_registry(tmp_path, "servers:" + UTILITY + """
  spare:
    enabled: false
    command: whatever
""")
    specs = read_server_specs(path)
    assert [s.name for s in specs] == ["utility"]
    assert specs[0].command == sys.executable
    assert specs[0].cwd == str(PROJECT_ROOT)


@pytest.mark.parametrize("body", [
    "servers: {}",
    "servers:\n  Bad-Name:\n    command: x",
    "servers:\n  nocmd:\n    args: []",
    "servers:\n  off:\n    command: x",
])
def test_specs_reject_bad_config(tmp_path, body):
    with pytest.raises(ValueError):
        read_server_specs(write_registry(tmp_path, body))


# ---------------------------------------------------------------------------
# Layer 2: real servers over stdio
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_real_registry_loads_all_servers():
    # Expectations come from servers.yaml itself, so adding a server needs no test edit.
    expected = {spec.name for spec in read_server_specs()}
    report = await load_mcp_tools()
    assert report.failed == {}
    assert set(report.loaded) == expected
    assert all(report.loaded[name] for name in expected)        # every server offers tools
    assert all(tool.name.split("_", 1)[0] in expected for tool in report.tools)  # prefixed


@pytest.mark.anyio
async def test_broken_servers_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_client, "LOAD_TIMEOUT_SECONDS", 5)
    path = write_registry(tmp_path, "servers:" + UTILITY + """
  typo:
    command: ${PYTHON}
    args: ["-m", "servers.does_not_exist"]
  ghost:
    command: no-such-command-xyz
  sleepy:
    command: ${PYTHON}
    args: ["-c", "import time; time.sleep(60)"]
""")
    report = await load_mcp_tools(path)
    assert list(report.loaded) == ["utility"]
    assert set(report.failed) == {"typo", "ghost", "sleepy"}
    assert "TimeoutError" in report.failed["sleepy"]


@pytest.mark.anyio
async def test_loaded_tools_are_callable_and_report_errors():
    report = await load_mcp_tools()
    tools = {tool.name: tool for tool in report.tools}
    calc = tools["utility_calculate"]

    ok = await calc.ainvoke({"expression": "(3 + 5) * 12"})
    assert ok[0]["text"] == "96"

    failed = await calc.ainvoke({
        "type": "tool_call", "id": "t1", "name": "utility_calculate",
        "args": {"expression": "1/0"},
    })
    assert failed.status == "error" and "Division by zero" in str(failed.content)
