"""MCP client layer: read config/servers.yaml, connect to every enabled server, return LangChain tools.

The agent only ever receives the tool list built here. Which servers exist, how they
start and which tools they offer is decided by servers.yaml and the servers themselves.
"""

import asyncio
import logging
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from fastmcp import Client
from fastmcp.client.group import ClientGroup
from langchain_core.tools import BaseTool

from core.settings import PROJECT_ROOT

# langchain.mcp is beta (chosen deliberately in stage 0); versions are pinned, so silence the notice.
warnings.filterwarnings("ignore", message=r".*langchain\.mcp.* is in beta")
from langchain.mcp import MCPAdapter  # noqa: E402

logger = logging.getLogger(__name__)

SERVERS_FILE = PROJECT_ROOT / "config" / "servers.yaml"
LOAD_TIMEOUT_SECONDS = 30
MAX_TOOL_NAME_LENGTH = 64          # function-calling APIs reject longer names
_SERVER_NAME = re.compile(r"^[a-z][a-z0-9_]{0,30}$")


@dataclass
class ServerSpec:
    name: str
    command: str
    args: list[str]
    cwd: str
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class LoadReport:
    tools: list[BaseTool] = field(default_factory=list)
    loaded: dict[str, list[str]] = field(default_factory=dict)   # server -> tool names
    failed: dict[str, str] = field(default_factory=dict)         # server -> reason
    disabled: list[str] = field(default_factory=list)            # enabled: false in servers.yaml


def _expand(value: object) -> str:
    return str(value).replace("${PYTHON}", sys.executable).replace("${PROJECT_ROOT}", str(PROJECT_ROOT))


def read_server_specs(path: Path = SERVERS_FILE) -> list[ServerSpec]:
    """Parse and validate servers.yaml. Bad config is a ValueError, not a silent skip."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    servers = data.get("servers")
    if not isinstance(servers, dict) or not servers:
        raise ValueError(f"{path} defines no servers")

    specs: list[ServerSpec] = []
    for name, raw in servers.items():
        raw = raw or {}
        if not isinstance(name, str):   # YAML reads bare yes/no/on/off/true as booleans
            raise ValueError(f"Server name {name!r} is not text; quote it in servers.yaml")
        if not _SERVER_NAME.match(name):
            raise ValueError(f"Invalid server name {name!r}: use lowercase letters, digits and '_'")
        if not raw.get("enabled", True):
            continue
        if "command" not in raw:
            raise ValueError(f"Server {name!r} has no 'command'")
        specs.append(ServerSpec(
            name=name,
            command=_expand(raw["command"]),
            args=[_expand(a) for a in raw.get("args", [])],
            cwd=_expand(raw.get("cwd", "${PROJECT_ROOT}")),
            env={str(k): _expand(v) for k, v in (raw.get("env") or {}).items()},
            description=str(raw.get("description", "")),
        ))
    return specs


async def _load_one(spec: ServerSpec) -> list[BaseTool]:
    """Connect to ONE server and list its tools. Tools stay callable after this returns."""
    entry: dict = {"command": spec.command, "args": spec.args, "cwd": spec.cwd}
    if spec.env:
        entry["env"] = spec.env
    # A one-member ClientGroup gives this server its own connection and prefixes its tools.
    group = ClientGroup({spec.name: Client({"mcpServers": {spec.name: entry}})})
    async with MCPAdapter(group) as adapter:
        tools = await adapter.list_tools()
    for tool in tools:
        if len(tool.name) > MAX_TOOL_NAME_LENGTH:
            raise ValueError(f"Tool name {tool.name!r} is longer than {MAX_TOOL_NAME_LENGTH} chars")
    return tools


async def load_mcp_tools(path: Path = SERVERS_FILE) -> LoadReport:
    """Load every enabled server in parallel. One failing server is reported, never fatal."""
    specs = read_server_specs(path)
    enabled = {spec.name for spec in specs}
    all_names = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("servers") or {}
    results = await asyncio.gather(
        *(asyncio.wait_for(_load_one(spec), LOAD_TIMEOUT_SECONDS) for spec in specs),
        return_exceptions=True,
    )
    report = LoadReport(disabled=[str(n) for n in all_names if str(n) not in enabled])
    for spec, result in zip(specs, results):
        if isinstance(result, BaseException):
            reason = f"{type(result).__name__}: {result}".strip().rstrip(":")[:300]
            report.failed[spec.name] = reason
            logger.warning("MCP server %r failed to load: %s", spec.name, reason)
        else:
            report.tools.extend(result)
            report.loaded[spec.name] = [tool.name for tool in result]
    return report
