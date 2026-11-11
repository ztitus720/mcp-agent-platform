"""Show every tool the agent would see: discovered live from the MCP servers in servers.yaml.

Run from the project root:
    python -m scripts.inspect_tools                 # uses config/servers.yaml
    python -m scripts.inspect_tools other.yaml      # any other registry file
Exit code is 0 only if every enabled server loaded.
"""

import asyncio
import sys
from pathlib import Path

from app.mcp_client import SERVERS_FILE, load_mcp_tools, read_server_specs


def _schema_of(tool) -> dict:
    schema = tool.args_schema
    return schema if isinstance(schema, dict) else schema.model_json_schema()


def describe(tool) -> list[str]:
    """One tool -> a few human-readable lines: name, first docstring line, parameters."""
    summary = (tool.description or "").strip().splitlines()[0] if tool.description else "(no description)"
    schema = _schema_of(tool)
    required = set(schema.get("required", []))
    lines = [f"    {tool.name}", f"        {summary}"]
    for name, prop in schema.get("properties", {}).items():
        flag = "required" if name in required else f"default={prop.get('default')!r}"
        lines.append(f"        - {name}: {prop.get('type', 'any')} ({flag})")
    return lines


async def main(path: Path) -> int:
    specs = {spec.name: spec for spec in read_server_specs(path)}
    report = await load_mcp_tools(path)
    tools_by_name = {tool.name: tool for tool in report.tools}

    print(f"Registry: {path}")
    for name, spec in specs.items():
        if name in report.loaded:
            print(f"\n[OK  ] {name} - {spec.description} ({len(report.loaded[name])} tools)")
            for tool_name in report.loaded[name]:
                print("\n".join(describe(tools_by_name[tool_name])))
        else:
            print(f"\n[FAIL] {name} - {report.failed[name]}")

    print("-" * 60)
    print(f"{len(report.tools)} tools from {len(report.loaded)}/{len(specs)} servers")
    return 1 if report.failed else 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else SERVERS_FILE
    sys.exit(asyncio.run(main(target)))
