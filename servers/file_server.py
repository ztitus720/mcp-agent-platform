"""Server-D: text files inside a sandboxed workspace/ folder (list, read, write).

Added in stage 8 WITHOUT touching agent code: only this file, one entry in
config/servers.yaml and one line in config/policy.yaml (write needs approval).
    python -m servers.file_server
Never print() to stdout in this file: stdout is the MCP wire.
"""

from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from core.settings import PROJECT_ROOT

mcp = MCPServer(
    "files",
    instructions="Read and write text files inside the sandboxed workspace/ folder only.",
)

WORKSPACE_DIR = (PROJECT_ROOT / "workspace").resolve()
ALLOWED_SUFFIXES = {".txt", ".md", ".json", ".csv"}
MAX_READ_CHARS = 20_000
MAX_WRITE_CHARS = 100_000
MAX_LISTED = 200

# ---------------------------------------------------------------------------
# Plain logic (testable without MCP)
# ---------------------------------------------------------------------------


def resolve_in_workspace(relative_path: str) -> Path:
    """Map a caller-supplied path to a text file INSIDE the workspace, or raise ToolError."""
    candidate = (WORKSPACE_DIR / relative_path.strip()).resolve()
    if candidate == WORKSPACE_DIR or not candidate.is_relative_to(WORKSPACE_DIR):
        raise ToolError(f"Path {relative_path!r} is outside the workspace. Use a relative path such as 'notes/todo.md'.")
    if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ToolError(f"Only {sorted(ALLOWED_SUFFIXES)} files are allowed, got {candidate.name!r}.")
    return candidate


def relative(path: Path) -> str:
    return path.relative_to(WORKSPACE_DIR).as_posix()


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_files() -> str:
    """List the text files in the workspace folder (relative paths)."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(relative(p) for p in WORKSPACE_DIR.rglob("*")
                   if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES)
    if not files:
        return "The workspace is empty."
    shown = files[:MAX_LISTED]
    more = f"\n... and {len(files) - MAX_LISTED} more" if len(files) > MAX_LISTED else ""
    return "\n".join(shown) + more


@mcp.tool()
def read_file(path: str) -> str:
    """Read a text file from the workspace. `path` is relative, e.g. "notes/todo.md"."""
    target = resolve_in_workspace(path)
    if not target.is_file():
        raise ToolError(f"File {path!r} does not exist. Call list_files to see what is there.")
    text = target.read_text(encoding="utf-8")
    return text if len(text) <= MAX_READ_CHARS else text[:MAX_READ_CHARS] + "\n...[truncated]"


@mcp.tool()
def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """Create a text file in the workspace (UTF-8). This changes data on disk.

    `path` is relative, e.g. "notes/todo.md"; folders are created as needed.
    An existing file is only replaced when overwrite is true.
    """
    target = resolve_in_workspace(path)
    if len(content) > MAX_WRITE_CHARS:
        raise ToolError(f"Content is longer than {MAX_WRITE_CHARS} characters.")
    if target.exists() and not overwrite:
        raise ToolError(f"File {path!r} already exists. Set overwrite to true to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {relative(target)}."


if __name__ == "__main__":
    mcp.run()  # stdio
