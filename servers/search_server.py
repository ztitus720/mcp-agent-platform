"""Server-C: live web search via Tavily, with a clearly-labelled mock when no key is set.

Runs as an independent MCP server process over stdio:
    python -m servers.search_server
Never print() to stdout in this file: stdout is the MCP wire. Log to stderr instead.
"""

import logging

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from core.settings import get_settings

logger = logging.getLogger(__name__)

mcp = MCPServer(
    "search",
    instructions="Live web search for public, up-to-date information.",
)

MAX_RESULTS_LIMIT = 8
MAX_QUERY_CHARS = 400
SNIPPET_CHARS = 500
TAVILY_TIMEOUT_SECONDS = 20

MOCK_HEADER = (
    "[MOCK MODE] Web search is not configured (no TAVILY_API_KEY). "
    "The results below are placeholders, NOT real information. "
    "Tell the user that live web search is currently unavailable."
)

# ---------------------------------------------------------------------------
# Plain logic (testable without MCP or network)
# ---------------------------------------------------------------------------


def validate(query: str, max_results: int) -> str:
    query = query.strip()
    if not query:
        raise ToolError("Query is empty.")
    if len(query) > MAX_QUERY_CHARS:
        raise ToolError(f"Query is longer than {MAX_QUERY_CHARS} characters.")
    if not 1 <= max_results <= MAX_RESULTS_LIMIT:
        raise ToolError(f"max_results must be between 1 and {MAX_RESULTS_LIMIT}.")
    return query


def mock_results(query: str, max_results: int) -> list[dict]:
    """Deterministic placeholders; '.invalid' is a domain reserved to never resolve."""
    return [
        {"title": f"Placeholder result {i} for: {query}",
         "url": f"https://example.invalid/mock/{i}",
         "content": "Mock data generated locally. It contains no real information."}
        for i in range(1, min(max_results, 2) + 1)
    ]


def tavily_results(query: str, max_results: int) -> list[dict]:
    """Call Tavily; any failure (network, auth, quota) becomes a ToolError, never a crash."""
    from tavily import TavilyClient

    api_key = get_settings().tavily_api_key.get_secret_value()
    try:
        response = TavilyClient(api_key=api_key).search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            timeout=TAVILY_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.exception("tavily search failed")
        raise ToolError(
            f"Web search is temporarily unavailable ({type(exc).__name__}). "
            "Answer without live data and tell the user so, or try again later."
        ) from None
    return response.get("results", [])


def format_results(query: str, results: list[dict], mode: str) -> str:
    header = MOCK_HEADER if mode == "mock" else f"Web search results for {query!r} (source: Tavily)."
    if not results:
        return f"{header}\nNo results found."
    blocks = [header]
    for i, r in enumerate(results, start=1):
        snippet = (r.get("content") or "").strip()[:SNIPPET_CHARS]
        blocks.append(f"[{i}] {r.get('title') or '(no title)'}\nURL: {r.get('url', '')}\n{snippet}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# MCP tool: thin wrapper. The docstring is what the LLM reads.
# ---------------------------------------------------------------------------


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the public web for current information (news, recent events, public facts).

    Do NOT use this for Xinghe Tech's internal policies; use the knowledge base for those.
    Each result has a title, URL and snippet; cite the URLs you rely on.
    max_results: how many results to return (1-8, default 5).
    """
    query = validate(query, max_results)
    if get_settings().search_mode == "mock":
        return format_results(query, mock_results(query, max_results), "mock")
    return format_results(query, tavily_results(query, max_results), "tavily")


if __name__ == "__main__":
    logger.info("search server starting in %s mode", get_settings().search_mode)
    mcp.run()  # stdio
