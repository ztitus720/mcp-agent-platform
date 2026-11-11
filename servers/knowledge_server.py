"""Server-A: knowledge-base tools (semantic search, list documents, read a document).

Runs as an independent MCP server process over stdio:
    python -m servers.knowledge_server
Build the index first with: python -m scripts.ingest
Never print() to stdout in this file: stdout is the MCP wire. Log to stderr instead.
"""

import logging
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from core.embeddings import get_embedder
from core.settings import PROJECT_ROOT
from core.vector_store import IndexNotReadyError, open_collection

logger = logging.getLogger(__name__)

mcp = MCPServer(
    "knowledge",
    instructions=(
        "Internal knowledge base of Xinghe Tech: leave, expense, IT security "
        "and onboarding policies."
    ),
)

DOCS_DIR = (PROJECT_ROOT / "data" / "docs").resolve()  # same folder scripts/ingest.py reads
MAX_TOP_K = 8
MIN_SIMILARITY = 0.40        # passages below this are treated as irrelevant; calibrated in stage 2
MAX_DOCUMENT_CHARS = 20_000

# ---------------------------------------------------------------------------
# Plain logic (testable without MCP)
# ---------------------------------------------------------------------------


def resolve_doc_path(filename: str) -> Path:
    """Map a caller-supplied file name to a .md file directly inside DOCS_DIR, or raise ToolError."""
    candidate = (DOCS_DIR / filename.strip()).resolve()
    if candidate.parent != DOCS_DIR or candidate.suffix != ".md" or not candidate.is_file():
        available = ", ".join(sorted(p.name for p in DOCS_DIR.glob("*.md")))
        raise ToolError(f"Unknown document {filename!r}. Available documents: {available}")
    return candidate


def doc_title(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def search(query: str, top_k: int) -> list[dict]:
    """Embed the query and return the top_k nearest chunks with a similarity score."""
    query = query.strip()
    if not query:
        raise ToolError("Query is empty.")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ToolError(f"top_k must be between 1 and {MAX_TOP_K}.")
    try:
        embedder = get_embedder()
        collection = open_collection(embedder)          # checks the embedding signature
        raw = collection.query(
            query_embeddings=[embedder.embed_query(query)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except IndexNotReadyError as exc:
        raise ToolError(str(exc)) from None
    except Exception as exc:  # network, quota, Chroma errors: report, never crash the server
        logger.exception("knowledge search failed")
        raise ToolError(
            f"Knowledge search is temporarily unavailable ({type(exc).__name__}). Try again later."
        ) from None

    return [
        {"source": meta["source"], "section": meta["section"],
         "similarity": round(1 - distance, 3), "text": text}
        for text, meta, distance in zip(
            raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        )
    ]


def format_hits(query: str, hits: list[dict]) -> str:
    """Turn hits into cited passages, or an explicit 'not found' that tells the model not to guess."""
    relevant = [h for h in hits if h["similarity"] >= MIN_SIMILARITY]
    if not relevant:
        best = max((h["similarity"] for h in hits), default=0.0)
        return (
            f"No relevant passages found in the knowledge base for {query!r} "
            f"(best similarity {best:.2f} < {MIN_SIMILARITY}). "
            "Do not guess: tell the user the knowledge base does not cover this."
        )
    return "\n\n".join(
        f"[{i}] {h['source']} # {h['section']} (similarity {h['similarity']:.2f})\n{h['text']}"
        for i, h in enumerate(relevant, start=1)
    )


# ---------------------------------------------------------------------------
# MCP tools: thin wrappers. The docstring is what the LLM reads.
# ---------------------------------------------------------------------------


@mcp.tool()
def search_knowledge(query: str, top_k: int = 3) -> str:
    """Search Xinghe Tech's internal policy documents and return the most relevant passages.

    Use this for ANY question about the company's rules: leave, expenses and
    reimbursement, IT security, onboarding. Each passage is labelled with its source
    file and section; cite them in your answer. If nothing relevant is found, say so
    instead of guessing. top_k: how many passages to return (1-8, default 3).
    """
    return format_hits(query, search(query, top_k))


@mcp.tool()
def list_documents() -> str:
    """List every document in the knowledge base as 'file name: title'."""
    paths = sorted(DOCS_DIR.glob("*.md"))
    if not paths:
        raise ToolError("The knowledge base folder is empty.")
    return "\n".join(f"{p.name}: {doc_title(p)}" for p in paths)


@mcp.tool()
def read_document(filename: str) -> str:
    """Return the full text of one knowledge-base document, e.g. "leave_policy.md".

    Use this when search passages are not enough and the whole document is needed.
    Call list_documents first if you do not know the exact file name.
    """
    text = resolve_doc_path(filename).read_text(encoding="utf-8")
    if len(text) > MAX_DOCUMENT_CHARS:
        text = text[:MAX_DOCUMENT_CHARS] + "\n...[truncated]"
    return text


if __name__ == "__main__":
    mcp.run()  # stdio
