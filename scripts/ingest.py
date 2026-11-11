"""Build the knowledge index: read data/docs/*.md, chunk by section, embed, store in Chroma.

Run from the project root:
    python -m scripts.ingest
Every run rebuilds the index from scratch, so it always matches data/docs exactly.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from core.embeddings import get_embedder
from core.settings import PROJECT_ROOT
from core.vector_store import recreate_collection

DOCS_DIR = PROJECT_ROOT / "data" / "docs"
MAX_CHUNK_CHARS = 600


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def split_long(body: str, limit: int) -> list[str]:
    """Split an over-long section on paragraph boundaries so each part stays near the limit."""
    parts: list[str] = []
    current = ""
    for paragraph in (p.strip() for p in body.split("\n")):
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) + 1 > limit:
            parts.append(current)
            current = paragraph
        else:
            current = f"{current}\n{paragraph}" if current else paragraph
    if current:
        parts.append(current)
    return parts


def chunk_markdown(path: Path) -> list[Chunk]:
    """One chunk per '## ' section; long sections are split, every part keeps its heading."""
    lines = path.read_text(encoding="utf-8").splitlines()
    doc_title = next((line[2:].strip() for line in lines if line.startswith("# ")), path.stem)

    sections: list[tuple[str, list[str]]] = []
    for line in lines:
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif sections:
            sections[-1][1].append(line)
    # Text before the first "## " (document title, disclaimer) is intentionally skipped.

    chunks: list[Chunk] = []
    for s_index, (heading, body_lines) in enumerate(sections):
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        for p_index, part in enumerate(split_long(body, MAX_CHUNK_CHARS)):
            chunks.append(Chunk(
                id=f"{path.stem}::{s_index}::{p_index}",
                text=f"{doc_title} > {heading}\n{part}",
                metadata={"source": path.name, "doc_title": doc_title,
                          "section": heading, "part": p_index},
            ))
    return chunks


def main() -> int:
    files = sorted(DOCS_DIR.glob("*.md"))
    if not files:
        print(f"No .md files found in {DOCS_DIR}", file=sys.stderr)
        return 1
    chunks = [chunk for path in files for chunk in chunk_markdown(path)]
    print(f"Read {len(files)} files -> {len(chunks)} chunks")

    # Embed everything BEFORE touching the old index: if this fails, the old index survives.
    embedder = get_embedder()
    start = time.perf_counter()
    vectors = embedder.embed_documents([c.text for c in chunks])
    print(f"Embedded with {embedder.signature} in {time.perf_counter() - start:.1f}s")

    collection = recreate_collection(embedder)
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
        embeddings=vectors,
    )
    print(f"Stored {collection.count()} chunks in collection '{collection.name}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
