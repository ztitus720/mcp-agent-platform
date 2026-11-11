"""Thin wrapper around ChromaDB, shared by scripts/ingest.py and the knowledge server.

Chroma only stores vectors and does nearest-neighbour search. Vectors always come
from core.embeddings, so documents and queries live in the same vector space.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from core.embeddings import Embedder
from core.settings import get_settings

SIGNATURE_KEY = "embedding_signature"


class IndexNotReadyError(RuntimeError):
    """The knowledge index is missing, empty, or built with a different embedding."""


def _client():
    settings = get_settings()
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),  # no phone-home on a flaky network
    )


def recreate_collection(embedder: Embedder):
    """Drop the collection if it exists, then create an empty one stamped with the signature."""
    settings = get_settings()
    client = _client()
    try:
        client.delete_collection(settings.chroma_collection)
    except NotFoundError:  # first run: nothing to delete
        pass
    return client.create_collection(
        name=settings.chroma_collection,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={SIGNATURE_KEY: embedder.signature},
    )


def open_collection(embedder: Embedder):
    """Open the existing collection; refuse to use it unless it matches the current embedding."""
    settings = get_settings()
    try:
        collection = _client().get_collection(settings.chroma_collection)
    except NotFoundError:  # only a genuinely missing index is reported as "not found";
        raise IndexNotReadyError(
            "Knowledge index not found. Build it with: python -m scripts.ingest"
        ) from None
    # any other error (e.g. the store is briefly locked by another process) propagates as-is

    stored = (collection.metadata or {}).get(SIGNATURE_KEY)
    if stored != embedder.signature:
        raise IndexNotReadyError(
            f"Index was built with {stored!r}, but the current embedding is "
            f"{embedder.signature!r}. Rebuild it with: python -m scripts.ingest"
        )
    if collection.count() == 0:
        raise IndexNotReadyError("Knowledge index is empty. Build it with: python -m scripts.ingest")
    return collection
