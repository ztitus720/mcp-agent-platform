"""Pluggable text-embedding interface, shared by scripts/ingest.py and the knowledge server.

Today: any OpenAI-compatible endpoint (Bailian text-embedding-v4 by default).
Later: add a local provider in this file; nothing else in the project has to change.
"""

from collections.abc import Sequence
from functools import lru_cache

from core.settings import Settings, get_settings

# Bailian text-embedding-v4 accepts at most 10 texts per request.
OPENAI_COMPATIBLE_BATCH_SIZE = 10


class EmbeddingDimensionError(RuntimeError):
    """The provider returned vectors of a different size than EMBEDDING_DIM."""


class Embedder:
    """Wraps a LangChain Embeddings backend and guarantees the vector size."""

    def __init__(self, backend, provider: str, model: str, dim: int) -> None:
        self._backend = backend
        self.provider = provider
        self.model = model
        self.dim = dim

    @property
    def signature(self) -> str:
        """Identifies the vector space a stored vector belongs to."""
        return f"{self.provider}:{self.model}:{self.dim}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._backend.embed_documents(list(texts))
        self._check(vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vector = self._backend.embed_query(text)
        self._check([vector])
        return vector

    def _check(self, vectors: list[list[float]]) -> None:
        for vector in vectors:
            if len(vector) != self.dim:
                raise EmbeddingDimensionError(
                    f"{self.model} returned {len(vector)} dims, but EMBEDDING_DIM={self.dim}."
                )


def _openai_compatible(settings: Settings) -> Embedder:
    from langchain_openai import OpenAIEmbeddings

    backend = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.resolved_embedding_api_key,
        base_url=settings.resolved_embedding_base_url,
        dimensions=settings.embedding_dim,
        chunk_size=OPENAI_COMPATIBLE_BATCH_SIZE,   # texts per HTTP request
        check_embedding_ctx_length=False,          # send raw strings, not tiktoken ids
        max_retries=2,
        timeout=30,
    )
    return Embedder(backend, "openai_compatible", settings.embedding_model, settings.embedding_dim)


def _local(settings: Settings) -> Embedder:
    raise NotImplementedError(
        "EMBEDDING_PROVIDER=local is reserved but not implemented yet. "
        "Add a local backend in core/embeddings.py, or use openai_compatible."
    )


_PROVIDERS = {"openai_compatible": _openai_compatible, "local": _local}


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Build the embedder selected by EMBEDDING_PROVIDER (once per process)."""
    settings = get_settings()
    return _PROVIDERS[settings.embedding_provider](settings)
