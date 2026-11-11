"""Stage-5 regression tests: knowledge store errors are retried or reported precisely (no network).

    python -m pytest tests/test_knowledge_retry.py -v
"""

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import servers.knowledge_server as ks
from core.vector_store import IndexNotReadyError

FAKE_RESULT = {"documents": [["text"]], "metadatas": [[{"source": "a.md", "section": "1"}]],
               "distances": [[0.2]]}


class FakeEmbedder:
    def embed_query(self, text):
        return [0.0, 1.0, 0.0]


class FakeCollection:
    def query(self, **kwargs):
        return FAKE_RESULT


@pytest.fixture(autouse=True)
def no_network_no_sleep(monkeypatch):
    monkeypatch.setattr(ks, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(ks, "STORE_BACKOFF_SECONDS", 0)


def test_transient_store_error_is_retried(monkeypatch):
    calls = []

    def flaky_open(embedder):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("database is locked")
        return FakeCollection()

    monkeypatch.setattr(ks, "open_collection", flaky_open)
    hits = ks.search("year leave", 3)
    assert len(calls) == 3 and hits[0]["source"] == "a.md"


def test_persistent_store_error_reports_real_type(monkeypatch):
    def always_locked(embedder):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(ks, "open_collection", always_locked)
    with pytest.raises(ToolError, match=r"temporarily unavailable \(RuntimeError\)"):
        ks.search("year leave", 3)


def test_missing_index_is_not_retried(monkeypatch):
    calls = []

    def missing(embedder):
        calls.append(1)
        raise IndexNotReadyError("Knowledge index not found. Build it with: python -m scripts.ingest")

    monkeypatch.setattr(ks, "open_collection", missing)
    with pytest.raises(ToolError, match="index not found"):
        ks.search("year leave", 3)
    assert len(calls) == 1
