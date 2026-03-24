from __future__ import annotations

import importlib
import threading
import tempfile
from pathlib import Path

import pytest

from copilot.knowledge.bm25 import BM25Retriever
from copilot.knowledge.retrieval import HybridRetriever
from copilot.knowledge.rerank import rerank
from copilot.knowledge.tools import SearchCompanyQuestionsTool, SearchConceptTool

def _make_log_path() -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="copilot-test-", suffix=".md", delete=False)
    handle.close()
    return Path(handle.name)


@pytest.mark.asyncio
async def test_query_rewriter_success(monkeypatch):
    class _Completions:
        async def create(self, **kwargs):
            assert kwargs["model"] == "rewrite-model"

            class _Response:
                choices = [
                    type("Choice", (), {"message": type("Msg", (), {"content": "agent rag interview"})()})
                ]

            return _Response()

    class _Client:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": _Completions()})()

    monkeypatch.setattr(
        "copilot.knowledge.rewrite.get_text_settings",
        lambda task="rewrite": {"api_key": "key", "model": "rewrite-model"},
    )
    monkeypatch.setattr(
        "copilot.knowledge.rewrite.create_async_client",
        lambda task="rewrite": (_Client(), {"model": "rewrite-model"}),
    )

    from copilot.knowledge.rewrite import rewrite_query

    assert await rewrite_query("raw query") == "agent rag interview"


@pytest.mark.asyncio
async def test_query_rewriter_fallback(monkeypatch):
    class _Completions:
        async def create(self, **kwargs):
            raise RuntimeError("network error")

    class _Client:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": _Completions()})()

    monkeypatch.setattr(
        "copilot.knowledge.rewrite.get_text_settings",
        lambda task="rewrite": {"api_key": "key", "model": "rewrite-model"},
    )
    monkeypatch.setattr(
        "copilot.knowledge.rewrite.create_async_client",
        lambda task="rewrite": (_Client(), {"model": "rewrite-model"}),
    )

    from copilot.knowledge.rewrite import rewrite_query

    assert await rewrite_query("raw query") == "raw query"


def test_bm25_search_and_cache():
    class _Collection:
        def __init__(self):
            self.calls = 0

        def get(self, include=None):
            self.calls += 1
            return {
                "ids": ["1", "2", "3"],
                "documents": ["agent loop design", "rag system design", "database sharding"],
                "metadatas": [{"source": "a.md"}, {"source": "b.md"}, {"source": "c.md"}],
            }

    collection = _Collection()
    retriever = BM25Retriever(collection=collection)

    first = retriever.search("agent design", top_k=2)
    second = retriever.search("database", top_k=1)

    assert len(first) == 2
    assert first[0]["id"] == "1"
    assert "bm25_score" in first[0]
    assert len(second) == 1
    assert collection.calls == 1


@pytest.mark.asyncio
async def test_reranker_success(monkeypatch):
    rerank_module = importlib.import_module("copilot.knowledge.rerank")

    class _Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.95, "document": {"text": "doc2"}},
                        {"index": 0, "relevance_score": 0.78, "document": {"text": "doc1"}},
                    ]
                }
            }

    class _Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            assert url == "https://rerank.test"
            return _Response()

    monkeypatch.setattr(
        rerank_module,
        "get_rerank_settings",
        lambda provider=None: {
            "provider": "dashscope",
            "api_key": "key",
            "model": "rerank-model",
            "url": "https://rerank.test",
        },
    )
    monkeypatch.setattr(rerank_module.httpx, "AsyncClient", _Client)

    result = await rerank("query", ["doc1", "doc2", "doc3"], top_n=2)
    assert len(result) == 2
    assert result[0]["score"] == 0.95


@pytest.mark.asyncio
async def test_reranker_fallback(monkeypatch):
    rerank_module = importlib.import_module("copilot.knowledge.rerank")

    class _Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            raise RuntimeError("timeout")

    monkeypatch.setattr(
        rerank_module,
        "get_rerank_settings",
        lambda provider=None: {
            "provider": "dashscope",
            "api_key": "key",
            "model": "rerank-model",
            "url": "https://rerank.test",
        },
    )
    monkeypatch.setattr(rerank_module.httpx, "AsyncClient", _Client)

    result = await rerank("query", ["doc1", "doc2", "doc3"], top_n=2)
    assert result == [
        {"index": 0, "score": 0.0, "text": "doc1"},
        {"index": 1, "score": 0.0, "text": "doc2"},
    ]


def test_rrf_fuse_scoring():
    k = 60
    bm25 = [
        {"id": "a", "text": "doc a", "retrieval_sources": {"bm25"}},
        {"id": "overlap", "text": "doc overlap", "retrieval_sources": {"bm25"}},
    ]
    vector = [
        {"id": "overlap", "text": "doc overlap", "retrieval_sources": {"vector"}},
        {"id": "b", "text": "doc b", "retrieval_sources": {"vector"}},
    ]

    fused = HybridRetriever._rrf_fuse(bm25, vector, k=k)

    assert fused[0]["id"] == "overlap"
    assert fused[0]["rrf_score"] == pytest.approx(1 / (k + 1) + 1 / (k + 0))
    assert fused[0]["retrieval_sources"] == {"bm25", "vector"}


@pytest.mark.asyncio
async def test_hybrid_retriever_integration(monkeypatch):
    class _Collection:
        def query(self, query_texts, n_results):
            return {
                "ids": [["v1", "v2"]],
                "documents": [["doc vec 1", "doc vec 2"]],
                "metadatas": [[{"source": "v1.md"}, {"source": "v2.md"}]],
                "distances": [[0.1, 0.3]],
            }

    class _BM25:
        def search(self, query, top_k=10):
            return [
                {"id": "b1", "text": "doc bm25 1", "metadata": {"source": "b1.md"}, "bm25_score": 5.0},
                {"id": "v2", "text": "doc vec 2", "metadata": {"source": "v2.md"}, "bm25_score": 4.0},
            ]

    async def _rewrite(query):
        return "rewritten query"

    async def _rerank(query, documents, top_n=3):
        assert query == "rewritten query"
        assert documents[0] == "doc vec 2"
        return [{"index": 0, "score": 0.99, "text": documents[0]}]

    monkeypatch.setattr("copilot.knowledge.retrieval.rewrite_query", _rewrite)
    monkeypatch.setattr("copilot.knowledge.retrieval.rerank", _rerank)

    result = await HybridRetriever(collection=_Collection(), bm25_retriever=_BM25()).search(
        "raw query",
        top_k_retrieve=5,
        top_n_rerank=1,
    )

    assert len(result) == 1
    assert result[0]["text"] == "doc vec 2"
    assert sorted(result[0]["retrieval_sources"]) == ["bm25", "vector"]


@pytest.mark.asyncio
async def test_tools_use_hybrid_pipeline(monkeypatch):
    class _Retriever:
        async def search(self, raw_query, top_k_retrieve=10, top_n_rerank=3):
            return [
                {
                    "id": "1",
                    "text": "interview content",
                    "metadata": {"source": "sample.md"},
                    "rerank_score": 0.9,
                    "retrieval_sources": ["bm25", "vector"],
                    "query": raw_query,
                }
            ]

    monkeypatch.setattr("copilot.knowledge.tools._retriever", _Retriever())

    assert "sample.md" in await SearchConceptTool().execute("RAG")
    assert "sample.md" in await SearchCompanyQuestionsTool().execute("ByteDance", "MLE")


def test_recursive_chunk_basic():
    from copilot.knowledge.index import _recursive_chunk

    text = "## Section 1\n\n" + "content " * 100 + "\n\n## Section 2\n\n" + "content " * 100
    chunks = _recursive_chunk(text, chunk_size=300, overlap=50)

    assert len(chunks) >= 2
    assert all(len(chunk) <= 400 for chunk in chunks)


def test_weakness_tracker_thread_safety():
    from copilot.profile.store import WeaknessTracker

    tracker = WeaknessTracker(log_path=_make_log_path())
    threads = [threading.Thread(target=tracker._append, args=(f"entry-{index}",)) for index in range(10)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    content = tracker.log_path.read_text(encoding="utf-8")
    for index in range(10):
        assert f"entry-{index}" in content
