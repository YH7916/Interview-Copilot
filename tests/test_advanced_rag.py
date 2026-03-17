"""单元测试：Advanced RAG 全链路（Query Rewrite → BM25 → Vector → RRF → Rerank）"""

from __future__ import annotations

import json
import threading

import pytest

from copilot.rag.bm25_retriever import BM25Retriever
from copilot.rag.hybrid_retriever import HybridRetriever
from copilot.rag.reranker import rerank
from copilot.rag.tools import SearchCompanyQuestionsTool, SearchConceptTool


# ── Query Rewriter ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_rewriter_success(monkeypatch):
    class _Completions:
        async def create(self, **kw):
            assert kw["model"] == "qwen-turbo"

            class _R:
                choices = [type("C", (), {"message": type("M", (), {"content": "字节跳动 面试题"})()})]

            return _R()

    class _Client:
        def __init__(self, *a, **kw):
            self.chat = type("Chat", (), {"completions": _Completions()})()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr("copilot.rag.query_rewriter.AsyncOpenAI", _Client)

    from copilot.rag.query_rewriter import rewrite_query

    assert await rewrite_query("查查字节的面经") == "字节跳动 面试题"


@pytest.mark.asyncio
async def test_query_rewriter_fallback(monkeypatch):
    class _Completions:
        async def create(self, **kw):
            raise RuntimeError("network error")

    class _Client:
        def __init__(self, *a, **kw):
            self.chat = type("Chat", (), {"completions": _Completions()})()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr("copilot.rag.query_rewriter.AsyncOpenAI", _Client)

    from copilot.rag.query_rewriter import rewrite_query

    raw = "查查字节的面经"
    assert await rewrite_query(raw) == raw


# ── BM25 Retriever ──────────────────────────────────────────────


def test_bm25_search_and_cache():
    class _Coll:
        def __init__(self):
            self.calls = 0

        def get(self, include=None):
            self.calls += 1
            return {
                "ids": ["1", "2", "3"],
                "documents": ["字节跳动 算法 面试题", "阿里 云计算 面试", "Transformer 原理"],
                "metadatas": [{"source": "a.md"}, {"source": "b.md"}, {"source": "c.md"}],
            }

    coll = _Coll()
    r = BM25Retriever(collection=coll)

    first = r.search("字节 算法", top_k=2)
    second = r.search("Transformer", top_k=1)

    assert len(first) == 2 and first[0]["id"] == "1" and "bm25_score" in first[0]
    assert len(second) == 1
    assert coll.calls == 1  # 索引只构建一次


# ── Reranker ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reranker_success(monkeypatch):
    class _Resp:
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
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def post(self, url, **kw):
            assert "text-rerank" in url
            return _Resp()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr("copilot.rag.reranker.httpx.AsyncClient", _Client)

    result = await rerank("query", ["doc1", "doc2", "doc3"], top_n=2)
    assert len(result) == 2 and result[0]["score"] == 0.95


@pytest.mark.asyncio
async def test_reranker_fallback(monkeypatch):
    class _Client:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def post(self, *a, **kw):
            raise RuntimeError("timeout")

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr("copilot.rag.reranker.httpx.AsyncClient", _Client)

    result = await rerank("q", ["d1", "d2", "d3"], top_n=2)
    assert result == [{"index": 0, "score": 0.0, "text": "d1"}, {"index": 1, "score": 0.0, "text": "d2"}]


# ── RRF Fusion ──────────────────────────────────────────────────


def test_rrf_fuse_scoring():
    k = 60
    bm25 = [
        {"id": "a", "text": "doc a", "retrieval_sources": {"bm25"}},
        {"id": "overlap", "text": "doc overlap", "retrieval_sources": {"bm25"}},
        {"id": "c", "text": "doc c", "retrieval_sources": {"bm25"}},
    ]
    vec = [
        {"id": "overlap", "text": "doc overlap", "retrieval_sources": {"vector"}},
        {"id": "v2", "text": "doc v2", "retrieval_sources": {"vector"}},
        {"id": "v3", "text": "doc v3", "retrieval_sources": {"vector"}},
    ]

    fused = HybridRetriever._rrf_fuse(bm25, vec, k=k)

    # 双路命中的 overlap 应排第一
    assert fused[0]["id"] == "overlap"
    assert fused[0]["rrf_score"] == pytest.approx(1 / (k + 1) + 1 / (k + 0))
    assert fused[0]["retrieval_sources"] == {"bm25", "vector"}
    # 分数降序
    assert [x["rrf_score"] for x in fused] == sorted([x["rrf_score"] for x in fused], reverse=True)


# ── Hybrid Retriever 集成 ───────────────────────────────────────


@pytest.mark.asyncio
async def test_hybrid_retriever_integration(monkeypatch):
    class _Coll:
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

    async def _rewrite(q):
        return "优化查询"

    async def _rerank(q, docs, top_n=3):
        assert q == "优化查询" and len(docs) == 3
        # RRF 排序后 "doc vec 2" 在第一位（双路命中）
        assert docs[0] == "doc vec 2"
        return [{"index": 0, "score": 0.99, "text": docs[0]}, {"index": 1, "score": 0.88, "text": docs[1]}]

    monkeypatch.setattr("copilot.rag.hybrid_retriever.rewrite_query", _rewrite)
    monkeypatch.setattr("copilot.rag.hybrid_retriever.rerank", _rerank)

    result = await HybridRetriever(collection=_Coll(), bm25_retriever=_BM25()).search(
        "原始query", top_k_retrieve=5, top_n_rerank=2
    )

    assert len(result) == 2
    assert result[0]["text"] == "doc vec 2"
    assert sorted(result[0]["retrieval_sources"]) == ["bm25", "vector"]


# ── Tools ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tools_use_hybrid_pipeline(monkeypatch):
    class _Mock:
        async def search(self, raw_query, top_k_retrieve=10, top_n_rerank=3):
            return [
                {
                    "id": "1",
                    "text": "面试内容",
                    "metadata": {"source": "sample.md"},
                    "rerank_score": 0.9,
                    "retrieval_sources": ["bm25", "vector"],
                    "query": raw_query,
                }
            ]

    monkeypatch.setattr("copilot.rag.tools._retriever", _Mock())

    assert "sample.md" in await SearchConceptTool().execute("RAG")
    assert "sample.md" in await SearchCompanyQuestionsTool().execute("字节跳动", "算法工程师")


# ── Config ──────────────────────────────────────────────────────


def test_config_env_priority(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    from copilot.config import get_dashscope_api_key

    assert get_dashscope_api_key() == "env-key"


def test_config_file_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    cfg_dir = tmp_path / ".nanobot"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(
        json.dumps({"providers": {"dashscope": {"apiKey": "file-key"}}}), encoding="utf-8"
    )
    monkeypatch.setattr("copilot.config.Path.home", lambda: tmp_path)
    from copilot.config import get_dashscope_api_key

    assert get_dashscope_api_key() == "file-key"


def test_config_returns_none(monkeypatch, tmp_path):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr("copilot.config.Path.home", lambda: tmp_path)
    from copilot.config import get_dashscope_api_key

    assert get_dashscope_api_key() is None


# ── Recursive Chunking ──────────────────────────────────────────


def test_recursive_chunk_basic():
    from copilot.rag.engine import _recursive_chunk

    text = "## 第一节\n\n" + "内容" * 100 + "\n\n## 第二节\n\n" + "内容" * 100
    chunks = _recursive_chunk(text, chunk_size=300, overlap=50)
    assert len(chunks) >= 2
    assert all(len(c) <= 400 for c in chunks)


def test_recursive_chunk_overlap():
    from copilot.rag.engine import _recursive_chunk

    text = "\n\n".join(f"第{i}段技术内容，包含知识点。" for i in range(30))
    chunks = _recursive_chunk(text, chunk_size=200, overlap=50)
    assert len(chunks) >= 2
    assert all(c for c in chunks)


# ── Thread Safety ───────────────────────────────────────────────


def test_weakness_tracker_thread_safety(tmp_path):
    from copilot.memory.weakness_tracker import WeaknessTracker

    tracker = WeaknessTracker(log_path=tmp_path / "log.md")
    n = 10
    threads = [threading.Thread(target=tracker._append, args=(f"entry-{i}",)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    content = tracker.log_path.read_text(encoding="utf-8")
    for i in range(n):
        assert f"entry-{i}" in content
