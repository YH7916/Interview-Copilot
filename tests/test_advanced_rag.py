from __future__ import annotations

import pytest

from copilot.rag.bm25_retriever import BM25Retriever
from copilot.rag.hybrid_retriever import HybridRetriever
from copilot.rag.reranker import rerank
from copilot.rag.tools import SearchCompanyQuestionsTool, SearchConceptTool


@pytest.mark.asyncio
async def test_query_rewriter_success(monkeypatch):
    class _MockCompletions:
        async def create(self, **kwargs):
            assert kwargs["model"] == "qwen-turbo"

            class _Msg:
                content = "字节跳动 面试经验 面试题 技术面试"

            class _Choice:
                message = _Msg()

            class _Resp:
                choices = [_Choice()]

            return _Resp()

    class _MockClient:
        def __init__(self, *args, **kwargs):
            self.chat = type("Chat", (), {"completions": _MockCompletions()})()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr("copilot.rag.query_rewriter.AsyncOpenAI", _MockClient)

    from copilot.rag.query_rewriter import rewrite_query

    result = await rewrite_query("查查字节的面经")
    assert result == "字节跳动 面试经验 面试题 技术面试"


@pytest.mark.asyncio
async def test_query_rewriter_fallback(monkeypatch):
    class _MockCompletions:
        async def create(self, **kwargs):
            raise RuntimeError("network error")

    class _MockClient:
        def __init__(self, *args, **kwargs):
            self.chat = type("Chat", (), {"completions": _MockCompletions()})()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr("copilot.rag.query_rewriter.AsyncOpenAI", _MockClient)

    from copilot.rag.query_rewriter import rewrite_query

    raw = "查查字节的面经"
    result = await rewrite_query(raw)
    assert result == raw


def test_bm25_retriever_search_and_cache():
    class _Collection:
        def __init__(self):
            self.get_called = 0

        def get(self, include=None):
            self.get_called += 1
            return {
                "ids": ["1", "2", "3"],
                "documents": [
                    "字节跳动 算法 面试题",
                    "阿里 云计算 面试经验",
                    "深度学习 Transformer 原理",
                ],
                "metadatas": [
                    {"source": "a.md"},
                    {"source": "b.md"},
                    {"source": "c.md"},
                ],
            }

    coll = _Collection()
    retriever = BM25Retriever(collection=coll)

    first = retriever.search("字节 算法 面试", top_k=2)
    second = retriever.search("Transformer", top_k=1)

    assert len(first) == 2
    assert first[0]["id"] == "1"
    assert "bm25_score" in first[0]
    assert len(second) == 1
    assert coll.get_called == 1


@pytest.mark.asyncio
async def test_reranker_success(monkeypatch):
    class _MockResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "output": {
                    "results": [
                        {
                            "index": 1,
                            "relevance_score": 0.95,
                            "document": {"text": "doc2"},
                        },
                        {
                            "index": 0,
                            "relevance_score": 0.78,
                            "document": {"text": "doc1"},
                        },
                    ]
                }
            }

    class _MockClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            assert "text-rerank" in url
            assert json["model"] == "gte-rerank"
            return _MockResponse()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr("copilot.rag.reranker.httpx.AsyncClient", _MockClient)

    docs = ["doc1", "doc2", "doc3"]
    result = await rerank("query", docs, top_n=2)

    assert len(result) == 2
    assert result[0]["index"] == 1
    assert result[0]["score"] == 0.95
    assert result[0]["text"] == "doc2"


@pytest.mark.asyncio
async def test_reranker_fallback(monkeypatch):
    class _MockClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            raise RuntimeError("timeout")

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr("copilot.rag.reranker.httpx.AsyncClient", _MockClient)

    docs = ["doc1", "doc2", "doc3"]
    result = await rerank("query", docs, top_n=2)

    assert result == [
        {"index": 0, "score": 0.0, "text": "doc1"},
        {"index": 1, "score": 0.0, "text": "doc2"},
    ]


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

    class _MockBM25:
        def search(self, query, top_k=10):
            return [
                {"id": "b1", "text": "doc bm25 1", "metadata": {"source": "b1.md"}, "bm25_score": 5.0},
                {"id": "v2", "text": "doc vec 2", "metadata": {"source": "v2.md"}, "bm25_score": 4.0},
            ]

    async def _mock_rewrite(q):
        return "优化查询"

    async def _mock_rerank(q, docs, top_n=3):
        assert q == "优化查询"
        assert len(docs) == 3
        return [{"index": 1, "score": 0.99, "text": docs[1]}, {"index": 2, "score": 0.88, "text": docs[2]}]

    monkeypatch.setattr("copilot.rag.hybrid_retriever.rewrite_query", _mock_rewrite)
    monkeypatch.setattr("copilot.rag.hybrid_retriever.rerank", _mock_rerank)

    retriever = HybridRetriever(collection=_Collection(), bm25_retriever=_MockBM25())
    result = await retriever.search("原始query", top_k_retrieve=5, top_n_rerank=2)

    assert len(result) == 2
    assert result[0]["text"] == "doc vec 2"
    assert result[0]["query"] == "优化查询"
    assert sorted(result[0]["retrieval_sources"]) == ["bm25", "vector"]


@pytest.mark.asyncio
async def test_tools_use_hybrid_pipeline(monkeypatch):
    class _MockRetriever:
        async def search(self, raw_query, top_k_retrieve=10, top_n_rerank=3):
            return [
                {
                    "id": "1",
                    "text": "这是检索到的面试内容",
                    "metadata": {"source": "sample.md"},
                    "rerank_score": 0.9,
                    "retrieval_sources": ["bm25", "vector"],
                    "query": raw_query,
                }
            ]

    monkeypatch.setattr("copilot.rag.tools._hybrid_retriever", _MockRetriever())

    concept_tool = SearchConceptTool()
    concept_result = await concept_tool.execute("RAG")
    assert "sample.md" in concept_result
    assert "RAG" in concept_result

    company_tool = SearchCompanyQuestionsTool()
    company_result = await company_tool.execute("字节跳动", "算法工程师")
    assert "sample.md" in company_result
    assert "字节跳动" in company_result
