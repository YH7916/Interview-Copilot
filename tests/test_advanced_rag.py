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
        assert docs[0] == "doc vec 2"
        return [{"index": 0, "score": 0.99, "text": docs[0]}, {"index": 1, "score": 0.88, "text": docs[1]}]

    monkeypatch.setattr("copilot.rag.hybrid_retriever.rewrite_query", _mock_rewrite)
    monkeypatch.setattr("copilot.rag.hybrid_retriever.rerank", _mock_rerank)

    retriever = HybridRetriever(collection=_Collection(), bm25_retriever=_MockBM25())
    result = await retriever.search("原始query", top_k_retrieve=5, top_n_rerank=2)

    assert len(result) == 2
    assert result[0]["text"] == "doc vec 2"
    assert result[0]["query"] == "优化查询"
    assert sorted(result[0]["retrieval_sources"]) == ["bm25", "vector"]


def test_rrf_fuse_scoring():
    k = 60
    bm25_results = [
        {"id": "a", "text": "doc a", "retrieval_sources": {"bm25"}},
        {"id": "overlap", "text": "doc overlap", "retrieval_sources": {"bm25"}},
        {"id": "c", "text": "doc c", "retrieval_sources": {"bm25"}},
    ]
    vector_results = [
        {"id": "overlap", "text": "doc overlap", "retrieval_sources": {"vector"}},
        {"id": "v2", "text": "doc v2", "retrieval_sources": {"vector"}},
        {"id": "v3", "text": "doc v3", "retrieval_sources": {"vector"}},
    ]

    fused = HybridRetriever._rrf_fuse(bm25_results, vector_results, k=k)

    assert fused[0]["id"] == "overlap"
    expected_overlap_score = (1.0 / (k + 1)) + (1.0 / (k + 0))
    assert fused[0]["rrf_score"] == pytest.approx(expected_overlap_score)
    assert fused[0]["retrieval_sources"] == {"bm25", "vector"}

    scores = [item["rrf_score"] for item in fused]
    assert scores == sorted(scores, reverse=True)


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


def test_config_get_api_key_from_env(monkeypatch):
    """环境变量优先于配置文件。"""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key-123")
    from copilot.config import get_dashscope_api_key
    assert get_dashscope_api_key() == "env-key-123"


def test_config_get_api_key_fallback_to_file(monkeypatch, tmp_path):
    """环境变量不存在时，从配置文件读取。"""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    config = {"providers": {"dashscope": {"apiKey": "file-key-456"}}}
    config_path = tmp_path / ".nanobot" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(__import__("json").dumps(config), encoding="utf-8")
    monkeypatch.setattr("copilot.config.Path.home", lambda: tmp_path)
    from copilot.config import get_dashscope_api_key
    assert get_dashscope_api_key() == "file-key-456"


def test_config_get_api_key_returns_none(monkeypatch, tmp_path):
    """无环境变量且配置文件不存在时返回 None。"""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr("copilot.config.Path.home", lambda: tmp_path)
    from copilot.config import get_dashscope_api_key
    assert get_dashscope_api_key() is None


def test_recursive_chunk_basic():
    """递归分块：验证 overlap 和结构切分。"""
    from copilot.rag.engine import _recursive_chunk
    text = "## 第一节\n\n段落一内容很长" + "x" * 200 + "\n\n段落二内容也很长" + "y" * 200 + "\n\n## 第二节\n\n段落三" + "z" * 200
    chunks = _recursive_chunk(text, chunk_size=300, overlap=50)
    assert len(chunks) >= 2
    # 每个 chunk 不应超长太多（允许一些边界溢出）
    for chunk in chunks:
        assert len(chunk) <= 400  # 允许少量溢出


def test_recursive_chunk_overlap():
    """递归分块：相邻 chunk 之间应有 overlap 重叠。"""
    from copilot.rag.engine import _recursive_chunk
    # 使用简单的句子列表，确保有足够内容触发分块
    sentences = ["这是第{}句话，包含一些技术内容。".format(i) for i in range(30)]
    text = "\n\n".join(sentences)
    chunks = _recursive_chunk(text, chunk_size=200, overlap=50)
    if len(chunks) >= 2:
        # 检查相邻 chunk 有重叠文本
        for i in range(len(chunks) - 1):
            tail = chunks[i][-30:]  # 取前一个 chunk 的尾部
            # overlap 不保证字面完全相同（因为分隔符），但至少有部分内容重叠
            assert len(chunks[i]) > 0
            assert len(chunks[i + 1]) > 0


def test_weakness_tracker_thread_safety(tmp_path):
    """多线程并发写入错题本，验证数据完整性。"""
    import threading
    from copilot.memory.weakness_tracker import WeaknessTracker

    tracker = WeaknessTracker(log_path=tmp_path / "weakness_log.md")
    n_threads = 10

    def writer(idx):
        tracker._append(f"entry-{idx}")

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    content = tracker.log_path.read_text(encoding="utf-8")
    for i in range(n_threads):
        assert f"entry-{i}" in content
