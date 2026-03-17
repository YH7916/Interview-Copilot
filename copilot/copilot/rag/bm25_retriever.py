"""BM25 关键词检索器 — 基于 jieba 中文分词 + BM25Okapi。"""

import logging
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

from copilot.rag.engine import get_chroma_collection

logger = logging.getLogger(__name__)


class BM25Retriever:
    """从 ChromaDB 全量文档构建 BM25 倒排索引，支持中文关键词检索。

    索引在首次 search() 时懒构建，之后缓存复用。
    """

    def __init__(self, collection: Any | None = None):
        self._collection = collection
        self._bm25: BM25Okapi | None = None
        self._docs: list[str] = []
        self._ids: list[str] = []
        self._metas: list[dict] = []

    # ── 公开接口 ────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        self._ensure_index()
        if not self._docs:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)

        return [
            {
                "id": self._ids[i] if i < len(self._ids) else f"bm25_{i}",
                "text": self._docs[i],
                "metadata": self._metas[i] if i < len(self._metas) else {},
                "bm25_score": float(scores[i]),
            }
            for i in ranked[:top_k]
        ]

    # ── 内部方法 ────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        """首次调用时从 ChromaDB 拉取全量文档并构建 BM25 索引。"""
        if self._bm25 is not None:
            return

        coll = self._collection or get_chroma_collection()
        data = coll.get(include=["documents", "metadatas"])

        self._docs = data.get("documents") or []
        self._ids = data.get("ids") or [f"bm25_{i}" for i in range(len(self._docs))]
        self._metas = data.get("metadatas") or [{} for _ in self._docs]

        corpus = [_tokenize(doc) for doc in self._docs]
        self._bm25 = BM25Okapi(corpus) if corpus else BM25Okapi([[]])


def _tokenize(text: str) -> list[str]:
    """jieba 精确模式分词，过滤空白。"""
    return [t for t in jieba.lcut(text or "") if t.strip()]
