import logging
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

from copilot.rag.engine import get_chroma_collection

logger = logging.getLogger(__name__)


class BM25Retriever:
    """基于 BM25 的关键词检索器（中文使用 jieba 分词）。"""

    def __init__(self, collection: Any | None = None):
        self._collection = collection
        self._bm25: BM25Okapi | None = None
        self._documents: list[str] = []
        self._ids: list[str] = []
        self._metadatas: list[dict[str, Any]] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [tok.strip() for tok in jieba.lcut(text or "") if tok.strip()]

    def _ensure_index(self) -> None:
        if self._bm25 is not None:
            return

        collection = self._collection or get_chroma_collection()
        payload = collection.get(include=["documents", "metadatas"])
        self._documents = payload.get("documents") or []
        self._ids = payload.get("ids") or [f"bm25_{i}" for i in range(len(self._documents))]
        self._metadatas = payload.get("metadatas") or [{} for _ in self._documents]

        tokenized_corpus = [self._tokenize(doc) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else BM25Okapi([[]])

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        self._ensure_index()
        if not self._documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)

        results: list[dict[str, Any]] = []
        for idx in ranked_indices[: max(top_k, 1)]:
            results.append(
                {
                    "id": self._ids[idx] if idx < len(self._ids) else f"bm25_{idx}",
                    "text": self._documents[idx],
                    "metadata": self._metadatas[idx] if idx < len(self._metadatas) else {},
                    "bm25_score": float(scores[idx]),
                }
            )
        return results
