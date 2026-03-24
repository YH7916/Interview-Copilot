"""BM25 retrieval over the local interview knowledge base."""

from __future__ import annotations

import re
from typing import Any

from copilot.knowledge.index import get_chroma_collection

try:
    import jieba
except ImportError:  # pragma: no cover
    jieba = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    class BM25Okapi:  # type: ignore[override]
        def __init__(self, corpus: list[list[str]]):
            self.corpus = corpus

        def get_scores(self, query_tokens: list[str]) -> list[float]:
            return [sum(float(document.count(token)) for token in query_tokens) for document in self.corpus]


class BM25Retriever:
    def __init__(self, collection: Any | None = None):
        self._collection = collection
        self._bm25: BM25Okapi | None = None
        self._docs: list[str] = []
        self._ids: list[str] = []
        self._metas: list[dict] = []

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        self._ensure_index()
        if not self._docs:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)
        return [
            {
                "id": self._ids[index] if index < len(self._ids) else f"bm25_{index}",
                "text": self._docs[index],
                "metadata": self._metas[index] if index < len(self._metas) else {},
                "bm25_score": float(scores[index]),
            }
            for index in ranked[:top_k]
        ]

    def _ensure_index(self) -> None:
        if self._bm25 is not None:
            return

        collection = self._collection or get_chroma_collection()
        data = collection.get(include=["documents", "metadatas"])
        self._docs = data.get("documents") or []
        self._ids = data.get("ids") or [f"bm25_{index}" for index in range(len(self._docs))]
        self._metas = data.get("metadatas") or [{} for _ in self._docs]

        corpus = [_tokenize(document) for document in self._docs]
        self._bm25 = BM25Okapi(corpus) if corpus else BM25Okapi([[]])


def _tokenize(text: str) -> list[str]:
    if jieba is not None:
        return [token for token in jieba.lcut(text or "") if token.strip()]
    return re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", text or "")
