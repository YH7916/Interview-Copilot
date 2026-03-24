"""Hybrid retrieval for interview knowledge search."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from copilot.knowledge.bm25 import BM25Retriever
from copilot.knowledge.index import get_chroma_collection
from copilot.knowledge.rerank import rerank
from copilot.knowledge.rewrite import rewrite_query

logger = logging.getLogger(__name__)
RRF_K = 60


class HybridRetriever:
    def __init__(self, collection: Any | None = None, bm25_retriever: BM25Retriever | None = None):
        self._collection = collection or get_chroma_collection()
        self._bm25 = bm25_retriever or BM25Retriever(self._collection)

    async def search(
        self,
        raw_query: str,
        top_k_retrieve: int = 10,
        top_n_rerank: int = 3,
    ) -> list[dict[str, Any]]:
        started_at = time.perf_counter()
        query = await rewrite_query(raw_query)
        bm25_items, vector_items = await self._recall_candidates(query, top_k_retrieve)
        merged = self._rrf_fuse(bm25_items, vector_items, k=RRF_K)
        if not merged:
            return []

        reranked = await rerank(query, [item["text"] for item in merged], top_n=top_n_rerank)
        results = self._build_final_results(query, merged, reranked)
        logger.info("Hybrid retrieval returned %d results in %.2fs", len(results), time.perf_counter() - started_at)
        return results

    async def _recall_candidates(
        self,
        query: str,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        bm25_task = asyncio.to_thread(self._bm25.search, query, top_k)
        vector_task = asyncio.to_thread(self._vector_search, query, top_k)
        bm25_raw, vector_raw = await asyncio.gather(bm25_task, vector_task, return_exceptions=True)

        bm25_items = _tag_source(bm25_raw, "bm25") if not isinstance(bm25_raw, Exception) else []
        vector_items = vector_raw if not isinstance(vector_raw, Exception) else []

        if isinstance(bm25_raw, Exception):
            logger.warning("BM25 recall failed: %s", bm25_raw)
        if isinstance(vector_raw, Exception):
            logger.warning("Vector recall failed: %s", vector_raw)
        return bm25_items, vector_items

    def _vector_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        raw = self._collection.query(query_texts=[query], n_results=top_k)
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        ids = (raw.get("ids") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        return [
            {
                "id": ids[index] if index < len(ids) else f"vec_{index}",
                "text": docs[index],
                "metadata": metas[index] if index < len(metas) else {},
                "vector_distance": float(distances[index]) if index < len(distances) else None,
                "retrieval_sources": {"vector"},
            }
            for index in range(len(docs))
        ]

    @staticmethod
    def _build_final_results(
        query: str,
        merged: list[dict[str, Any]],
        reranked: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        results = []
        for item in reranked:
            index = int(item.get("index", 0))
            if not (0 <= index < len(merged)):
                continue
            source = merged[index]
            results.append(
                {
                    "id": source.get("id"),
                    "text": item.get("text") or source.get("text", ""),
                    "metadata": source.get("metadata", {}),
                    "rerank_score": float(item.get("score", 0.0)),
                    "retrieval_sources": sorted(source.get("retrieval_sources", set())),
                    "query": query,
                }
            )
        return results

    @staticmethod
    def _rrf_fuse(
        results_a: list[dict[str, Any]],
        results_b: list[dict[str, Any]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        fused: dict[tuple[str, str], dict[str, Any]] = {}

        for results in (results_a, results_b):
            for rank, item in enumerate(results):
                key = (str(item.get("id", "")), (item.get("text") or "").strip())
                if key not in fused:
                    entry = dict(item)
                    entry["retrieval_sources"] = set(item.get("retrieval_sources", set()))
                    entry["rrf_score"] = 0.0
                    fused[key] = entry

                fused[key]["rrf_score"] += 1.0 / (k + rank)
                fused[key]["retrieval_sources"].update(item.get("retrieval_sources", set()))

        return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)


def _tag_source(items: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    for item in items:
        item["retrieval_sources"] = {source}
    return items
