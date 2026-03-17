"""混合检索流水线 — 串联 Query Rewrite → BM25 + Vector → RRF → Rerank。"""

import asyncio
import logging
import time
from typing import Any

from copilot.rag.bm25_retriever import BM25Retriever
from copilot.rag.engine import get_chroma_collection
from copilot.rag.query_rewriter import rewrite_query
from copilot.rag.reranker import rerank

logger = logging.getLogger(__name__)


class HybridRetriever:
    """四阶段检索: 改写 → 双路召回 → RRF 融合 → 重排序。"""

    def __init__(self, collection: Any | None = None, bm25_retriever: BM25Retriever | None = None):
        self._collection = collection or get_chroma_collection()
        self._bm25 = bm25_retriever or BM25Retriever(self._collection)

    # ── 公开接口 ────────────────────────────────────────────────

    async def search(
        self, raw_query: str, top_k_retrieve: int = 10, top_n_rerank: int = 3
    ) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        logger.info("search | query=%s", raw_query[:80])

        # ① Query Rewrite
        rewritten = await rewrite_query(raw_query)
        logger.info("rewrite | %s → %s", raw_query[:40], rewritten[:40])

        # ② 并行双路召回
        bm25_task = asyncio.to_thread(self._bm25.search, rewritten, top_k_retrieve)
        vec_task = asyncio.to_thread(self._vector_search, rewritten, top_k_retrieve)
        bm25_raw, vec_raw = await asyncio.gather(bm25_task, vec_task, return_exceptions=True)

        bm25_items = _tag_source(bm25_raw, "bm25") if not isinstance(bm25_raw, Exception) else []
        vec_items = vec_raw if not isinstance(vec_raw, Exception) else []

        if isinstance(bm25_raw, Exception):
            logger.warning("BM25 failed: %s", bm25_raw)
        if isinstance(vec_raw, Exception):
            logger.warning("Vector failed: %s", vec_raw)

        logger.info("recall | bm25=%d vector=%d", len(bm25_items), len(vec_items))

        # ③ RRF 融合
        merged = self._rrf_fuse(bm25_items, vec_items, k=60)
        if not merged:
            return []

        # ④ Rerank
        reranked = await rerank(rewritten, [m["text"] for m in merged], top_n=top_n_rerank)

        final = []
        for item in reranked:
            idx = int(item.get("index", 0))
            if not (0 <= idx < len(merged)):
                continue
            src = merged[idx]
            final.append(
                {
                    "id": src.get("id"),
                    "text": item.get("text") or src.get("text", ""),
                    "metadata": src.get("metadata", {}),
                    "rerank_score": float(item.get("score", 0.0)),
                    "retrieval_sources": sorted(src.get("retrieval_sources", set())),
                    "query": rewritten,
                }
            )

        logger.info("done | results=%d elapsed=%.2fs", len(final), time.perf_counter() - t0)
        return final

    # ── 向量检索 ────────────────────────────────────────────────

    def _vector_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        raw = self._collection.query(query_texts=[query], n_results=top_k)
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        ids = (raw.get("ids") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        return [
            {
                "id": ids[i] if i < len(ids) else f"vec_{i}",
                "text": docs[i],
                "metadata": metas[i] if i < len(metas) else {},
                "vector_distance": float(dists[i]) if i < len(dists) else None,
                "retrieval_sources": {"vector"},
            }
            for i in range(len(docs))
        ]

    # ── RRF 融合 ────────────────────────────────────────────────

    @staticmethod
    def _rrf_fuse(
        results_a: list[dict[str, Any]],
        results_b: list[dict[str, Any]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank)。

        同时命中两路的文档天然获得更高分数。k=60 为经典值 (Cormack et al., 2009)。
        """
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

        return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)


def _tag_source(items: list[dict], source: str) -> list[dict]:
    """为 BM25 结果标记来源。"""
    for item in items:
        item["retrieval_sources"] = {source}
    return items
