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
    """高级检索流水线：改写 -> BM25+向量召回 -> 合并去重 -> 重排。"""

    def __init__(self, collection: Any | None = None, bm25_retriever: BM25Retriever | None = None):
        self._collection = collection or get_chroma_collection()
        self._bm25 = bm25_retriever or BM25Retriever(self._collection)

    def _vector_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        raw = self._collection.query(query_texts=[query], n_results=top_k)
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        ids = (raw.get("ids") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: list[dict[str, Any]] = []
        for idx, text in enumerate(docs):
            results.append(
                {
                    "id": ids[idx] if idx < len(ids) else f"vec_{idx}",
                    "text": text,
                    "metadata": metas[idx] if idx < len(metas) else {},
                    "vector_distance": float(distances[idx]) if idx < len(distances) else None,
                    "retrieval_sources": {"vector"},
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
                doc_id = str(item.get("id", ""))
                text = (item.get("text") or "").strip()
                key = (doc_id, text)

                if key not in fused:
                    new_item = dict(item)
                    new_item["retrieval_sources"] = set(item.get("retrieval_sources", set()))
                    new_item["rrf_score"] = 0.0
                    fused[key] = new_item

                fused_item = fused[key]
                fused_item["rrf_score"] += 1.0 / (k + rank)
                fused_item.setdefault("retrieval_sources", set()).update(item.get("retrieval_sources", set()))

        return sorted(fused.values(), key=lambda x: float(x.get("rrf_score", 0.0)), reverse=True)

    async def search(
        self,
        raw_query: str,
        top_k_retrieve: int = 10,
        top_n_rerank: int = 3,
    ) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        logger.info("HybridRetriever.search | raw_query=%s", raw_query[:80])

        rewritten_query = await rewrite_query(raw_query)
        logger.info("Query rewritten | original=%s | rewritten=%s", raw_query[:50], rewritten_query[:50])

        bm25_task = asyncio.to_thread(self._bm25.search, rewritten_query, top_k_retrieve)
        vector_task = asyncio.to_thread(self._vector_search, rewritten_query, top_k_retrieve)
        bm25_results, vector_results = await asyncio.gather(bm25_task, vector_task, return_exceptions=True)

        bm25_items: list[dict[str, Any]] = []
        if isinstance(bm25_results, Exception):
            logger.warning("BM25 retrieval failed: %s", bm25_results)
        else:
            for item in bm25_results:
                new_item = dict(item)
                new_item["retrieval_sources"] = {"bm25"}
                bm25_items.append(new_item)

        vector_items: list[dict[str, Any]] = []
        if isinstance(vector_results, Exception):
            logger.warning("Vector retrieval failed: %s", vector_results)
        else:
            vector_items = vector_results

        logger.info("Retrieval done | bm25=%d | vector=%d", len(bm25_items), len(vector_items))

        merged = self._rrf_fuse(bm25_items, vector_items, k=60)
        logger.info("RRF fusion done | merged=%d", len(merged))
        if not merged:
            logger.info(
                "HybridRetriever.search done | results=%d | elapsed=%.2fs",
                0, time.perf_counter() - t0,
            )
            return []

        merged_docs = [item.get("text", "") for item in merged]
        reranked = await rerank(rewritten_query, merged_docs, top_n=top_n_rerank)

        final_results: list[dict[str, Any]] = []
        for item in reranked:
            idx = int(item.get("index", 0))
            if idx < 0 or idx >= len(merged):
                continue
            src = merged[idx]
            final_results.append(
                {
                    "id": src.get("id"),
                    "text": item.get("text") or src.get("text", ""),
                    "metadata": src.get("metadata", {}),
                    "rerank_score": float(item.get("score", 0.0)),
                    "retrieval_sources": sorted(src.get("retrieval_sources", set())),
                    "query": rewritten_query,
                }
            )

        logger.info(
            "HybridRetriever.search done | results=%d | elapsed=%.2fs",
            len(final_results), time.perf_counter() - t0,
        )
        return final_results
