"""Reranker — 调用 DashScope gte-rerank 对候选文档重新打分。"""

import logging

import httpx

from copilot.config import get_dashscope_api_key

logger = logging.getLogger(__name__)

_RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


async def rerank(query: str, documents: list[str], top_n: int = 3) -> list[dict]:
    """返回 [{index, score, text}, ...]。失败时按原顺序返回前 top_n。"""
    if not documents:
        return []

    top_n = max(1, min(top_n, len(documents)))
    api_key = get_dashscope_api_key()

    if not api_key:
        return _fallback(documents, top_n)

    payload = {
        "model": "gte-rerank",
        "input": {"query": query, "documents": documents},
        "parameters": {"top_n": top_n, "return_documents": True},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(_RERANK_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in (data.get("output") or {}).get("results") or []:
            idx = int(item.get("index", 0))
            text = (item.get("document") or {}).get("text") or (
                documents[idx] if 0 <= idx < len(documents) else ""
            )
            results.append({"index": idx, "score": float(item.get("relevance_score", 0.0)), "text": text})
        return results[:top_n]
    except Exception as exc:
        logger.warning("Rerank failed, fallback: %s", exc)
        return _fallback(documents, top_n)


def _fallback(documents: list[str], top_n: int) -> list[dict]:
    return [{"index": i, "score": 0.0, "text": documents[i]} for i in range(top_n)]
