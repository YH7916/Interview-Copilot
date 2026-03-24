"""Rerank retrieved candidates when a dedicated rerank model is available."""

from __future__ import annotations

import logging

import httpx

from copilot.config import get_rerank_settings

logger = logging.getLogger(__name__)


async def rerank(query: str, documents: list[str], top_n: int = 3) -> list[dict]:
    if not documents:
        return []

    top_n = max(1, min(top_n, len(documents)))
    settings = get_rerank_settings()
    if not (settings["api_key"] and settings["model"] and settings["url"]):
        return _fallback(documents, top_n)

    payload = {
        "model": settings["model"],
        "input": {"query": query, "documents": documents},
        "parameters": {"top_n": top_n, "return_documents": True},
    }
    headers = {
        "Authorization": f"Bearer {settings['api_key']}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(settings["url"], headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("Rerank failed, keeping original order: %s", exc)
        return _fallback(documents, top_n)

    results = []
    for item in (data.get("output") or {}).get("results") or []:
        index = int(item.get("index", 0))
        if 0 <= index < len(documents):
            text = (item.get("document") or {}).get("text") or documents[index]
            results.append(
                {
                    "index": index,
                    "score": float(item.get("relevance_score", 0.0)),
                    "text": text,
                }
            )
    return results[:top_n]


def _fallback(documents: list[str], top_n: int) -> list[dict]:
    return [{"index": index, "score": 0.0, "text": documents[index]} for index in range(top_n)]
