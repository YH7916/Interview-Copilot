import logging

import httpx

from copilot.config import get_dashscope_api_key

logger = logging.getLogger(__name__)


async def rerank(query: str, documents: list[str], top_n: int = 3) -> list[dict]:
    """调用 DashScope rerank 接口，返回重排后的文档。失败时按原顺序返回前 top_n。"""
    if not documents:
        return []

    top_n = max(1, min(top_n, len(documents)))
    api_key = get_dashscope_api_key()
    if not api_key:
        return [{"index": i, "score": 0.0, "text": documents[i]} for i in range(top_n)]

    url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gte-rerank",
        "input": {"query": query, "documents": documents},
        "parameters": {"top_n": top_n, "return_documents": True},
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        parsed: list[dict] = []
        for item in (data.get("output") or {}).get("results") or []:
            idx = int(item.get("index", 0))
            score = float(item.get("relevance_score", 0.0))
            doc_obj = item.get("document") or {}
            text = doc_obj.get("text") or (documents[idx] if 0 <= idx < len(documents) else "")
            parsed.append({"index": idx, "score": score, "text": text})
        return parsed[:top_n]
    except Exception as exc:
        logger.warning("Rerank failed, fallback to original order: %s", exc)
        return [{"index": i, "score": 0.0, "text": documents[i]} for i in range(top_n)]
