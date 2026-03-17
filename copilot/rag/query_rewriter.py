"""Query 重写 — 用轻量 LLM 把口语查询转为检索友好的关键词。"""

import logging

from openai import AsyncOpenAI

from copilot.config import get_dashscope_api_key

logger = logging.getLogger(__name__)

_SYSTEM = (
    "你是检索优化器。把用户口语问题改写为适合向量与关键词检索的简洁中文查询，"
    "保留核心实体和意图。只输出改写后的查询，不要解释。"
)


async def rewrite_query(raw_query: str) -> str:
    """口语 → 标准化检索词。失败时静默回退到原始 query。"""
    if not raw_query or not raw_query.strip():
        return raw_query

    api_key = get_dashscope_api_key()
    if not api_key:
        return raw_query

    try:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        resp = await client.chat.completions.create(
            model="qwen-turbo",
            messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": raw_query}],
            temperature=0.1,
            max_tokens=64,
        )
        return (resp.choices[0].message.content or "").strip() or raw_query
    except Exception as exc:
        logger.warning("Query rewrite failed, fallback: %s", exc)
        return raw_query
