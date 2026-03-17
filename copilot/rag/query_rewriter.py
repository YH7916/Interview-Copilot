import json
import logging
import os
from pathlib import Path

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def _read_config_key() -> str | None:
    try:
        cfg = json.loads((Path.home() / ".nanobot" / "config.json").read_text(encoding="utf-8"))
        return cfg.get("providers", {}).get("dashscope", {}).get("apiKey")
    except Exception:
        return None


async def rewrite_query(raw_query: str) -> str:
    """将口语化查询改写成更适合检索的查询词。失败时返回原始 query。"""
    if not raw_query or not raw_query.strip():
        return raw_query

    api_key = os.environ.get("DASHSCOPE_API_KEY") or _read_config_key()
    if not api_key:
        return raw_query

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    try:
        resp = await client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "你是检索优化器。把用户口语问题改写为适合向量与关键词检索的简洁中文查询，保留核心实体和意图。只输出改写后的查询，不要解释。",
                },
                {"role": "user", "content": raw_query},
            ],
            temperature=0.1,
            max_tokens=64,
        )
        content = (resp.choices[0].message.content or "").strip()
        return content or raw_query
    except Exception as exc:
        logger.warning("Query rewrite failed, fallback to raw query: %s", exc)
        return raw_query
