"""Query rewriting for retrieval-friendly search terms."""

from __future__ import annotations

import logging

from copilot.config import get_text_settings
from copilot.llm import create_async_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Rewrite the user's interview query into concise retrieval-friendly terms. "
    "Keep key technologies, job context, and company names. Return only the rewritten query."
)


async def rewrite_query(raw_query: str) -> str:
    if not raw_query or not raw_query.strip():
        return raw_query

    settings = get_text_settings("rewrite")
    if not settings["api_key"]:
        return raw_query

    try:
        client, _ = create_async_client(task="rewrite")
        response = await client.chat.completions.create(
            model=settings["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_query},
            ],
            temperature=0.1,
            max_tokens=64,
        )
        return (response.choices[0].message.content or "").strip() or raw_query
    except Exception as exc:
        logger.warning("Query rewrite failed, using raw query: %s", exc)
        return raw_query
