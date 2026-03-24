"""Nanobot tools backed by the interview knowledge base."""

from __future__ import annotations

import asyncio
from typing import Any

from nanobot.agent.tools.base import Tool

from copilot.knowledge.retrieval import HybridRetriever

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _format_hits(title: str, hits: list[dict[str, Any]]) -> str:
    blocks = []
    for hit in hits:
        source = hit["metadata"].get("source", "unknown")
        blocks.append(f"Source: {source}\n{hit['text']}")
    return f"{title}\n\n" + "\n\n---\n\n".join(blocks)


class SearchConceptTool(Tool):
    @property
    def name(self) -> str:
        return "search_concept"

    @property
    def description(self) -> str:
        return "Search the local interview knowledge base for an AI or agent concept."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"topic": {"type": "string", "description": "Concept to search for"}},
            "required": ["topic"],
        }

    async def execute(self, topic: str, **kwargs: Any) -> str:
        retriever = await asyncio.to_thread(_get_retriever)
        hits = await retriever.search(topic, top_k_retrieve=10, top_n_rerank=2)
        if not hits:
            return f"No concept notes found for '{topic}'."
        return _format_hits(f"Concept notes for '{topic}'", hits)


class SearchCompanyQuestionsTool(Tool):
    @property
    def name(self) -> str:
        return "search_company_questions"

    @property
    def description(self) -> str:
        return "Search the local interview knowledge base for company-specific questions."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "company": {"type": "string", "description": "Company name"},
                "position": {"type": "string", "description": "Position name"},
            },
            "required": ["company", "position"],
        }

    async def execute(self, company: str, position: str, **kwargs: Any) -> str:
        query = f"{company} {position} 面试题"
        retriever = await asyncio.to_thread(_get_retriever)
        hits = await retriever.search(query, top_k_retrieve=10, top_n_rerank=3)
        if not hits:
            return f"No company-specific notes found for '{company} {position}'."
        return _format_hits(f"Interview notes for {company} {position}", hits)
