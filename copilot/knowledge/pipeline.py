"""High-level knowledge rebuild helpers."""

from __future__ import annotations

from typing import Any

from copilot.knowledge.answer_cards import rebuild_answer_cards
from copilot.knowledge.index import rebuild_chroma_collection
from copilot.knowledge.question_bank import rebuild_question_bank


async def rebuild_knowledge(*, with_web: bool = False, max_cards: int | None = None) -> dict[str, Any]:
    bank = rebuild_question_bank()
    cards = await rebuild_answer_cards(use_web=with_web, max_cards=max_cards)
    rebuild_chroma_collection()
    return {
        "question_bank_generated_at": bank.get("generated_at", ""),
        "answer_cards_generated_at": cards.get("generated_at", ""),
    }


__all__ = ["rebuild_knowledge"]
