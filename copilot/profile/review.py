"""Background review agent for summarising interview weak points."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from copilot.profile.store import WeaknessTracker
from copilot.llm import call_text

logger = logging.getLogger(__name__)

REVIEW_PROMPT = """Review the interview dialogue and summarise the candidate's recurring weak points.
Return 2-4 markdown bullet points. Focus on reasoning, clarity, depth, and missing evidence.

Dialogue:
{dialogue}
"""

WEAK_KEYWORDS = ("不清楚", "模糊", "不会", "不太会", "不知道", "忘了", "卡住")


class ReviewAgent:
    def __init__(self, tracker: WeaknessTracker | None = None):
        self.tracker = tracker

    def analyze_background(self, messages: list[dict[str, Any]]) -> asyncio.Task[None]:
        return asyncio.create_task(self._run_background(messages))

    async def analyze(self, messages: list[dict[str, Any]]) -> str:
        return await asyncio.to_thread(self._analyze, messages)

    async def _run_background(self, messages: list[dict[str, Any]]) -> None:
        try:
            await asyncio.to_thread(self._analyze, messages)
            logger.info("ReviewAgent finished background analysis.")
        except Exception:
            logger.exception("ReviewAgent failed during background analysis.")

    def _analyze(self, messages: list[dict[str, Any]]) -> str:
        from copilot.interview.orchestrator import InterviewRunner

        structured = InterviewRunner(tracker=self.tracker).review_messages(
            messages,
            persist=self.tracker is not None,
        )
        if structured["results"]:
            return structured["summary"]

        dialogue = _format_dialogue(messages)

        try:
            bullets = call_text(REVIEW_PROMPT.format(dialogue=dialogue), task="analysis")
        except Exception as exc:
            bullets = _fallback_extract(messages, exc)

        if self.tracker is not None:
            self.tracker._append(_build_review_entry(bullets))
        return bullets


def _format_dialogue(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = "interviewer" if message.get("role") == "assistant" else "candidate"
        content = _message_text(message)
        if content:
            lines.append(f"[{role}]: {content[:300]}")
    return "\n".join(lines)


def _fallback_extract(messages: list[dict[str, Any]], err: Exception) -> str:
    hints = [
        f"- **{_message_text(message)[:40]}**: the answer sounds uncertain or incomplete."
        for message in messages
        if message.get("role") == "user"
        and any(keyword in _message_text(message) for keyword in WEAK_KEYWORDS)
    ]
    return "\n".join(hints[:4]) or f"- Review fallback used because analysis failed: {err}"


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        return " ".join(item.get("text", "") for item in content if isinstance(item, dict))
    return str(content or "")


def _build_review_entry(bullets: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"## [{timestamp}] Interview review\n\n{bullets}\n"
