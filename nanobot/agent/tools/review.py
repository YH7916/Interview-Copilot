"""Tool for triggering post-interview review in the background."""

from __future__ import annotations

import logging
from typing import Any

from nanobot.agent.tools.base import Tool

logger = logging.getLogger(__name__)


class TriggerReviewTool(Tool):
    """Ask copilot to summarize weak points after an interview session."""

    def __init__(self) -> None:
        self._history: list[dict] = []

    @property
    def name(self) -> str:
        return "trigger_interview_review"

    @property
    def description(self) -> str:
        return (
            "Trigger a background interview review. The tool analyzes the current "
            "dialogue history and prepares a review summary for later practice."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "session_summary": {
                    "type": "string",
                    "description": "Short summary of the interview session.",
                }
            },
            "required": ["session_summary"],
        }

    def set_history(self, messages: list[dict]) -> None:
        self._history = messages

    async def execute(self, session_summary: str, **kwargs: Any) -> str:
        try:
            from copilot.app import trigger_background_review

            trigger_background_review(self._history)
            logger.info("ReviewAgent triggered: %s", session_summary)
            return (
                f"Triggered interview review: {session_summary}\n"
                "A background review task is now running."
            )
        except Exception as exc:
            logger.warning("TriggerReviewTool failed: %s", exc)
            return (
                f"Failed to trigger interview review ({exc}). "
                "Run `python evals/run_eval.py` manually if needed."
            )
