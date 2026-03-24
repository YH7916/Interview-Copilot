"""Thin nanobot tool for starting a lightweight mock interview."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool


class PrepareMockInterviewTool(Tool):
    @property
    def name(self) -> str:
        return "prepare_mock_interview"

    @property
    def description(self) -> str:
        return (
            "Prepare a lightweight mock interview plan from the local question bank. "
            "Use this before starting an interview session instead of inventing questions from scratch."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Optional focus topic such as Agent, RAG, Prompt, or Python.",
                },
                "max_questions": {
                    "type": "integer",
                    "description": "Number of planned questions to prepare.",
                    "minimum": 3,
                    "maximum": 12,
                },
                "recent": {
                    "type": "boolean",
                    "description": "Use the recent question bank instead of the full history.",
                },
                "candidate_profile": {
                    "type": "string",
                    "description": "Optional candidate profile or resume summary used to tailor the question mix.",
                },
                "candidate_profile_path": {
                    "type": "string",
                    "description": "Optional local path to a .txt, .md, or .typ resume file.",
                },
                "interview_style": {
                    "type": "string",
                    "description": "Optional interview flow style: auto, coding-first, or no-coding.",
                },
            },
        }

    async def execute(
        self,
        topic: str = "",
        max_questions: int = 6,
        recent: bool = True,
        candidate_profile: str = "",
        candidate_profile_path: str = "",
        interview_style: str = "auto",
    ) -> str:
        from copilot.app import render_mock_interview_plan

        return render_mock_interview_plan(
            topic=topic,
            max_questions=max_questions,
            recent=recent,
            candidate_profile=candidate_profile,
            candidate_profile_path=candidate_profile_path,
            interview_style=interview_style,
        )
