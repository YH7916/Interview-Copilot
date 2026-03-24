"""Thin nanobot tool for generating an interview preparation pack."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool


class PrepareInterviewPrepTool(Tool):
    @property
    def name(self) -> str:
        return "prepare_interview_prep"

    @property
    def description(self) -> str:
        return (
            "Generate an interview preparation pack from the local resume/profile, target role, "
            "and question bank. Use this when the user wants to prepare before running a mock interview."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Optional focus topic such as Agent, RAG, Prompt, Frontend, or Python.",
                },
                "company": {
                    "type": "string",
                    "description": "Optional target company name.",
                },
                "position": {
                    "type": "string",
                    "description": "Optional target role name.",
                },
                "target_description": {
                    "type": "string",
                    "description": "Optional short JD or target-role summary.",
                },
                "max_questions": {
                    "type": "integer",
                    "description": "Number of seed questions to include in the prep pack.",
                    "minimum": 4,
                    "maximum": 12,
                },
                "recent": {
                    "type": "boolean",
                    "description": "Use the recent question bank instead of the full history.",
                },
                "candidate_profile": {
                    "type": "string",
                    "description": "Optional candidate profile or resume summary used to tailor the prep pack.",
                },
                "candidate_profile_path": {
                    "type": "string",
                    "description": "Optional local path to a .txt, .md, or .typ resume file.",
                },
            },
        }

    async def execute(
        self,
        topic: str = "",
        company: str = "",
        position: str = "",
        target_description: str = "",
        max_questions: int = 8,
        recent: bool = True,
        candidate_profile: str = "",
        candidate_profile_path: str = "",
    ) -> str:
        from copilot.app import render_interview_prep_pack

        return render_interview_prep_pack(
            topic=topic,
            company=company,
            position=position,
            target_description=target_description,
            max_questions=max_questions,
            recent=recent,
            candidate_profile=candidate_profile,
            candidate_profile_path=candidate_profile_path,
        )
