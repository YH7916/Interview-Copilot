"""Thin nanobot tool for the local interview daily digest."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool


class ShowDailyDigestTool(Tool):
    @property
    def name(self) -> str:
        return "show_daily_digest"

    @property
    def description(self) -> str:
        return (
            "Show the local Agent interview daily digest built from the existing knowledge base. "
            "Use this when the user wants today's summary, recent trends, top reports, or hot topics."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Look back this many days when building the digest.",
                    "minimum": 1,
                    "maximum": 30,
                }
            },
        }

    async def execute(self, days: int = 1) -> str:
        from copilot.app import render_daily_digest

        return render_daily_digest(days=days)
