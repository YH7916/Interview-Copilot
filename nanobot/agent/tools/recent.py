"""Thin nanobot tool for browsing recently kept interview materials."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool


class ShowRecentReportsTool(Tool):
    @property
    def name(self) -> str:
        return "show_recent_reports"

    @property
    def description(self) -> str:
        return (
            "Show recently kept interview materials from the local knowledge base. "
            "Use this to inspect what was collected before starting an interview or digest."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Only show materials collected within this many days.",
                    "minimum": 1,
                    "maximum": 30,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of recent materials to show.",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
        }

    async def execute(self, days: int = 7, limit: int = 10) -> str:
        from copilot.app import render_recent_reports_overview

        return render_recent_reports_overview(days=days, limit=limit)
