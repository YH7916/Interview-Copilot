"""Business tool for collecting Nowcoder interview materials."""

from __future__ import annotations

import json
from typing import Any

from nanobot.agent.tools.base import Tool


class CollectNowcoderInterviewsTool(Tool):
    """Wrap the copilot Nowcoder ingestion flow as a nanobot tool."""

    def __init__(self, ingestor: Any | None = None) -> None:
        self._ingestor = ingestor

    @property
    def name(self) -> str:
        return "collect_nowcoder_interviews"

    @property
    def description(self) -> str:
        return (
            "Collect AI/Agent interview materials from Nowcoder, write them into "
            "knowledge_base/10-RealQuestions, and rebuild the local question bank. "
            "Use this first for 面经, 面试题, 牛客, 面筋, or 凉经 instead of generic web_search. "
            "For broad collection tasks, leave query empty so the tool uses built-in "
            "company-plus-round search templates and expands the local knowledge base."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional custom Nowcoder-oriented search query.",
                },
                "count_per_query": {
                    "type": "integer",
                    "description": "Search results to inspect for each query.",
                    "minimum": 1,
                    "maximum": 50,
                },
                "max_reports": {
                    "type": "integer",
                    "description": "Maximum interview material files to save.",
                    "minimum": 1,
                    "maximum": 200,
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Discover materials without writing files.",
                },
                "fetch_timeout": {
                    "type": "number",
                    "description": "Per-page fetch timeout in seconds.",
                    "minimum": 3,
                    "maximum": 30,
                },
                "updated_within_days": {
                    "type": "integer",
                    "description": "Only keep posts updated within this many days.",
                    "minimum": 1,
                    "maximum": 90,
                },
            },
        }

    async def execute(
        self,
        query: str | None = None,
        count_per_query: int = 6,
        max_reports: int = 8,
        dry_run: bool = False,
        fetch_timeout: float = 12.0,
        updated_within_days: int = 30,
    ) -> str:
        from copilot.app import collect_nowcoder_interviews

        if self._ingestor is not None:
            result = await self._ingestor.run(
                queries=[query] if query else None,
                count_per_query=count_per_query,
                max_reports=max_reports,
                dry_run=dry_run,
                rebuild_index=not dry_run,
            )
        else:
            result = await collect_nowcoder_interviews(
                query=query,
                count_per_query=count_per_query,
                max_reports=max_reports,
                dry_run=dry_run,
                fetch_timeout=fetch_timeout,
                updated_within_days=updated_within_days,
                rebuild_index=not dry_run,
            )
        return json.dumps(result, ensure_ascii=False, indent=2)
