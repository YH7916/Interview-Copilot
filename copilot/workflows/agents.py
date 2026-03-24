"""Executable role-based workflow for daily copilot operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Awaitable, Callable

from copilot.knowledge.pipeline import rebuild_knowledge
from copilot.workflows.daily_digest import build_daily_digest

CollectFn = Callable[..., Awaitable[dict[str, Any]]]
RebuildFn = Callable[..., Awaitable[dict[str, Any]]]
DigestFn = Callable[..., dict[str, Any]]
StageRunner = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def build_agent_workflow() -> list[dict[str, str]]:
    return [
        {
            "role": "scout",
            "kind": "ingest",
            "responsibility": "collect fresh interview reports from Nowcoder",
        },
        {
            "role": "curator",
            "kind": "knowledge",
            "responsibility": "rebuild the question bank, answer cards, and retrieval index",
        },
        {
            "role": "reporter",
            "kind": "digest",
            "responsibility": "publish the local daily digest",
        },
    ]


async def run_agent_workflow(
    *,
    collect_fn: CollectFn,
    rebuild_fn: RebuildFn = rebuild_knowledge,
    digest_fn: DigestFn = build_daily_digest,
    updated_within_days: int = 7,
    count_per_query: int = 20,
    max_reports: int = 50,
    fetch_timeout: float = 12.0,
    with_web: bool = False,
    max_cards: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "collect_fn": collect_fn,
        "rebuild_fn": rebuild_fn,
        "digest_fn": digest_fn,
        "updated_within_days": updated_within_days,
        "count_per_query": count_per_query,
        "max_reports": max_reports,
        "fetch_timeout": fetch_timeout,
        "with_web": with_web,
        "max_cards": max_cards,
        "dry_run": dry_run,
    }
    stages = []

    for step in build_agent_workflow():
        runner = _STAGE_RUNNERS[step["role"]]
        started_at = datetime.now().isoformat(timespec="seconds")
        result = await runner(context)
        finished_at = datetime.now().isoformat(timespec="seconds")
        stages.append(
            {
                "role": step["role"],
                "kind": step["kind"],
                "responsibility": step["responsibility"],
                "started_at": started_at,
                "finished_at": finished_at,
                "result": result,
            }
        )
        context[step["role"]] = result

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "workflow": "daily_ops",
        "stages": stages,
        "digest": context.get("reporter", {}).get("full_digest", {}),
    }


async def _run_scout(context: dict[str, Any]) -> dict[str, Any]:
    return await context["collect_fn"](
        count_per_query=context["count_per_query"],
        max_reports=context["max_reports"],
        dry_run=context["dry_run"],
        fetch_timeout=context["fetch_timeout"],
        updated_within_days=context["updated_within_days"],
        rebuild_index=False,
    )


async def _run_curator(context: dict[str, Any]) -> dict[str, Any]:
    if context["dry_run"]:
        return {"status": "skipped", "reason": "dry_run"}

    knowledge = await context["rebuild_fn"](
        with_web=context["with_web"],
        max_cards=context["max_cards"],
    )
    return {"status": "ok", "knowledge": knowledge}


async def _run_reporter(context: dict[str, Any]) -> dict[str, Any]:
    digest = context["digest_fn"](days=context["updated_within_days"])
    return {
        "status": "ok",
        "summary": digest.get("summary", {}),
        "top_reports": digest.get("top_reports", [])[:5],
        "full_digest": digest,
    }


_STAGE_RUNNERS: dict[str, StageRunner] = {
    "scout": _run_scout,
    "curator": _run_curator,
    "reporter": _run_reporter,
}

