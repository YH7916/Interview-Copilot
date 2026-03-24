"""Lightweight overview helpers for recently collected interview reports."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from copilot.knowledge.question_bank import load_report_index
from copilot.sources.nowcoder import is_agent_relevant


def build_recent_reports_overview(*, days: int = 7, limit: int = 10) -> dict[str, Any]:
    reports = _recent_reports(load_report_index(), days=days)
    reports.sort(key=lambda item: item.get("captured_at", ""), reverse=True)
    items = [
        {
            "title": report.get("title", ""),
            "captured_at": report.get("captured_at", ""),
            "question_count": len(report.get("questions") or []),
            "source_url": report.get("source_url", ""),
            "source_path": report.get("source_path", ""),
        }
        for report in reports[:limit]
    ]
    return {
        "days": days,
        "limit": limit,
        "reports": len(reports),
        "questions": sum(len(report.get("questions") or []) for report in reports),
        "items": items,
    }


def render_recent_reports_overview(*, days: int = 7, limit: int = 10) -> str:
    overview = build_recent_reports_overview(days=days, limit=limit)
    if not overview["items"]:
        return f"No recent materials found in the last {days} day(s). Run /ingest {days} first."

    lines = [
        "# Recent Materials",
        "",
        f"- window_days: {days}",
        f"- reports: {overview['reports']}",
        f"- questions: {overview['questions']}",
        "",
    ]
    for index, item in enumerate(overview["items"], 1):
        lines.extend(
            [
                f"{index}. {item['title']}",
                f"   - captured_at: {item['captured_at']}",
                f"   - questions: {item['question_count']}",
                f"   - url: {item['source_url']}",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _recent_reports(reports: list[dict[str, Any]], *, days: int) -> list[dict[str, Any]]:
    cutoff = datetime.now() - timedelta(days=max(days, 1))
    recent: list[dict[str, Any]] = []
    for report in reports:
        try:
            captured_at = datetime.fromisoformat(report.get("captured_at", ""))
        except ValueError:
            continue
        haystack = f"{report.get('title', '')}\n" + "\n".join(report.get("questions") or [])
        if not is_agent_relevant(haystack.lower(), title=str(report.get("title", ""))):
            continue
        if captured_at >= cutoff:
            recent.append(report)
    return recent


__all__ = ["build_recent_reports_overview", "render_recent_reports_overview"]
