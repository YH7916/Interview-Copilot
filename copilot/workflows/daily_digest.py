"""Formal daily digest built from the local interview knowledge base."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from copilot.knowledge.question_bank import RECENT_JSON_PATH, REPORT_INDEX_PATH
from copilot.sources.nowcoder import is_agent_relevant

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)
FRONTMATTER_LINE_RE = re.compile(r"^([A-Za-z0-9_]+):\s*(.+)$")
TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
ROUND_RE = re.compile(r"(一二面|二三面|三四面|一面|二面|三面|四面|五面|HR面|终面|面经|凉经|oc|offer)", re.IGNORECASE)
COMPANY_CLEAN_RE = re.compile(r"(?i)\b(ai|agent|llm|rag|mcp|genai)\b")
ROLE_SUFFIX_RE = re.compile(
    r"(?i)\s*(开发|研发|前端|后端|算法|测试|客户端|服务端|产品|java|c\+\+软件开发|实习|校招|社招)\s*$"
)


def build_daily_digest(
    *,
    days: int = 1,
    report_index_path: Path = REPORT_INDEX_PATH,
    recent_bank_path: Path = RECENT_JSON_PATH,
) -> dict[str, Any]:
    window_days = max(1, days)
    cutoff = datetime.now() - timedelta(days=window_days)
    records = _load_json(report_index_path, default=[])
    bank = _load_json(recent_bank_path, default={"categories": []})

    fresh_reports = [
        _build_report_snapshot(record, cutoff)
        for record in records
        if _is_fresh_record(record, cutoff) and _is_digest_candidate(record)
    ]
    fresh_reports = [item for item in fresh_reports if item is not None]
    fresh_reports.sort(key=_report_sort_key, reverse=True)

    company_breakdown = _build_company_breakdown(fresh_reports)
    category_breakdown = _build_category_breakdown(bank)
    representative_questions = _build_representative_questions(bank)
    summary = _build_summary(fresh_reports, company_breakdown, category_breakdown)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "window_days": window_days,
        "summary": summary,
        "top_reports": fresh_reports[:5],
        "company_breakdown": company_breakdown[:5],
        "category_breakdown": category_breakdown[:5],
        "representative_questions": representative_questions[:6],
    }


def render_daily_digest_markdown(digest: dict[str, Any]) -> str:
    summary = digest.get("summary", {})
    lines = [
        "# Daily Digest",
        "",
        f"- generated_at: {digest.get('generated_at', '')}",
        f"- window_days: {digest.get('window_days', 1)}",
        "",
        "## Summary",
        f"- fresh_reports: {summary.get('fresh_reports', 0)}",
        f"- fresh_questions: {summary.get('fresh_questions', 0)}",
        f"- active_companies: {summary.get('active_companies', 0)}",
        f"- hottest_company: {summary.get('hottest_company', 'N/A')}",
        f"- hottest_topic: {summary.get('hottest_topic', 'N/A')}",
        f"- average_questions_per_report: {summary.get('avg_questions_per_report', 0.0):.1f}",
    ]

    newest_report = summary.get("newest_report")
    if newest_report:
        lines.append(f"- newest_report: {newest_report}")

    lines.extend(["", "## Top Reports"])
    for report in digest.get("top_reports", []):
        lines.extend(
            [
                f"- {report.get('title', 'Untitled')}",
                f"  company: {report.get('company', 'Unknown')}, round: {report.get('round', 'Unknown')}, questions: {report.get('question_count', 0)}",
                f"  updated_at: {report.get('source_updated_at', '')}, signal: {report.get('signal_score', 0.0):.1f}",
                f"  url: {report.get('source_url', '')}",
                "",
            ]
        )

    lines.extend(["## Active Companies"])
    for item in digest.get("company_breakdown", []):
        lines.append(f"- {item['name']}: {item['count']} reports")

    lines.extend(["", "## Hot Topics"])
    for item in digest.get("category_breakdown", []):
        lines.append(f"- {item['label']}: {item['count']} questions")

    lines.extend(["", "## Representative Questions"])
    for item in digest.get("representative_questions", []):
        lines.append(
            f"- [{item['category_label']}] {item['question']} "
            f"(sources: {item['source_count']}, latest: {item['latest_source_at']})"
        )

    return "\n".join(lines).rstrip() + "\n"


def _load_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _is_fresh_record(record: dict[str, Any], cutoff: datetime) -> bool:
    captured_at = _parse_time(record.get("captured_at"))
    return captured_at is not None and captured_at >= cutoff


def _build_report_snapshot(record: dict[str, Any], cutoff: datetime) -> dict[str, Any] | None:
    captured_at = _parse_time(record.get("captured_at"))
    if captured_at is None:
        return None

    snapshot = dict(record)
    snapshot.update(_load_report_metrics(record.get("source_path", "")))
    snapshot["title"] = snapshot.get("title") or "Untitled"
    snapshot["company"] = _extract_company(snapshot["title"])
    snapshot["round"] = _extract_round(snapshot["title"])
    snapshot["question_count"] = len(snapshot.get("questions", []))
    snapshot["signal_score"] = _signal_score(snapshot, captured_at, cutoff)
    return snapshot


def _load_report_metrics(source_path: str) -> dict[str, Any]:
    path = Path(source_path)
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    metrics: dict[str, Any] = {}

    match = FRONTMATTER_RE.match(text)
    if match:
        for line in match.group(1).splitlines():
            parsed = FRONTMATTER_LINE_RE.match(line.strip())
            if not parsed:
                continue
            key, raw_value = parsed.groups()
            value = raw_value.strip().strip('"')
            metrics[key] = int(value) if value.isdigit() else value

    title_match = TITLE_RE.search(text)
    if title_match:
        metrics.setdefault("title", title_match.group(1).strip())
    return metrics


def _signal_score(item: dict[str, Any], captured_at: datetime, cutoff: datetime) -> float:
    freshness = max((captured_at - cutoff).total_seconds() / 86400.0, 0.0)
    questions = float(len(item.get("questions", [])))
    likes = float(item.get("source_like_count", 0))
    comments = float(item.get("source_comment_count", 0))
    views = float(item.get("source_view_count", 0))
    return questions * 4.0 + likes * 3.0 + comments * 2.0 + views / 50.0 + freshness


def _report_sort_key(item: dict[str, Any]) -> tuple[float, str, int]:
    return (
        float(item.get("signal_score", 0.0)),
        str(item.get("source_updated_at") or item.get("captured_at") or ""),
        int(item.get("question_count", 0)),
    )


def _build_company_breakdown(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for report in reports:
        company = str(report.get("company") or "Unknown")
        counts[company] = counts.get(company, 0) + 1
    return [
        {"name": name, "count": count}
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _build_category_breakdown(bank: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for category in bank.get("categories", []):
        items = category.get("clusters") or category.get("questions") or []
        result.append(
            {
                "name": category.get("name", ""),
                "label": category.get("label", category.get("name", "")),
                "count": len(items),
            }
        )
    result.sort(key=lambda item: (-item["count"], item["label"]))
    return result


def _build_representative_questions(bank: dict[str, Any]) -> list[dict[str, Any]]:
    questions = []
    for category in bank.get("categories", []):
        label = category.get("label", category.get("name", ""))
        items = category.get("clusters") or category.get("questions") or []
        for item in items:
            questions.append(
                {
                    "question": item.get("question", ""),
                    "category_label": label,
                    "source_count": int(item.get("source_count", 0)),
                    "latest_source_at": item.get("latest_source_at", ""),
                }
            )
    questions.sort(
        key=lambda item: (
            item["source_count"],
            item["latest_source_at"],
            item["question"],
        ),
        reverse=True,
    )
    return questions


def _build_summary(
    reports: list[dict[str, Any]],
    company_breakdown: list[dict[str, Any]],
    category_breakdown: list[dict[str, Any]],
) -> dict[str, Any]:
    fresh_reports = len(reports)
    fresh_questions = sum(int(report.get("question_count", 0)) for report in reports)
    newest_report = reports[0]["title"] if reports else ""
    return {
        "fresh_reports": fresh_reports,
        "fresh_questions": fresh_questions,
        "active_companies": len(company_breakdown),
        "avg_questions_per_report": fresh_questions / fresh_reports if fresh_reports else 0.0,
        "hottest_company": company_breakdown[0]["name"] if company_breakdown else "N/A",
        "hottest_topic": category_breakdown[0]["label"] if category_breakdown else "N/A",
        "newest_report": newest_report,
    }


def _is_digest_candidate(record: dict[str, Any]) -> bool:
    haystack = f"{record.get('title', '')}\n" + "\n".join(record.get("questions") or [])
    return is_agent_relevant(haystack.lower(), title=str(record.get("title", "")))


def _extract_company(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", title).strip()
    if not cleaned:
        return "Unknown"

    match = ROUND_RE.search(cleaned)
    head = cleaned[: match.start()].strip(" -_:|/") if match else cleaned
    while True:
        updated = COMPANY_CLEAN_RE.sub("", head)
        updated = ROLE_SUFFIX_RE.sub("", updated)
        updated = re.sub(r"\s+", " ", updated).strip(" -_:|/")
        if updated == head:
            break
        head = updated
    return head[:24] if head else "Unknown"


def _extract_round(title: str) -> str:
    match = ROUND_RE.search(title)
    return match.group(1) if match else "Unknown"


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
