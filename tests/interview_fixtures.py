from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from copilot.interview.orchestrator import InterviewRunner
from copilot.interview.planner import InterviewPlanner, PlannedQuestion
from copilot.profile.store import WeaknessTracker

GENERATED_AT = "2026-03-20T12:00:00"
SOURCE_AT = "2026-03-20T10:00:00"


def make_log_path(prefix: str = "copilot-review-") -> Path:
    handle = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".md", delete=False)
    handle.close()
    return Path(handle.name)


def make_trace_dir() -> Path:
    path = Path(__file__).resolve().parents[1] / "data" / "_test_workspace" / "traces"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_source(
    title: str = "字节一面",
    source_url: str = "a",
    source_path: str = "a.md",
    captured_at: str = SOURCE_AT,
) -> dict[str, str]:
    return {
        "title": title,
        "source_url": source_url,
        "source_path": source_path,
        "captured_at": captured_at,
    }


def make_report(
    title: str,
    questions: list[str],
    *,
    source_url: str = "https://example.com/report",
    source_path: str = "report.md",
    captured_at: str = SOURCE_AT,
    **extra: Any,
) -> dict[str, Any]:
    report = {
        "title": title,
        "source_url": source_url,
        "source_path": source_path,
        "captured_at": captured_at,
        "questions": questions,
    }
    report.update(extra)
    return report


def make_question_entry(
    question: str,
    category: str,
    category_label: str,
    *,
    stage: str = "project",
    aliases: list[str] | None = None,
    source_count: int = 2,
    latest_source_at: str = SOURCE_AT,
    sources: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    entry = {
        "question": question,
        "category": category,
        "category_label": category_label,
        "stage": stage,
        "aliases": aliases or [question],
        "source_count": source_count,
        "latest_source_at": latest_source_at,
        "sources": sources or [make_source()],
    }
    entry.update(extra)
    return entry


def make_cluster(
    question: str,
    category: str,
    category_label: str,
    *,
    stage: str = "project",
    aliases: list[str] | None = None,
    follow_ups: list[str] | None = None,
    source_count: int = 2,
    latest_source_at: str = SOURCE_AT,
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return make_question_entry(
        question,
        category,
        category_label,
        stage=stage,
        aliases=aliases,
        source_count=source_count,
        latest_source_at=latest_source_at,
        sources=sources,
        follow_ups=follow_ups or [],
    )


def make_card(
    question: str,
    category: str,
    category_label: str,
    *,
    stage: str = "project",
    status: str = "grounded",
    answer: str = "先讲核心目标，再讲链路、权衡和指标。",
    pitfalls: list[str] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    follow_ups: list[str] | None = None,
) -> dict[str, Any]:
    card = {
        "question": question,
        "category": category,
        "category_label": category_label,
        "stage": stage,
        "status": status,
        "answer": answer,
        "pitfalls": pitfalls or ["只讲概念，不讲线上权衡。"],
        "evidence": evidence
        or [
            {
                "kind": "web_page",
                "title": "RAG guide",
                "url": "https://example.com",
                "note": "pipeline",
            }
        ],
    }
    if follow_ups is not None:
        card["follow_ups"] = follow_ups
    return card


def make_category(
    name: str,
    label: str,
    *,
    stage: str,
    clusters: list[dict[str, Any]] | None = None,
    questions: list[dict[str, Any]] | None = None,
    cards: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    category = {
        "name": name,
        "label": label,
        "stage": stage,
    }
    if clusters is not None:
        category["clusters"] = clusters
    if questions is not None:
        category["questions"] = questions
    if cards is not None:
        category["cards"] = cards
    return category


def make_bank(*categories: dict[str, Any], source_dir: str = "test", generated_at: str = GENERATED_AT) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "source_dir": source_dir,
        "categories": list(categories),
    }


def make_answer_card_bundle(
    *categories: dict[str, Any],
    generated_at: str = GENERATED_AT,
    source_bank_generated_at: str | None = None,
) -> dict[str, Any]:
    bundle = {
        "generated_at": generated_at,
        "categories": list(categories),
    }
    if source_bank_generated_at is not None:
        bundle["source_bank_generated_at"] = source_bank_generated_at
    return bundle


def make_runner(bank: dict[str, Any] | None = None) -> InterviewRunner:
    return InterviewRunner(
        planner=InterviewPlanner(bank=bank or make_bank()),
        tracker=WeaknessTracker(),
    )


def make_planned_question(question: str, category: str, *, stage: str = "project") -> PlannedQuestion:
    return PlannedQuestion(
        stage=stage,
        stage_label=stage,
        category=category,
        category_label=category,
        question=question,
        follow_ups=[],
        source_count=1,
    )
