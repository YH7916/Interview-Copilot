"""Thin application boundary for copilot business services."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from copilot.interview import InterviewHarness, InterviewPlanner, InterviewRunner, InterviewSession
from copilot.interview.evaluation import render_review_summary_with_drill
from copilot.interview.planner import render_plan
from copilot.knowledge.overview import (
    build_recent_reports_overview,
    render_recent_reports_overview as _render_recent_reports_overview,
)
from copilot.knowledge.pipeline import rebuild_knowledge
from copilot.prep import render_interview_prep_pack as _render_interview_prep_pack
from copilot.profile import ReviewAgent, WeaknessTracker, load_candidate_profile
from copilot.sources.nowcoder import NowcoderInterviewIngestor
from copilot.workflows import (
    build_agent_workflow,
    build_daily_digest,
    render_daily_digest_markdown,
    run_agent_workflow as _run_agent_workflow,
)


def build_nowcoder_ingestor(
    *,
    fetch_timeout_seconds: float = 12.0,
    updated_within_days: int = 30,
) -> NowcoderInterviewIngestor:
    return NowcoderInterviewIngestor(
        fetch_timeout_seconds=fetch_timeout_seconds,
        updated_within_days=updated_within_days,
    )


async def collect_nowcoder_interviews(
    *,
    query: str | None = None,
    count_per_query: int = 6,
    max_reports: int = 8,
    dry_run: bool = False,
    fetch_timeout: float = 12.0,
    updated_within_days: int = 30,
    rebuild_index: bool = True,
) -> dict:
    ingestor = build_nowcoder_ingestor(
        fetch_timeout_seconds=fetch_timeout,
        updated_within_days=updated_within_days,
    )
    return await ingestor.run(
        queries=[query] if query else None,
        count_per_query=count_per_query,
        max_reports=max_reports,
        dry_run=dry_run,
        rebuild_index=rebuild_index,
    )


def build_weakness_tracker(log_path: Path | None = None) -> WeaknessTracker:
    if log_path is None:
        return WeaknessTracker()
    return WeaknessTracker(log_path=log_path)


def build_review_agent(
    *,
    tracker: WeaknessTracker | None = None,
    log_path: Path | None = None,
) -> ReviewAgent:
    if tracker is not None:
        return ReviewAgent(tracker=tracker)
    if log_path is not None:
        return ReviewAgent(tracker=build_weakness_tracker(log_path))
    return ReviewAgent()


def trigger_background_review(
    messages: list[dict],
    *,
    tracker: WeaknessTracker | None = None,
    log_path: Path | None = None,
) -> ReviewAgent:
    agent = build_review_agent(tracker=tracker, log_path=log_path)
    agent.analyze_background(list(messages))
    return agent


async def rebuild_local_knowledge(*, with_web: bool = False, max_cards: int | None = None) -> dict:
    return await rebuild_knowledge(with_web=with_web, max_cards=max_cards)


def get_daily_digest(*, days: int = 1) -> dict:
    return build_daily_digest(days=days)


def render_daily_digest(*, days: int = 1) -> str:
    return render_daily_digest_markdown(build_daily_digest(days=days))


def get_recent_reports_overview(*, days: int = 7, limit: int = 10) -> dict:
    return build_recent_reports_overview(days=days, limit=limit)


def render_recent_reports_overview(*, days: int = 7, limit: int = 10) -> str:
    return _render_recent_reports_overview(days=days, limit=limit)


def render_interview_prep_pack(
    *,
    user_id: str = "nanobot",
    topic: str = "",
    company: str = "",
    position: str = "",
    target_description: str = "",
    max_questions: int = 8,
    recent: bool = True,
    candidate_profile: str = "",
    candidate_profile_path: str = "",
) -> str:
    return _render_interview_prep_pack(
        user_id=user_id,
        topic=topic,
        company=company,
        position=position,
        target_description=target_description,
        max_questions=max_questions,
        recent=recent,
        candidate_profile=candidate_profile,
        candidate_profile_path=candidate_profile_path,
    )


def get_mock_interview_plan(
    *,
    topic: str = "",
    max_questions: int = 6,
    recent: bool = True,
    candidate_profile: str = "",
    candidate_profile_path: str = "",
    interview_style: str = "auto",
) -> list[dict]:
    session = InterviewSession(
        user_id="nanobot",
        focus_topics=[topic] if topic else [],
        candidate_profile=load_candidate_profile(
            profile_text=candidate_profile,
            profile_path=candidate_profile_path,
        ),
        interview_style=interview_style,
    )
    plan = InterviewPlanner(recent=recent).plan(session, max_questions=max_questions)
    return [asdict(item) for item in plan]


def render_mock_interview_plan(
    *,
    topic: str = "",
    max_questions: int = 6,
    recent: bool = True,
    candidate_profile: str = "",
    candidate_profile_path: str = "",
    interview_style: str = "auto",
) -> str:
    session = InterviewSession(
        user_id="nanobot",
        focus_topics=[topic] if topic else [],
        candidate_profile=load_candidate_profile(
            profile_text=candidate_profile,
            profile_path=candidate_profile_path,
        ),
        interview_style=interview_style,
    )
    plan = InterviewPlanner(recent=recent).plan(session, max_questions=max_questions)
    return render_plan(plan)


def build_live_interview(
    *,
    user_id: str = "nanobot",
    topic: str = "",
    max_questions: int = 6,
    recent: bool = True,
    candidate_profile: str = "",
    candidate_profile_path: str = "",
    interview_style: str = "auto",
) -> InterviewHarness:
    session = InterviewSession(
        user_id=user_id,
        focus_topics=[topic] if topic else [],
        candidate_profile=load_candidate_profile(
            profile_text=candidate_profile,
            profile_path=candidate_profile_path,
        ),
        interview_style=interview_style,
    )
    return InterviewHarness(
        runner=InterviewRunner(recent=recent),
        session=session,
        max_questions=max_questions,
        topic=topic,
    )


def review_interview_messages(messages: list[dict], *, persist: bool = False) -> dict:
    return InterviewRunner(recent=True).review_messages(messages, persist=persist)


def render_interview_review(messages: list[dict], *, persist: bool = False) -> str:
    report = review_interview_messages(messages, persist=persist)
    if report["count"] == 0:
        return "No interview question-answer pairs found yet. Start with /interview and answer a few questions first."
    return report.get("summary") or render_review_summary_with_drill(
        report["results"],
        drill_plan=report.get("drill_plan"),
    )


def render_assistant_menu() -> str:
    return "\n".join(
        [
            "Interview Copilot",
            "",
            "Available actions:",
            "- /ingest [days]  Collect recent Nowcoder interview posts",
            "- /recent [days]  Show recently kept materials",
            "- /digest [days]  Show the local daily digest",
            "- /prep [topic] [--resume path] [--company name] [--position role] [--target jd]  Build an interview prep pack",
            "- /interview [topic] [--resume path] [--style auto|coding-first|no-coding]  Start a live mock interview",
            "- /review  Review the current interview session",
            "- /schedule_digest  Schedule a 9:00 daily digest",
            "- /restart  Restart the local runtime",
            "- /help  Show this menu",
            "",
            "Examples:",
            "- /ingest 7",
            "- /recent 7",
            "- /digest 1",
            '- /prep agent --resume "D:\\\\resume\\\\CV.typ" --company ByteDance --position "AI Agent Intern" --target "Build agent workflows, retrieval, evaluation"',
            "- /interview agent",
            '- /interview agent --resume "D:\\\\resume\\\\CV.typ"',
            "- /interview agent --style coding-first",
            "- answer directly after /interview",
            "- /review",
            "- /schedule_digest",
            "- /restart",
        ]
    )


def get_agent_workflow() -> list[dict]:
    return build_agent_workflow()


async def run_agent_workflow(
    *,
    updated_within_days: int = 7,
    count_per_query: int = 20,
    max_reports: int = 50,
    fetch_timeout: float = 12.0,
    with_web: bool = False,
    max_cards: int | None = None,
    dry_run: bool = False,
) -> dict:
    return await _run_agent_workflow(
        collect_fn=collect_nowcoder_interviews,
        updated_within_days=updated_within_days,
        count_per_query=count_per_query,
        max_reports=max_reports,
        fetch_timeout=fetch_timeout,
        with_web=with_web,
        max_cards=max_cards,
        dry_run=dry_run,
    )


__all__ = [
    "build_nowcoder_ingestor",
    "build_live_interview",
    "collect_nowcoder_interviews",
    "build_review_agent",
    "build_weakness_tracker",
    "get_agent_workflow",
    "get_daily_digest",
    "get_mock_interview_plan",
    "get_recent_reports_overview",
    "rebuild_local_knowledge",
    "render_interview_review",
    "render_interview_prep_pack",
    "render_assistant_menu",
    "render_daily_digest",
    "render_mock_interview_plan",
    "render_recent_reports_overview",
    "review_interview_messages",
    "run_agent_workflow",
    "trigger_background_review",
]
