from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from copilot.interview.planner import PlannedQuestion


@pytest.mark.asyncio
async def test_collect_nowcoder_interviews_uses_facade(monkeypatch) -> None:
    class _FakeIngestor:
        async def run(self, **kwargs):
            return kwargs

    monkeypatch.setattr(
        "copilot.app.build_nowcoder_ingestor",
        lambda **kwargs: _FakeIngestor(),
    )

    from copilot.app import collect_nowcoder_interviews

    result = await collect_nowcoder_interviews(
        query="agent 面经",
        count_per_query=3,
        max_reports=2,
        dry_run=True,
        fetch_timeout=8,
        updated_within_days=7,
        rebuild_index=False,
    )

    assert result == {
        "queries": ["agent 面经"],
        "count_per_query": 3,
        "max_reports": 2,
        "dry_run": True,
        "rebuild_index": False,
    }


def test_build_review_agent_accepts_custom_log_path() -> None:
    from copilot.app import build_review_agent

    log_path = Path(__file__).resolve().parents[1] / "data" / "_test_workspace" / "app" / "weakness.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    agent = build_review_agent(log_path=log_path)

    assert agent.tracker.log_path == log_path


def test_trigger_background_review_uses_agent_facade(monkeypatch) -> None:
    mock_agent = MagicMock()

    monkeypatch.setattr(
        "copilot.app.build_review_agent",
        lambda **kwargs: mock_agent,
    )

    from copilot.app import trigger_background_review

    result = trigger_background_review([{"role": "user", "content": "hello"}])

    assert result is mock_agent
    mock_agent.analyze_background.assert_called_once_with([{"role": "user", "content": "hello"}])


def test_get_daily_digest_uses_workflow_facade(monkeypatch) -> None:
    monkeypatch.setattr(
        "copilot.app.build_daily_digest",
        lambda **kwargs: {"days": kwargs["days"]},
    )

    from copilot.app import get_daily_digest

    assert get_daily_digest(days=3) == {"days": 3}


def test_render_daily_digest_uses_workflow_facade(monkeypatch) -> None:
    monkeypatch.setattr(
        "copilot.app.build_daily_digest",
        lambda **kwargs: {"days": kwargs["days"]},
    )
    monkeypatch.setattr(
        "copilot.app.render_daily_digest_markdown",
        lambda digest: f"digest:{digest['days']}",
    )

    from copilot.app import render_daily_digest

    assert render_daily_digest(days=2) == "digest:2"


def test_get_recent_reports_overview_uses_knowledge_facade(monkeypatch) -> None:
    monkeypatch.setattr(
        "copilot.app.build_recent_reports_overview",
        lambda **kwargs: {"days": kwargs["days"], "limit": kwargs["limit"]},
    )

    from copilot.app import get_recent_reports_overview

    assert get_recent_reports_overview(days=5, limit=3) == {"days": 5, "limit": 3}


def test_render_recent_reports_overview_uses_knowledge_facade(monkeypatch) -> None:
    monkeypatch.setattr(
        "copilot.app._render_recent_reports_overview",
        lambda **kwargs: f"recent:{kwargs['days']}:{kwargs['limit']}",
    )

    from copilot.app import render_recent_reports_overview

    assert render_recent_reports_overview(days=4, limit=2) == "recent:4:2"


def test_render_interview_review_uses_runner(monkeypatch) -> None:
    monkeypatch.setattr(
        "copilot.app.review_interview_messages",
        lambda messages, persist=False: {
            "count": 1,
            "results": [
                {
                    "question": "什么是 ReAct？",
                    "overall_score": 4.0,
                    "accuracy_score": 4,
                    "reason": "主线清楚。",
                }
            ],
        },
    )

    from copilot.app import render_interview_review

    result = render_interview_review([{"role": "assistant", "content": "什么是 ReAct？"}])

    assert "# Interview Review" in result
    assert "什么是 ReAct？" in result


def test_get_mock_interview_plan_uses_planner(monkeypatch) -> None:
    planner = MagicMock()
    planner.plan.return_value = [
        PlannedQuestion(
            stage="project",
            stage_label="项目",
            category="agent_architecture",
            category_label="Agent",
            question="什么是 ReAct？",
            follow_ups=[],
            source_count=2,
            latest_source_at="2026-03-20T10:00:00",
            answer_status="",
            reference_answer="",
            pitfalls=[],
            evidence=[],
        )
    ]
    monkeypatch.setattr("copilot.app.InterviewPlanner", lambda recent=True: planner)

    from copilot.app import get_mock_interview_plan

    plan = get_mock_interview_plan(topic="Agent", max_questions=4, recent=False)

    assert plan[0]["question"] == "什么是 ReAct？"
    planner.plan.assert_called_once()


def test_render_mock_interview_plan_uses_planner_and_renderer(monkeypatch) -> None:
    planner = MagicMock()
    planner.plan.return_value = ["planned"]
    monkeypatch.setattr("copilot.app.InterviewPlanner", lambda recent=True: planner)
    monkeypatch.setattr("copilot.app.render_plan", lambda plan: f"plan:{len(plan)}")

    from copilot.app import render_mock_interview_plan

    assert render_mock_interview_plan(topic="RAG", max_questions=5) == "plan:1"


def test_build_live_interview_uses_runner_boundary(monkeypatch) -> None:
    mock_runner = MagicMock()
    mock_harness = MagicMock()
    monkeypatch.setattr("copilot.app.InterviewRunner", lambda recent=True: mock_runner)
    monkeypatch.setattr("copilot.app.InterviewHarness", lambda runner, session, max_questions=6, topic="": mock_harness)

    from copilot.app import build_live_interview

    result = build_live_interview(user_id="u1", topic="Agent", max_questions=4, recent=False)

    assert result is mock_harness


def test_get_mock_interview_plan_accepts_candidate_profile(monkeypatch) -> None:
    planner = MagicMock()
    planner.plan.return_value = []
    monkeypatch.setattr("copilot.app.InterviewPlanner", lambda recent=True: planner)

    from copilot.app import get_mock_interview_plan

    get_mock_interview_plan(candidate_profile="Built a multi-agent RAG app with rerank.")

    session = planner.plan.call_args.args[0]
    assert "multi-agent" in session.candidate_profile


def test_get_mock_interview_plan_accepts_interview_style(monkeypatch) -> None:
    planner = MagicMock()
    planner.plan.return_value = []
    monkeypatch.setattr("copilot.app.InterviewPlanner", lambda recent=True: planner)

    from copilot.app import get_mock_interview_plan

    get_mock_interview_plan(interview_style="coding-first")

    session = planner.plan.call_args.args[0]
    assert session.interview_style == "coding-first"


def test_get_mock_interview_plan_loads_profile_from_file(monkeypatch, tmp_path) -> None:
    planner = MagicMock()
    planner.plan.return_value = []
    monkeypatch.setattr("copilot.app.InterviewPlanner", lambda recent=True: planner)

    resume = tmp_path / "resume.typ"
    resume.write_text("= Resume\n#show: project => multi-agent rag rerank", encoding="utf-8")

    from copilot.app import get_mock_interview_plan

    get_mock_interview_plan(candidate_profile_path=str(resume))

    session = planner.plan.call_args.args[0]
    assert "multi-agent" in session.candidate_profile


@pytest.mark.asyncio
async def test_run_agent_workflow_uses_app_boundary(monkeypatch) -> None:
    calls = {}

    async def _fake_workflow(**kwargs):
        calls.update(kwargs)
        return {"workflow": "daily_ops"}

    monkeypatch.setattr("copilot.app._run_agent_workflow", _fake_workflow)

    from copilot.app import run_agent_workflow

    result = await run_agent_workflow(
        updated_within_days=3,
        count_per_query=9,
        max_reports=12,
        fetch_timeout=7,
        with_web=True,
        max_cards=6,
        dry_run=True,
    )

    assert result == {"workflow": "daily_ops"}
    assert calls["collect_fn"].__name__ == "collect_nowcoder_interviews"
    assert calls["updated_within_days"] == 3
    assert calls["count_per_query"] == 9
    assert calls["max_reports"] == 12
    assert calls["fetch_timeout"] == 7
    assert calls["with_web"] is True
    assert calls["max_cards"] == 6
    assert calls["dry_run"] is True
