from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from copilot.profile import build_candidate_profile_summary, normalize_candidate_profile
from copilot.interview.planner import InterviewPlanner
from copilot.interview.session import InterviewSession
from copilot.knowledge.question_bank import build_question_bank
from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus


def test_get_mock_interview_plan_accepts_candidate_profile(monkeypatch) -> None:
    planner = MagicMock()
    planner.plan.return_value = []
    monkeypatch.setattr("copilot.app.InterviewPlanner", lambda recent=True: planner)

    from copilot.app import get_mock_interview_plan

    get_mock_interview_plan(candidate_profile="Built a multi-agent RAG app with rerank.")

    session = planner.plan.call_args.args[0]
    assert "multi-agent" in session.candidate_profile


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


def test_build_candidate_profile_summary_extracts_structured_snapshot() -> None:
    text = """
    = 教育背景
    浙江大学 计算机科学与技术学院 本科 2028年毕业
    = 项目经历
    Multi-Agent RAG Copilot
    负责 multi-agent 协作、BM25 + rerank、记忆系统和评测闭环
    = 技能
    Python React LangChain WebWorker
    """

    summary = build_candidate_profile_summary(text)

    assert "Candidate Snapshot" in summary
    assert "Education:" in summary
    assert "Project 1:" in summary
    assert "Focus Areas:" in summary
    assert "multi-agent" in summary.lower()
    assert "python" in summary.lower()


def test_build_candidate_profile_summary_prefers_llm_project_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(
        "copilot.profile.extractor.get_text_settings",
        lambda task="analysis": {"api_key": "test-key"},
    )

    with patch(
        "copilot.profile.extractor.call_text",
        return_value=(
            '{"education":"浙江大学 计算机科学与技术学院","stage":"本科大二，2028年毕业",'
            '"focus_areas":["multi-agent","RAG","memory"],'
            '"core_projects":['
            '{"name":"Interview Copilot","summary":"面向 AI Agent 岗位的模拟面试系统","candidate_ownership":"负责面试编排、记忆与追问策略","tech":["Python","RAG","Agent"],"deep_dive_points":["追问策略","记忆系统","动态选题"]},'
            '{"name":"CuteGo","summary":"围棋 AI 前端交互项目","candidate_ownership":"负责 WebWorker 与模型推理集成","tech":["WebWorker","KataGo","Frontend"],"deep_dive_points":["主线程拆分","性能优化"]}'
            '],'
            '"skills":["Python","React","Codex"]}'
        ),
    ):
        summary = build_candidate_profile_summary("这份简历主要有两个项目：Interview Copilot 和 CuteGo。")

    assert "Project 1: Interview Copilot" in summary
    assert "Project 2: CuteGo" in summary
    assert "Deep Dive: 追问策略" in summary


def test_normalize_candidate_profile_prefers_snapshot_for_resume_like_text() -> None:
    profile = normalize_candidate_profile(
        """
        教育背景
        浙江大学 计算机科学与技术学院 大二
        项目经历
        Agent Memory System
        我负责长期记忆写入和召回策略
        """
    )

    assert profile.startswith("Candidate Snapshot")
    assert "Project 1:" in profile


def test_normalize_candidate_profile_keeps_plain_text_signal() -> None:
    profile = normalize_candidate_profile("Built a multi-agent RAG app with rerank.")

    assert "multi-agent" in profile.lower()
    assert len(profile) > 0


def test_interview_planner_uses_candidate_profile_focus():
    bank = build_question_bank(
        [
            {
                "title": "Agent interview",
                "source_url": "https://example.com/1",
                "source_path": "a.md",
                "captured_at": "2026-03-19T10:00:00",
                "questions": [
                    "自我介绍",
                    "你们的多 Agent 协作是怎么设计的？",
                    "RAG 召回链路怎么设计？",
                    "Python 为什么有 GIL？",
                ],
            }
        ]
    )

    plan = InterviewPlanner(bank=bank).plan(
        InterviewSession(
            user_id="u4",
            candidate_profile="Built a multi-agent RAG app with BM25 and rerank.",
        ),
        max_questions=4,
    )

    categories = {item.category for item in plan}
    assert "agent_architecture" in categories
    assert "rag_retrieval" in categories


@pytest.mark.asyncio
async def test_agent_loop_passes_resume_path_to_live_interview() -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    fake_session = SimpleNamespace(messages=[], last_consolidated=0, key="cli:c1")
    fake_sessions = MagicMock()
    fake_sessions.get_or_create.return_value = fake_session

    with (
        patch("nanobot.agent.loop.ContextBuilder", return_value=MagicMock()),
        patch("nanobot.agent.loop.SessionManager", return_value=fake_sessions),
        patch("nanobot.agent.loop.SubagentManager", return_value=MagicMock()),
        patch("nanobot.agent.loop.MemoryConsolidator", return_value=MagicMock()),
    ):
        loop = AgentLoop(bus=bus, provider=provider, workspace=Path.cwd(), model="test-model")

    interview = MagicMock()
    interview.open.return_value = "start"

    with patch("copilot.app.build_live_interview", return_value=interview) as build_mock:
        response = await loop._process_message(
            InboundMessage(
                channel="cli",
                sender_id="u1",
                chat_id="c1",
                content='/interview agent --resume "D:\\Desktop\\Chinese-Resume-in-Typst-main\\CV.typ"',
            )
        )

    assert response.content == "start"
    assert build_mock.call_args.kwargs["candidate_profile_path"].endswith("CV.typ")


@pytest.mark.asyncio
async def test_agent_loop_passes_interview_style_to_live_interview() -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    fake_session = SimpleNamespace(messages=[], last_consolidated=0, key="cli:c1")
    fake_sessions = MagicMock()
    fake_sessions.get_or_create.return_value = fake_session

    with (
        patch("nanobot.agent.loop.ContextBuilder", return_value=MagicMock()),
        patch("nanobot.agent.loop.SessionManager", return_value=fake_sessions),
        patch("nanobot.agent.loop.SubagentManager", return_value=MagicMock()),
        patch("nanobot.agent.loop.MemoryConsolidator", return_value=MagicMock()),
    ):
        loop = AgentLoop(bus=bus, provider=provider, workspace=Path.cwd(), model="test-model")

    interview = MagicMock()
    interview.open.return_value = "start"

    with patch("copilot.app.build_live_interview", return_value=interview) as build_mock:
        response = await loop._process_message(
            InboundMessage(
                channel="cli",
                sender_id="u1",
                chat_id="c1",
                content='/interview agent --style coding-first --resume "D:\\Desktop\\Chinese-Resume-in-Typst-main\\CV.typ"',
            )
        )

    assert response.content == "start"
    assert build_mock.call_args.kwargs["interview_style"] == "coding-first"
