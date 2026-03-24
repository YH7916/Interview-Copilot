from __future__ import annotations

from unittest.mock import patch

from copilot.interview.interviewer import LLMInterviewer
from copilot.interview.orchestrator import InterviewHarness, InterviewRunner
from copilot.interview.planner import InterviewPlanner
from copilot.interview.session import InterviewSession
from copilot.profile.store import WeaknessTracker
from tests.interview_fixtures import make_bank, make_category, make_cluster, make_source

AGENT_LABEL = "Agent 架构与技能"
AGENT_QUESTION = "你是怎么设计 agent 的记忆系统"


def _sample_bank() -> dict:
    return make_bank(
        make_category(
            "agent_architecture",
            AGENT_LABEL,
            stage="project",
            clusters=[
                make_cluster(
                    AGENT_QUESTION,
                    "agent_architecture",
                    AGENT_LABEL,
                    aliases=["记忆系统怎么设计"],
                    follow_ups=["记忆写入和召回的策略是什么"],
                    source_count=3,
                    sources=[make_source(title="Agent 一面", source_url="mem", source_path="mem.md")],
                )
            ],
            questions=[],
        )
    )


def test_llm_interviewer_rewrites_question_and_follow_up() -> None:
    runner = InterviewRunner(
        planner=InterviewPlanner(bank=_sample_bank()),
        tracker=WeaknessTracker(),
    )
    interviewer = LLMInterviewer(enabled=True)
    harness = InterviewHarness(
        runner=runner,
        session=InterviewSession(
            user_id="u1",
            candidate_profile="浙江大学大二，做过多 agent 和记忆系统项目。",
            focus_topics=["agent"],
        ),
        max_questions=1,
        topic="agent",
        interviewer=interviewer,
    )

    def fake_interviewer_call(prompt: str, **_: object) -> str:
        if '"task": "render_question"' in prompt:
            return '{"prompt":"先结合你做过的 agent 项目，讲讲记忆系统是怎么设计的？"}'
        if '"task": "render_follow_up"' in prompt:
            return '{"prompt":"你刚才提到短期和长期记忆，那写入、更新和召回分别在什么时机触发？"}'
        return '{"prompt":"可以结合你最熟悉的项目来回答。"}'

    with (
        patch("copilot.interview.interviewer.call_text", side_effect=fake_interviewer_call),
        patch(
            "copilot.interview.evaluation.call_text",
            return_value='{"accuracy_score":3,"clarity_score":3,"depth_score":2,"evidence_score":2,"structure_score":3,"reason":"细节不够。"}',
        ),
    ):
        opening = harness.open()
        follow_up = harness.reply("我是用短期和长期相结合。")

    assert "先结合你做过的 agent 项目" in opening
    assert "写入、更新和召回" in follow_up


def test_llm_interviewer_falls_back_when_generation_fails() -> None:
    runner = InterviewRunner(
        planner=InterviewPlanner(bank=_sample_bank()),
        tracker=WeaknessTracker(),
    )
    interviewer = LLMInterviewer(enabled=True)
    harness = InterviewHarness(
        runner=runner,
        session=InterviewSession(user_id="u1"),
        max_questions=1,
        topic="agent",
        interviewer=interviewer,
    )

    with patch("copilot.interview.interviewer.call_text", side_effect=RuntimeError("llm down")):
        opening = harness.open()

    assert AGENT_QUESTION in opening
