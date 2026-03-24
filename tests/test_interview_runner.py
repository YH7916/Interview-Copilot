from __future__ import annotations

import asyncio
from unittest.mock import patch

import copilot.interview.planner as planner_module

from copilot.interview.interviewer import LLMInterviewer
from copilot.interview.orchestrator import InterviewHarness, InterviewRunner, extract_question_answer_pairs
from copilot.interview.policy import decide_next_action
from copilot.interview.session import InterviewSession
from copilot.profile.review import ReviewAgent
from copilot.profile.store import WeaknessTracker
from tests.interview_fixtures import (
    make_answer_card_bundle,
    make_bank,
    make_card,
    make_category,
    make_cluster,
    make_log_path,
    make_source,
    make_trace_dir,
)

RAG_LABEL = "RAG 与检索链路"
RAG_QUESTION = "RAG召回链路怎么设计？"


def _sample_bank() -> dict:
    return make_bank(
        make_category(
            "rag_retrieval",
            RAG_LABEL,
            stage="project",
            clusters=[
                make_cluster(
                    RAG_QUESTION,
                    "rag_retrieval",
                    RAG_LABEL,
                    follow_ups=["embedding 召回有哪些方案？"],
                    sources=[make_source(title="字节一面")],
                )
            ],
            questions=[],
        )
    )


def _opening_bank() -> dict:
    return make_bank(
        make_category(
            "opening",
            "开场介绍",
            stage="opening",
            clusters=[
                make_cluster(
                    "自我介绍",
                    "opening",
                    "开场介绍",
                    stage="opening",
                    aliases=["介绍一下你自己"],
                    follow_ups=["方便补充一下毕业时间。"],
                    sources=[make_source(title="Agent 一面", source_url="opening", source_path="opening.md")],
                )
            ],
            questions=[],
        )
    )


def _dispatch_bank() -> dict:
    return make_bank(
        make_category(
            "project_architecture",
            "项目架构与链路",
            stage="project",
            clusters=[
                make_cluster(
                    "调度策略是如何设计的",
                    "project_architecture",
                    "项目架构与链路",
                    aliases=["你们的调度策略如何设计"],
                    follow_ups=["是否有异常 fallback 策略"],
                    sources=[make_source(title="Agent 二面", source_url="dispatch", source_path="dispatch.md")],
                )
            ],
            questions=[],
        )
    )


def _mixed_bank() -> dict:
    return make_bank(
        make_category(
            "rag_retrieval",
            RAG_LABEL,
            stage="project",
            clusters=[
                make_cluster(
                    RAG_QUESTION,
                    "rag_retrieval",
                    RAG_LABEL,
                    follow_ups=["embedding 召回有哪些方案？"],
                    sources=[make_source(title="字节一面")],
                ),
                make_cluster(
                    "而是让 Workflow 控制主干流程",
                    "rag_retrieval",
                    RAG_LABEL,
                    source_count=3,
                    sources=[make_source(title="字节二面", source_url="b", source_path="b.md")],
                ),
            ],
            questions=[],
        )
    )


def _sample_cards() -> dict:
    return make_answer_card_bundle(
        make_category(
            "rag_retrieval",
            RAG_LABEL,
            stage="project",
            cards=[
                make_card(
                    RAG_QUESTION,
                    "rag_retrieval",
                    RAG_LABEL,
                    answer="先讲召回、重排、生成，再讲指标和降级策略。",
                    pitfalls=["只讲概念，不讲链路指标。"],
                )
            ],
        )
    )


def test_extract_question_answer_pairs() -> None:
    pairs = extract_question_answer_pairs(
        [
            {"role": "assistant", "content": "请介绍一下你的项目。\nRAG召回链路怎么设计？"},
            {"role": "user", "content": "先做 embedding 召回，再做 rerank。"},
        ]
    )

    assert pairs == [{"question": RAG_QUESTION, "answer": "先做 embedding 召回，再做 rerank。"}]


def test_interview_runner_reviews_answers(monkeypatch) -> None:
    monkeypatch.setattr(planner_module, "load_answer_cards_or_empty", lambda recent=False: _sample_cards())

    runner = InterviewRunner(
        planner=planner_module.InterviewPlanner(bank=_sample_bank()),
        tracker=WeaknessTracker(log_path=make_log_path()),
    )
    plan = runner.plan(InterviewSession(user_id="u1", focus_topics=["RAG"]), max_questions=1)

    with patch(
        "copilot.interview.evaluation.call_text",
        return_value='{"accuracy_score":4,"clarity_score":4,"depth_score":3,"evidence_score":3,"structure_score":4,"reason":"主线基本清晰，但指标不够具体。"}',
    ):
        report = runner.review_answers(plan, ["先做 embedding 召回，再做 rerank。"], persist=False)

    assert report["count"] == 1
    assert report["results"][0]["accuracy_score"] == 4
    assert report["drill_plan"]
    assert "## Next Drill" in report["summary"]
    assert "指标不够具体" in report["summary"]


def test_interview_runner_plan_filters_non_question_fragments(monkeypatch) -> None:
    monkeypatch.setattr(planner_module, "load_answer_cards_or_empty", lambda recent=False: _sample_cards())

    runner = InterviewRunner(
        planner=planner_module.InterviewPlanner(bank=_mixed_bank()),
        tracker=WeaknessTracker(log_path=make_log_path()),
    )
    plan = runner.plan(InterviewSession(user_id="u1", focus_topics=["RAG"]), max_questions=3)

    assert all(item.question != "而是让 Workflow 控制主干流程" for item in plan)


def test_review_agent_uses_structured_review(monkeypatch) -> None:
    tracker = WeaknessTracker(log_path=make_log_path())
    monkeypatch.setattr(
        "copilot.interview.orchestrator.InterviewPlanner",
        lambda recent=False: planner_module.InterviewPlanner(bank=_sample_bank()),
    )
    monkeypatch.setattr(
        "copilot.interview.orchestrator.InterviewRunner.review_messages",
        lambda self, messages, persist=True: {
            "count": 1,
            "average_overall": 3.5,
            "results": [
                {
                    "question": RAG_QUESTION,
                    "accuracy_score": 3,
                    "reason": "缺少指标",
                    "overall_score": 3.5,
                }
            ],
            "summary": "# Interview Review\n\n1. RAG召回链路怎么设计？\n   - overall: 3.5/5\n   - accuracy: 3/5\n   - reason: 缺少指标",
        },
    )

    summary = asyncio.run(
        ReviewAgent(tracker=tracker).analyze(
            [
                {"role": "assistant", "content": RAG_QUESTION},
                {"role": "user", "content": "我会先做召回。"},
            ]
        )
    )

    assert "Interview Review" in summary


def test_interview_runner_review_messages_falls_back_without_answer_card(monkeypatch) -> None:
    monkeypatch.setattr(planner_module, "load_answer_cards_or_empty", lambda recent=False: {"generated_at": "", "categories": []})
    runner = InterviewRunner(
        planner=planner_module.InterviewPlanner(bank=_sample_bank()),
        tracker=WeaknessTracker(log_path=make_log_path()),
    )

    with patch(
        "copilot.interview.evaluation.call_text",
        return_value='{"accuracy_score":3,"clarity_score":3,"depth_score":2,"evidence_score":2,"structure_score":3,"reason":"能给出主线，但细节不够。"}',
    ):
        report = runner.review_messages(
            [
                {"role": "assistant", "content": RAG_QUESTION},
                {"role": "user", "content": "先做召回，再做重排。"},
            ],
            persist=False,
        )

    assert report["count"] == 1
    assert report["results"][0]["answer_status"] == "missing"
    assert report["drill_plan"]


def test_interview_harness_asks_natural_follow_up_for_short_answer(monkeypatch) -> None:
    monkeypatch.setattr(planner_module, "load_answer_cards_or_empty", lambda recent=False: _sample_cards())
    runner = InterviewRunner(
        planner=planner_module.InterviewPlanner(bank=_sample_bank()),
        tracker=WeaknessTracker(log_path=make_log_path()),
    )
    harness = InterviewHarness(
        runner=runner,
        session=InterviewSession(user_id="u1", focus_topics=["RAG"]),
        max_questions=1,
        topic="RAG",
        trace_dir=make_trace_dir(),
        interviewer=LLMInterviewer(enabled=False),
    )

    with patch(
        "copilot.interview.evaluation.call_text",
        return_value='{"accuracy_score":2,"clarity_score":3,"depth_score":2,"evidence_score":1,"structure_score":3,"reason":"太浅了。"}',
    ):
        opening = harness.open()
        follow_up = harness.reply("先召回。")
        finish = harness.reply("我会补充 embedding、rerank 和指标。")

    assert "# Live Interview" in opening
    assert "追问" in follow_up
    assert "reason:" not in follow_up
    assert "展开" in follow_up or "场景" in follow_up
    assert "Interview finished." in finish
    assert "- trace:" in finish
    assert harness.completed is True
    assert harness.trace_path is not None
    assert harness.trace_path.exists()
    content = harness.trace_path.read_text(encoding="utf-8")
    assert '"policy_reason": "need_deeper_reasoning"' in content


def test_policy_returns_contextual_reason() -> None:
    decision = decide_next_action(
        has_follow_ups=True,
        answer_text="先做召回。",
        depth_score=2,
        evidence_score=1,
        overall_score=2.0,
    )

    assert decision.action == "follow_up"
    assert decision.reason == "need_deeper_reasoning"


def test_opening_follow_up_only_asks_for_missing_graduation(monkeypatch) -> None:
    monkeypatch.setattr(planner_module, "load_answer_cards_or_empty", lambda recent=False: {"generated_at": "", "categories": []})
    runner = InterviewRunner(
        planner=planner_module.InterviewPlanner(bank=_opening_bank()),
        tracker=WeaknessTracker(log_path=make_log_path()),
    )
    harness = InterviewHarness(
        runner=runner,
        session=InterviewSession(user_id="u1"),
        max_questions=1,
        topic="opening",
        trace_dir=make_trace_dir(),
        interviewer=LLMInterviewer(enabled=False),
    )

    with patch(
        "copilot.interview.evaluation.call_text",
        return_value='{"accuracy_score":4,"clarity_score":4,"depth_score":4,"evidence_score":3,"structure_score":4,"reason":"信息比较完整。"}',
    ):
        follow_up = harness.reply("我是浙江大学计算机科学与技术学院大二学生，做过几个 AI Agent 项目。")

    assert "毕业时间" in follow_up
    assert "学校" not in follow_up


def test_interview_harness_clarifies_when_candidate_asks_back(monkeypatch) -> None:
    monkeypatch.setattr(planner_module, "load_answer_cards_or_empty", lambda recent=False: {"generated_at": "", "categories": []})
    runner = InterviewRunner(
        planner=planner_module.InterviewPlanner(bank=_dispatch_bank()),
        tracker=WeaknessTracker(log_path=make_log_path()),
    )
    harness = InterviewHarness(
        runner=runner,
        session=InterviewSession(user_id="u1"),
        max_questions=1,
        topic="dispatch",
        trace_dir=make_trace_dir(),
        interviewer=LLMInterviewer(enabled=False),
    )

    reply = harness.reply("您指的是哪个项目？")

    assert "最熟悉" in reply
    assert "优先级" in reply
    assert harness.current_index == 0
    assert harness.session.turns == 0
