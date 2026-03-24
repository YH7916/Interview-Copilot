from __future__ import annotations

from unittest.mock import patch

from copilot.interview.interviewer import LLMInterviewer
from copilot.interview.orchestrator import InterviewTraceEntry, ReviewedQuestion
from copilot.interview.planner import PlannedQuestion
from copilot.interview.selector import LLMQuestionSelector
from copilot.interview.session import InterviewSession
from copilot.interview.state import InterviewGoalState, build_goal_state


def _reviewed_entry(
    *,
    category: str,
    answer: str,
    score: float,
    policy_reason: str = "",
) -> InterviewTraceEntry:
    return InterviewTraceEntry(
        question="示例问题",
        answer=answer,
        policy_reason=policy_reason,
        review=ReviewedQuestion(
            question="示例问题",
            category=category,
            stage="project",
            answer_status="grounded",
            candidate_answer=answer,
            reference_answer="",
            pitfalls=[],
            accuracy_score=3,
            clarity_score=3,
            depth_score=3,
            evidence_score=2,
            structure_score=3,
            overall_score=score,
            reason="细节不足",
        ),
    )


def test_build_goal_state_tracks_authenticity_and_focus() -> None:
    session = InterviewSession(
        user_id="u1",
        candidate_profile="Built a multi-agent RAG app with workflow and evaluation.",
        focus_topics=["agent"],
    )
    trace = [
        _reviewed_entry(
            category="agent_architecture",
            answer="我做了一个 agent 项目，负责记忆模块，但还没给出线上指标。",
            score=3.0,
            policy_reason="need_concrete_evidence",
        )
    ]

    goal_state = build_goal_state(session=session, trace=trace, pending_policy_reason="")

    assert goal_state.authenticity_status == "verified"
    assert goal_state.evidence_status in {"weak", "partial"}
    assert "agent_architecture" in goal_state.covered_categories
    assert "project_evaluation" in goal_state.recommended_focus
    assert "rag_retrieval" in goal_state.profile_signals


def test_selector_fallback_uses_goal_state_focus() -> None:
    selector = LLMQuestionSelector(enabled=False)
    candidates = [
        PlannedQuestion("project", "project", "python_system", "python_system", "Python GIL 为什么存在？", [], 1),
        PlannedQuestion("project", "project", "rag_retrieval", "rag_retrieval", "RAG 召回链路怎么设计？", [], 1),
    ]
    goal_state = InterviewGoalState(recommended_focus=["rag_retrieval"])

    selected = selector.select_next_question(
        session=InterviewSession(user_id="u1"),
        candidates=candidates,
        goal_state=goal_state,
        history=[],
    )

    assert selected == 1


def test_interviewer_prompt_contains_goal_state() -> None:
    interviewer = LLMInterviewer(enabled=True)
    item = PlannedQuestion(
        stage="project",
        stage_label="project",
        category="agent_architecture",
        category_label="agent_architecture",
        question="你是怎么设计 agent 的记忆系统？",
        follow_ups=["记忆写入和召回的策略是什么"],
        source_count=2,
    )
    goal_state = InterviewGoalState(
        covered_categories=["opening"],
        unresolved_points=["need_execution_detail"],
        recommended_focus=["project_architecture"],
        authenticity_status="partial",
        evidence_status="weak",
    )

    def fake_call(prompt: str, **_: object) -> str:
        assert '"goal_state"' in prompt
        assert "need_execution_detail" in prompt
        return '{"prompt":"结合你刚才提到的项目，讲一下记忆写入和召回策略。"}'

    with patch("copilot.interview.interviewer.call_text", side_effect=fake_call):
        rendered = interviewer.render_follow_up(
            item=item,
            session=InterviewSession(user_id="u1", candidate_profile="multi-agent memory"),
            candidate_answer="我做了短期和长期记忆。",
            policy_reason="need_execution_detail",
            goal_state=goal_state,
            fallback="fallback",
            history=[],
        )

    assert "记忆写入和召回" in rendered
