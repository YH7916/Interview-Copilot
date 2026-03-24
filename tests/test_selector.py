from __future__ import annotations

from unittest.mock import patch

from copilot.interview.interviewer import LLMInterviewer
from copilot.interview.orchestrator import InterviewHarness
from copilot.interview.selector import LLMQuestionSelector
from copilot.interview.session import InterviewSession
from tests.interview_fixtures import make_planned_question, make_runner


def test_harness_preserves_opening_as_first_question() -> None:
    harness = InterviewHarness(
        runner=make_runner(),
        session=InterviewSession(user_id="u1", focus_topics=["agent"]),
        max_questions=3,
        topic="agent",
        interviewer=LLMInterviewer(enabled=False),
        selector=LLMQuestionSelector(enabled=True),
    )
    harness.plan = [
        make_planned_question("自我介绍", "opening", stage="opening"),
        make_planned_question("RAG 召回链路怎么设计？", "rag_retrieval"),
        make_planned_question("你是怎么设计 agent 记忆系统的？", "agent_architecture"),
    ]
    harness.completed = False

    with patch(
        "copilot.interview.selector.call_text",
        return_value='{"selected_index": 2, "reason":"candidate has agent-heavy resume"}',
    ):
        opening = harness.open()

    assert "自我介绍" in opening


def test_selector_fallback_prefers_category_change() -> None:
    selector = LLMQuestionSelector(enabled=True)
    candidates = [
        make_planned_question("再问一个记忆细节", "agent_architecture"),
        make_planned_question("RAG 召回链路怎么设计？", "rag_retrieval"),
        make_planned_question("Python GIL 为什么存在？", "python_system"),
    ]
    history = [{"question": "你是怎么设计 agent 的记忆系统？", "category": "agent_architecture"}]

    with patch("copilot.interview.selector.call_text", side_effect=RuntimeError("selector down")):
        chosen = selector.select_next_question(
            session=InterviewSession(user_id="u1", focus_topics=["agent"]),
            candidates=candidates,
            history=history,
        )

    assert chosen == 1
