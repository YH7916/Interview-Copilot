from __future__ import annotations

from copilot.interview.orchestrator import InterviewTraceEntry, ReviewedQuestion
from copilot.interview.planner import PlannedQuestion
from copilot.interview.selector import LLMQuestionSelector
from copilot.interview.session import InterviewSession
from copilot.interview.state import build_goal_state
from copilot.profile import parse_candidate_projects


def _entry(question: str, answer: str, category: str = "agent_architecture") -> InterviewTraceEntry:
    return InterviewTraceEntry(
        question=question,
        answer=answer,
        review=ReviewedQuestion(
            question=question,
            category=category,
            stage="project",
            answer_status="grounded",
            candidate_answer=answer,
            reference_answer="",
            pitfalls=[],
            accuracy_score=4,
            clarity_score=4,
            depth_score=3,
            evidence_score=2,
            structure_score=4,
            overall_score=3.6,
            reason="ok",
        ),
    )


def test_parse_candidate_projects_from_snapshot() -> None:
    snapshot = "\n".join(
        [
            "Candidate Snapshot",
            "- Project 1: Interview Copilot | 面向 AI Agent 岗位的模拟面试系统 | Ownership: 负责面试编排、追问和记忆策略 | Tech: Python, RAG, Agent | Deep Dive: 追问策略, 动态选题",
            "- Project 2: CuteGo | 围棋 AI 前端交互项目 | Ownership: 负责 WebWorker 与模型推理集成 | Tech: WebWorker, KataGo, Frontend | Deep Dive: 主线程拆分, 性能优化",
        ]
    )

    projects = parse_candidate_projects(snapshot)

    assert len(projects) == 2
    assert projects[0]["name"] == "Interview Copilot"
    assert "动态选题" in projects[0]["deep_dive_points"]
    assert "WebWorker" in projects[1]["tech"]


def test_goal_state_tracks_active_project_from_trace() -> None:
    session = InterviewSession(
        user_id="u1",
        candidate_profile="\n".join(
            [
                "Candidate Snapshot",
                "- Project 1: Interview Copilot | 面向 AI Agent 岗位的模拟面试系统 | Ownership: 负责面试编排、追问和记忆策略 | Tech: Python, RAG, Agent | Deep Dive: 追问策略, 动态选题",
                "- Project 2: CuteGo | 围棋 AI 前端交互项目 | Ownership: 负责 WebWorker 与模型推理集成 | Tech: WebWorker, KataGo, Frontend | Deep Dive: 主线程拆分, 性能优化",
            ]
        ),
        focus_topics=["agent"],
    )

    goal_state = build_goal_state(
        session=session,
        trace=[_entry("你是怎么设计记忆系统的？", "在 Interview Copilot 里，我负责追问策略和记忆写入。")],
    )

    assert goal_state.active_project == "Interview Copilot"
    assert goal_state.project_focus_mode == "anchored"
    assert "Interview Copilot" in goal_state.discussed_projects


def test_selector_fallback_prefers_active_project_keywords() -> None:
    selector = LLMQuestionSelector(enabled=False)
    session = InterviewSession(
        user_id="u1",
        candidate_profile="\n".join(
            [
                "Candidate Snapshot",
                "- Project 1: Interview Copilot | 面向 AI Agent 岗位的模拟面试系统 | Ownership: 负责面试编排、追问和记忆策略 | Tech: Python, RAG, Agent | Deep Dive: 追问策略, 动态选题",
                "- Project 2: CuteGo | 围棋 AI 前端交互项目 | Ownership: 负责 WebWorker 与模型推理集成 | Tech: WebWorker, KataGo, Frontend | Deep Dive: 主线程拆分, 性能优化",
            ]
        ),
        focus_topics=["agent"],
    )
    goal_state = build_goal_state(
        session=session,
        trace=[_entry("项目介绍", "在 CuteGo 这个项目里，我主要做 WebWorker 和性能优化。", category="project_architecture")],
    )

    candidates = [
        PlannedQuestion("project", "project", "agent_architecture", "agent_architecture", "你是怎么设计记忆系统的？", [], 1),
        PlannedQuestion("project", "project", "project_architecture", "project_architecture", "WebWorker 在你的前端项目里是怎么拆分主线程负载的？", [], 1),
    ]

    chosen = selector.select_next_question(
        session=session,
        candidates=candidates,
        goal_state=goal_state,
        history=[{"category": "project_architecture"}],
    )

    assert chosen == 1


def test_goal_state_requires_switch_after_two_consecutive_turns() -> None:
    session = InterviewSession(
        user_id="u1",
        candidate_profile="\n".join(
            [
                "Candidate Snapshot",
                "- Project 1: Interview Copilot | 面向 AI Agent 岗位的模拟面试系统 | Ownership: 负责面试编排、追问和记忆策略 | Tech: Python, RAG, Agent | Deep Dive: 追问策略, 动态选题",
                "- Project 2: CuteGo | 围棋 AI 前端交互项目 | Ownership: 负责 WebWorker 与模型推理集成 | Tech: WebWorker, KataGo, Frontend | Deep Dive: 主线程拆分, 性能优化",
            ]
        ),
        focus_topics=["agent"],
    )

    goal_state = build_goal_state(
        session=session,
        trace=[
            _entry("项目背景是什么？", "在 Interview Copilot 里，我做的是 AI Agent 模拟面试。", category="project_background"),
            _entry("架构怎么设计？", "Interview Copilot 用了记忆、追问策略和动态选题。", category="agent_architecture"),
        ],
    )

    assert goal_state.active_project == "Interview Copilot"
    assert goal_state.active_project_turns == 2
    assert goal_state.project_switch_required is True
    assert "CuteGo" in goal_state.undiscussed_projects


def test_selector_fallback_avoids_active_project_when_switch_required() -> None:
    selector = LLMQuestionSelector(enabled=False)
    session = InterviewSession(
        user_id="u1",
        candidate_profile="\n".join(
            [
                "Candidate Snapshot",
                "- Project 1: Interview Copilot | 面向 AI Agent 岗位的模拟面试系统 | Ownership: 负责面试编排、追问和记忆策略 | Tech: Python, RAG, Agent | Deep Dive: 追问策略, 动态选题",
                "- Project 2: CuteGo | 围棋 AI 前端交互项目 | Ownership: 负责 WebWorker 与模型推理集成 | Tech: WebWorker, KataGo, Frontend | Deep Dive: 主线程拆分, 性能优化",
            ]
        ),
        focus_topics=["agent"],
    )
    goal_state = build_goal_state(
        session=session,
        trace=[
            _entry("项目背景是什么？", "Interview Copilot 主要做 AI Agent 模拟面试。", category="project_background"),
            _entry("架构怎么设计？", "Interview Copilot 里我做了追问策略和动态选题。", category="agent_architecture"),
        ],
    )

    candidates = [
        PlannedQuestion("project", "project", "agent_architecture", "agent_architecture", "你们的追问策略是怎么设计的？", [], 1),
        PlannedQuestion("project", "project", "project_architecture", "project_architecture", "WebWorker 在你的前端项目里是怎么拆分主线程负载的？", [], 1),
    ]

    chosen = selector.select_next_question(
        session=session,
        candidates=candidates,
        goal_state=goal_state,
        history=[{"category": "agent_architecture"}],
    )

    assert chosen == 1


def test_goal_state_marks_project_exhausted_after_multi_dimension_coverage() -> None:
    session = InterviewSession(
        user_id="u1",
        candidate_profile="\n".join(
            [
                "Candidate Snapshot",
                "- Project 1: Interview Copilot | 面向 AI Agent 岗位的模拟面试系统 | Ownership: 负责面试编排、追问和记忆策略 | Tech: Python, RAG, Agent | Deep Dive: 追问策略, 动态选题",
                "- Project 2: CuteGo | 围棋 AI 前端交互项目 | Ownership: 负责 WebWorker 与模型推理集成 | Tech: WebWorker, KataGo, Frontend | Deep Dive: 主线程拆分, 性能优化",
            ]
        ),
        focus_topics=["agent"],
    )

    goal_state = build_goal_state(
        session=session,
        trace=[
            _entry("背景是什么？", "Interview Copilot 面向 AI Agent 面试场景。", category="project_background"),
            _entry("架构怎么做？", "Interview Copilot 里我做了追问策略和动态选题。", category="agent_architecture"),
            _entry("遇到什么取舍？", "Interview Copilot 最大的 tradeoff 是追问深度和用户体验。", category="project_challenges"),
            _entry("怎么评估？", "Interview Copilot 用回放数据和命中率做评估。", category="project_evaluation"),
        ],
    )

    assert "Interview Copilot" in goal_state.exhausted_projects
    assert goal_state.project_switch_required is True


def test_goal_state_advances_project_phase_from_background_to_architecture() -> None:
    session = InterviewSession(
        user_id="u1",
        candidate_profile="\n".join(
            [
                "Candidate Snapshot",
                "- Project 1: Interview Copilot | AI interview trainer | Ownership: built interview flow | Tech: Python, Agent | Deep Dive: memory, orchestration",
            ]
        ),
        focus_topics=["agent"],
    )

    goal_state = build_goal_state(
        session=session,
        trace=[
            _entry(
                "What problem was Interview Copilot solving?",
                "In Interview Copilot I built an AI interview trainer for agent roles.",
                category="project_background",
            ),
            _entry(
                "How did you design the core architecture?",
                "Interview Copilot used an orchestrator, selector, and memory-aware follow-up loop.",
                category="agent_architecture",
            ),
        ],
    )

    assert goal_state.active_project == "Interview Copilot"
    assert goal_state.active_project_phase == "architecture"
    assert goal_state.next_project_phase == "tradeoff"
    assert goal_state.project_switch_required is False


def test_selector_fallback_prefers_next_project_phase_before_keyword_hit() -> None:
    selector = LLMQuestionSelector(enabled=False)
    goal_state = build_goal_state(
        session=InterviewSession(
            user_id="u1",
            candidate_profile="\n".join(
                [
                    "Candidate Snapshot",
                    "- Project 1: Interview Copilot | AI interview trainer | Ownership: built interview flow | Tech: Python, Agent | Deep Dive: memory, orchestration",
                ]
            ),
            focus_topics=["agent"],
        ),
        trace=[
            _entry(
                "What problem was Interview Copilot solving?",
                "In Interview Copilot I built an AI interview trainer for agent roles.",
                category="project_background",
            ),
        ],
    )

    candidates = [
        PlannedQuestion(
            "project",
            "project",
            "project_scope",
            "project_scope",
            "In Interview Copilot, how large was the project scope and team boundary?",
            [],
            1,
        ),
        PlannedQuestion(
            "project",
            "project",
            "agent_architecture",
            "agent_architecture",
            "How did you design the agent memory and orchestration flow?",
            [],
            1,
        ),
    ]

    chosen = selector.select_next_question(
        session=InterviewSession(user_id="u1", focus_topics=["agent"]),
        candidates=candidates,
        goal_state=goal_state,
        history=[{"category": "project_background"}],
    )

    assert goal_state.next_project_phase == "architecture"
    assert chosen == 1
