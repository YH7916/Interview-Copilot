"""LLM-backed question selection with deterministic fallback."""

from __future__ import annotations

import json
from typing import Any

from copilot.config import get_text_settings
from copilot.interview.planner import PlannedQuestion
from copilot.interview.session import InterviewSession
from copilot.interview.state import InterviewGoalState, preferred_categories_for_phase
from copilot.llm import call_text, parse_json_response

SELECTOR_SYSTEM_PROMPT = (
    "你是一位资深 AI Agent 岗位面试官，负责决定下一题问什么。"
    "题库只是一组候选题，你要根据候选人的简历、刚才的表现、已经问过的内容来决定下一题。"
    "优先保证面试流畅、有层次，避免重复同一个角度。"
    "如果已经围绕某个项目连续深挖两轮，并且另一个核心项目还没充分覆盖，就应该主动切项目。"
    "输出必须是 JSON。"
)


class LLMQuestionSelector:
    def __init__(self, *, enabled: bool | None = None):
        self.enabled = self._resolve_enabled() if enabled is None else enabled

    def select_next_question(
        self,
        *,
        session: InterviewSession,
        candidates: list[PlannedQuestion],
        goal_state: InterviewGoalState | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> int:
        if len(candidates) <= 1:
            return 0

        fallback_index = self._fallback_index(
            candidates=candidates,
            goal_state=goal_state,
            history=history or [],
        )
        if not self.enabled:
            return fallback_index

        payload = {
            "topic": ", ".join(session.focus_topics),
            "interview_style": session.interview_style,
            "candidate_profile": _shorten(session.candidate_profile),
            "goal_state": goal_state.to_dict() if goal_state is not None else {},
            "history": history or [],
            "candidates": [
                {
                    "index": index,
                    "question": item.question,
                    "category": item.category,
                    "stage": item.stage,
                    "source_count": item.source_count,
                    "latest_source_at": item.latest_source_at,
                    "follow_up_hints": item.follow_ups[:2],
                }
                for index, item in enumerate(candidates)
            ],
            "guidance": [
                "选择最适合现在继续往下问的一题",
                "尽量避免刚问过的同类问题，除非明确需要继续深挖",
                "优先兼顾候选人的项目背景和面试节奏",
                "如果 goal_state.project_switch_required 为 true，优先切到另一个核心项目或新的面试维度",
            ],
        }
        try:
            response = call_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                task="analysis",
                system_prompt=SELECTOR_SYSTEM_PROMPT + ' 输出格式: {"selected_index":0,"reason":"..."}',
                temperature=0.2,
                max_tokens=180,
            )
            parsed = parse_json_response(response)
        except Exception:
            return fallback_index

        try:
            selected_index = int(parsed.get("selected_index", fallback_index))
        except Exception:
            return fallback_index
        if 0 <= selected_index < len(candidates):
            return selected_index
        return fallback_index

    @staticmethod
    def _fallback_index(
        *,
        candidates: list[PlannedQuestion],
        goal_state: InterviewGoalState | None,
        history: list[dict[str, Any]],
    ) -> int:
        if getattr(goal_state, "project_switch_required", False):
            non_active_index = _first_non_active_candidate(candidates, goal_state)
            if non_active_index is not None:
                return non_active_index

        if not getattr(goal_state, "project_switch_required", False):
            best_phase_index = _best_phase_match_index(candidates, goal_state, history)
            if best_phase_index is not None:
                return best_phase_index
            best_project_index = _best_project_match_index(candidates, goal_state)
            if best_project_index is not None:
                return best_project_index

        recommended = list(getattr(goal_state, "recommended_focus", []) or [])
        if recommended:
            for preferred in recommended:
                for index, item in enumerate(candidates):
                    if item.category == preferred:
                        return index

        if not history:
            return 0
        last_category = str(history[-1].get("category", ""))
        for index, item in enumerate(candidates):
            if item.category != last_category:
                return index
        return 0

    @staticmethod
    def _resolve_enabled() -> bool:
        try:
            return bool(get_text_settings(task="analysis").get("api_key"))
        except Exception:
            return False


def _shorten(text: str, limit: int = 900) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _best_project_match_index(
    candidates: list[PlannedQuestion],
    goal_state: InterviewGoalState | None,
) -> int | None:
    keywords = list(getattr(goal_state, "active_project_keywords", []) or [])
    if not keywords:
        return None

    best_index: int | None = None
    best_score = 0
    for index, item in enumerate(candidates):
        score = _question_project_affinity(item.question, keywords)
        if score > best_score:
            best_index = index
            best_score = score
    return best_index if best_score > 0 else None


def _best_phase_match_index(
    candidates: list[PlannedQuestion],
    goal_state: InterviewGoalState | None,
    history: list[dict[str, Any]],
) -> int | None:
    phase_categories = preferred_categories_for_phase(getattr(goal_state, "next_project_phase", ""))
    if not phase_categories:
        return None

    last_category = str(history[-1].get("category", "")) if history else ""
    active_keywords = list(getattr(goal_state, "active_project_keywords", []) or [])
    best_index: int | None = None
    best_score: int | None = None

    for index, item in enumerate(candidates):
        if item.category not in phase_categories:
            continue
        score = 100 - phase_categories.index(item.category) * 10
        if item.category == last_category:
            score -= 12
        affinity = _question_project_affinity(item.question, active_keywords)
        if affinity > 0:
            score += 20 + affinity
        if best_score is None or score > best_score:
            best_index = index
            best_score = score

    return best_index


def _question_project_affinity(question: str, keywords: list[str]) -> int:
    lowered = str(question or "").lower()
    score = 0
    for keyword in keywords:
        token = str(keyword or "").strip().lower()
        if len(token) < 2:
            continue
        if token in lowered:
            score += 3 if " " in token or len(token) >= 8 else 1
    return score


def _first_non_active_candidate(
    candidates: list[PlannedQuestion],
    goal_state: InterviewGoalState | None,
) -> int | None:
    active_keywords = list(getattr(goal_state, "active_project_keywords", []) or [])
    if not active_keywords:
        return 0 if candidates else None
    for index, item in enumerate(candidates):
        if _question_project_affinity(item.question, active_keywords) == 0:
            return index
    return None


__all__ = ["LLMQuestionSelector", "SELECTOR_SYSTEM_PROMPT"]
