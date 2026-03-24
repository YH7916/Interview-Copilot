"""LLM-backed interviewer phrasing helpers with safe fallback."""

from __future__ import annotations

import json
from typing import Any

from copilot.config import get_text_settings
from copilot.interview.planner import PlannedQuestion
from copilot.interview.session import InterviewSession
from copilot.interview.state import InterviewGoalState
from copilot.llm import call_text, parse_json_response

INTERVIEWER_SYSTEM_PROMPT = (
    "你是一位资深 AI Agent 岗位面试官。"
    "你的任务不是照本宣科复述题库，而是基于题目意图、候选人简历和上下文，自然地发问。"
    "语气专业、简洁、有压迫感但不过度苛刻，要像真实一线面试官。"
    "不要暴露评分规则、题库、信号值、内部 reason，也不要像教练一样提示答题框架。"
    "如果候选人已经给出某些信息，就不要重复追问已经回答过的点。"
    "输出必须是 JSON。"
)


class LLMInterviewer:
    def __init__(self, *, enabled: bool | None = None):
        self.enabled = self._resolve_enabled() if enabled is None else enabled

    def render_question(
        self,
        *,
        item: PlannedQuestion,
        session: InterviewSession,
        index: int,
        total_questions: int,
        goal_state: InterviewGoalState | None = None,
        history: list[dict[str, str]] | None = None,
        fallback: str,
    ) -> str:
        prompt = {
            "task": "render_question",
            "index": index,
            "total_questions": total_questions,
            "topic": ", ".join(session.focus_topics),
            "interview_style": session.interview_style,
            "candidate_profile": _shorten(session.candidate_profile),
            "goal_state": goal_state.to_dict() if goal_state is not None else {},
            "active_project": getattr(goal_state, "active_project", ""),
            "active_project_phase": getattr(goal_state, "active_project_phase", ""),
            "next_project_phase": getattr(goal_state, "next_project_phase", ""),
            "history": history or [],
            "question_hint": item.question,
            "category": item.category,
            "stage": item.stage,
            "follow_up_hints": item.follow_ups[:2],
            "guidance": [
                "用一句自然的中文提问",
                "优先贴近候选人的项目背景",
                "如果 goal_state 里已经有 active_project，优先围绕这个项目继续深挖",
                "可以重写题面，但不要改变考察意图",
                "不要添加答案提示",
            ],
        }
        rendered = self._complete_json(prompt).get("prompt", "")
        return _clean_prompt_text(rendered) or fallback

    def render_follow_up(
        self,
        *,
        item: PlannedQuestion,
        session: InterviewSession,
        candidate_answer: str,
        policy_reason: str,
        goal_state: InterviewGoalState | None = None,
        fallback: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        prompt = {
            "task": "render_follow_up",
            "topic": ", ".join(session.focus_topics),
            "candidate_profile": _shorten(session.candidate_profile),
            "goal_state": goal_state.to_dict() if goal_state is not None else {},
            "active_project": getattr(goal_state, "active_project", ""),
            "active_project_phase": getattr(goal_state, "active_project_phase", ""),
            "next_project_phase": getattr(goal_state, "next_project_phase", ""),
            "history": history or [],
            "question_hint": item.question,
            "category": item.category,
            "stage": item.stage,
            "candidate_answer": candidate_answer,
            "follow_up_hints": item.follow_ups[:3],
            "follow_up_reason": policy_reason,
            "guidance": [
                "追问必须紧扣候选人刚才的回答",
                "如果已经锚定在某个 active_project，就继续围绕这个项目追问，不要轻易跳项目",
                "优先问缺失的细节、取舍、边界和真实项目信息",
                "只输出一句追问",
                "不要说“我想继续追问一个点”之类的台词",
            ],
        }
        rendered = self._complete_json(prompt).get("prompt", "")
        return _clean_prompt_text(rendered) or fallback

    def render_clarification(
        self,
        *,
        item: PlannedQuestion,
        session: InterviewSession,
        candidate_message: str,
        goal_state: InterviewGoalState | None = None,
        fallback: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        prompt = {
            "task": "render_clarification",
            "topic": ", ".join(session.focus_topics),
            "candidate_profile": _shorten(session.candidate_profile),
            "goal_state": goal_state.to_dict() if goal_state is not None else {},
            "active_project": getattr(goal_state, "active_project", ""),
            "active_project_phase": getattr(goal_state, "active_project_phase", ""),
            "next_project_phase": getattr(goal_state, "next_project_phase", ""),
            "history": history or [],
            "question_hint": item.question,
            "category": item.category,
            "stage": item.stage,
            "candidate_message": candidate_message,
            "guidance": [
                "候选人在请求澄清，你要像真人面试官一样补充上下文",
                "如果 goal_state 有 active_project，可以直接告诉候选人优先基于这个项目回答",
                "告诉他可以基于哪个项目或哪个角度来答",
                "不要重复完整原题，不要输出多段",
            ],
        }
        rendered = self._complete_json(prompt).get("prompt", "")
        return _clean_prompt_text(rendered) or fallback

    def _complete_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {}
        try:
            response = call_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                task="chat",
                system_prompt=INTERVIEWER_SYSTEM_PROMPT + ' 输出格式: {"prompt":"..."}',
                temperature=0.4,
                max_tokens=220,
            )
            data = parse_json_response(response)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _resolve_enabled() -> bool:
        try:
            return bool(get_text_settings(task="chat").get("api_key"))
        except Exception:
            return False


def _clean_prompt_text(text: str) -> str:
    value = str(text or "").strip().strip('"').strip()
    if not value:
        return ""
    value = value.replace("\r", " ").replace("\n", " ").strip()
    for prefix in ("追问：", "问题：", "面试官："):
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()
    return value


def _shorten(text: str, limit: int = 900) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


__all__ = ["LLMInterviewer", "INTERVIEWER_SYSTEM_PROMPT"]
