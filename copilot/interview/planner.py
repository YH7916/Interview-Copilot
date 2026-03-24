"""Build a compact interview candidate pool from the structured question bank."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from copilot.interview.session import InterviewSession
from copilot.knowledge.answer_cards import (
    build_answer_card_index,
    find_answer_card,
    load_answer_cards_or_empty,
)
from copilot.knowledge.question_bank import STAGE_BY_CATEGORY, classify_question, load_question_bank

STAGE_LABELS = {
    "opening": "开场介绍",
    "project": "项目深挖",
    "foundations": "基础追问",
    "coding": "算法与手撕",
}

CATEGORY_LABELS = {
    "opening": "开场与经历",
    "project_background": "项目背景与目标",
    "project_scope": "职责边界与业务影响",
    "project_data": "数据与样本工程",
    "project_architecture": "项目架构与链路",
    "project_evaluation": "评估、记忆与反馈",
    "project_deployment": "部署、性能与成本",
    "project_challenges": "难点、故障与取舍",
    "project": "项目深挖",
    "prompt_context": "提示词与上下文工程",
    "agent_architecture": "Agent 架构与技能",
    "rag_retrieval": "RAG 与检索链路",
    "code_agent": "Code Agent 与工程实现",
    "llm_fundamentals": "LLM 基础原理",
    "python_system": "Python 与系统基础",
    "coding": "算法与手撕",
}

ROUND_STAGE_TARGETS = {
    "standard": {
        1: ["opening", "project", "project", "project", "project", "foundations", "coding", "project"],
        2: ["project", "project", "project", "project", "foundations", "coding", "project"],
        3: ["project", "project", "project", "foundations", "coding", "project"],
    },
    "coding_first": {
        1: ["coding", "opening", "project", "project", "project", "foundations", "project"],
        2: ["coding", "project", "project", "project", "foundations", "project"],
        3: ["coding", "project", "project", "foundations", "project"],
    },
    "no_coding": {
        1: ["opening", "project", "project", "project", "project", "foundations", "project"],
        2: ["project", "project", "project", "project", "foundations", "project"],
        3: ["project", "project", "project", "foundations", "project"],
    },
}

PROJECT_CATEGORY_GROUPS = {
    "background": ["project_background", "project_scope", "project"],
    "architecture": [
        "project_architecture",
        "agent_architecture",
        "rag_retrieval",
        "prompt_context",
        "project_data",
        "code_agent",
        "project",
    ],
    "outcomes": ["project_evaluation", "project_deployment", "project_challenges", "project"],
}

PROJECT_FOCUSABLE_CATEGORIES = [
    "agent_architecture",
    "rag_retrieval",
    "prompt_context",
    "code_agent",
    "project_architecture",
    "project_data",
    "project_evaluation",
    "project_deployment",
    "project_challenges",
    "project_scope",
    "project_background",
    "project",
]

FOUNDATION_CATEGORIES = ["llm_fundamentals", "python_system"]
CODING_CATEGORIES = ["coding", "llm_fundamentals", "python_system"]

FOCUS_RULES = {
    "prompt": ["prompt_context"],
    "context": ["prompt_context"],
    "ownership": ["project_scope"],
    "responsibility": ["project_scope"],
    "scope": ["project_scope"],
    "impact": ["project_scope", "project_background"],
    "background": ["project_background"],
    "scenario": ["project_background"],
    "business": ["project_background"],
    "数据": ["project_data"],
    "dataset": ["project_data"],
    "label": ["project_data"],
    "clean": ["project_data"],
    "sampling": ["project_data"],
    "architecture": ["project_architecture"],
    "pipeline": ["project_architecture"],
    "module": ["project_architecture"],
    "router": ["project_architecture"],
    "orchestr": ["project_architecture"],
    "workflow": ["agent_architecture", "project_architecture"],
    "agent": ["agent_architecture"],
    "multi-agent": ["agent_architecture"],
    "multi agent": ["agent_architecture"],
    "react": ["agent_architecture"],
    "memory": ["agent_architecture", "project_evaluation"],
    "tool": ["agent_architecture"],
    "skill": ["agent_architecture"],
    "rag": ["rag_retrieval"],
    "retrieval": ["rag_retrieval"],
    "bm25": ["rag_retrieval"],
    "rerank": ["rag_retrieval"],
    "embedding": ["rag_retrieval"],
    "query": ["rag_retrieval"],
    "eval": ["project_evaluation"],
    "evaluation": ["project_evaluation"],
    "feedback": ["project_evaluation"],
    "metric": ["project_background", "project_evaluation"],
    "kpi": ["project_background", "project_evaluation"],
    "deploy": ["project_deployment"],
    "deployment": ["project_deployment"],
    "latency": ["project_deployment"],
    "throughput": ["project_deployment"],
    "qps": ["project_deployment"],
    "cost": ["project_deployment"],
    "tradeoff": ["project_challenges"],
    "failure": ["project_challenges"],
    "issue": ["project_challenges"],
    "debug": ["project_challenges"],
    "llm": ["llm_fundamentals"],
    "attention": ["llm_fundamentals"],
    "lora": ["llm_fundamentals"],
    "qlora": ["llm_fundamentals"],
    "python": ["python_system"],
    "gil": ["python_system"],
    "cpp": ["python_system"],
    "coding": ["coding"],
    "algorithm": ["coding"],
}


@dataclass(slots=True)
class PlannedQuestion:
    stage: str
    stage_label: str
    category: str
    category_label: str
    question: str
    follow_ups: list[str]
    source_count: int
    latest_source_at: str = ""
    answer_status: str = ""
    reference_answer: str = ""
    pitfalls: list[str] = field(default_factory=list)
    evidence: list[dict[str, str]] = field(default_factory=list)


class InterviewPlanner:
    def __init__(self, bank: dict[str, Any] | None = None, *, recent: bool = True):
        self.bank = bank or load_question_bank(recent=recent)
        self.fallback_bank = None if bank is not None or not recent else load_question_bank(recent=False)
        self.answer_cards = load_answer_cards_or_empty(recent=recent)
        self.fallback_answer_cards = (
            {"generated_at": "", "categories": []}
            if bank is not None or not recent
            else load_answer_cards_or_empty(recent=False)
        )
        self._by_category = _normalize_bank_categories(self.bank)
        self._fallback_by_category = _normalize_bank_categories(self.fallback_bank or {})
        self._answer_card_index = build_answer_card_index(self.answer_cards)
        self._fallback_answer_card_index = build_answer_card_index(self.fallback_answer_cards)

    def plan(self, session: InterviewSession, max_questions: int = 8) -> list[PlannedQuestion]:
        used: set[str] = set()
        planned: list[PlannedQuestion] = []
        recent_categories: list[str] = []
        category_counts: dict[str, int] = {}
        focus_categories = _resolve_focus_categories(
            [
                *session.focus_topics,
                session.company,
                session.position,
                session.candidate_profile,
            ]
        )
        interview_style = self._resolve_interview_style(session.interview_style)
        project_slot = 0

        for stage in _stage_targets_for_round(session.round_index, interview_style):
            if len(planned) >= max_questions:
                break
            categories = _categories_for_stage(stage, focus_categories, project_slot)
            if stage == "project":
                project_slot += 1
            picks = self._pick_questions(
                categories=categories,
                stage=stage,
                limit=1,
                used=used,
                company=session.company,
                focus_categories=focus_categories,
                recent_categories=recent_categories,
                category_counts=category_counts,
            )
            if not picks:
                continue
            pick = picks[0]
            planned.append(pick)
            recent_categories.append(pick.category)
            category_counts[pick.category] = category_counts.get(pick.category, 0) + 1

        return planned[:max_questions]

    def _resolve_interview_style(self, requested_style: str) -> str:
        normalized = str(requested_style or "auto").strip().lower().replace("_", "-")
        if normalized in {"standard", "coding-first", "no-coding"}:
            return normalized.replace("-", "_")
        if self._has_usable_category("coding"):
            return "standard"
        return "no_coding"

    def _has_usable_category(self, category: str) -> bool:
        for question in self._by_category.get(category, []):
            if _is_usable_question(question["question"], category=category):
                return True
        for question in self._fallback_by_category.get(category, []):
            if _is_usable_question(question["question"], category=category):
                return True
        return False

    def _pick_questions(
        self,
        *,
        categories: list[str],
        stage: str,
        limit: int,
        used: set[str],
        company: str,
        focus_categories: set[str],
        recent_categories: list[str] | None = None,
        category_counts: dict[str, int] | None = None,
    ) -> list[PlannedQuestion]:
        recent_categories = recent_categories or []
        category_counts = category_counts or {}
        grouped_candidates: list[tuple[str, list[tuple[int, int, dict[str, Any]]]]] = []

        for category_order, category in enumerate(categories):
            local_candidates: list[tuple[int, int, dict[str, Any]]] = []
            for question in self._by_category.get(category, []):
                if question["question"] in used or not _is_usable_question(question["question"], category=category):
                    continue
                score = self._score(
                    question,
                    company,
                    focus_categories,
                    category=category,
                    category_order=category_order,
                    recent_categories=recent_categories,
                    category_counts=category_counts,
                )
                local_candidates.append((score, 0, question))

            if not local_candidates:
                for question in self._fallback_by_category.get(category, []):
                    if question["question"] in used or not _is_usable_question(question["question"], category=category):
                        continue
                    score = self._score(
                        question,
                        company,
                        focus_categories,
                        category=category,
                        category_order=category_order,
                        recent_categories=recent_categories,
                        category_counts=category_counts,
                    ) - 5
                    local_candidates.append((score, 1, question))

            if not local_candidates:
                continue
            local_candidates.sort(key=lambda item: (-item[0], item[1], item[2]["question"]))
            grouped_candidates.append((category, local_candidates))

        if not grouped_candidates:
            return []

        ordered_groups = grouped_candidates[:]
        if len(ordered_groups) > 1:
            for index, (category, _) in enumerate(grouped_candidates):
                if recent_categories and category == recent_categories[-1]:
                    continue
                if index == 0 and category_counts.get(category, 0) > 0:
                    continue
                if index > 0:
                    ordered_groups = [
                        grouped_candidates[index],
                        *grouped_candidates[:index],
                        *grouped_candidates[index + 1 :],
                    ]
                break

        result = []
        for category, questions in ordered_groups:
            for _, _, question in questions:
                used.add(question["question"])
                card = self._lookup_answer_card(category, question["question"])
                resolved_stage = str(question.get("stage", "")) or stage
                result.append(
                    PlannedQuestion(
                        stage=resolved_stage,
                        stage_label=STAGE_LABELS.get(resolved_stage, STAGE_LABELS[stage]),
                        category=category,
                        category_label=CATEGORY_LABELS[category],
                        question=question["question"],
                        follow_ups=(
                            question.get("follow_ups")
                            or [alias for alias in question.get("aliases", []) if alias != question["question"]]
                        )[:3],
                        source_count=int(question.get("source_count", 0)),
                        latest_source_at=str(question.get("latest_source_at", "")),
                        answer_status=str(card.get("status", "")),
                        reference_answer=str(card.get("answer", "")),
                        pitfalls=list(card.get("pitfalls", [])),
                        evidence=list(card.get("evidence", [])),
                    )
                )
                if len(result) >= limit:
                    return result
        return result

    @staticmethod
    def _score(
        question: dict[str, Any],
        company: str,
        focus_categories: set[str],
        *,
        category: str,
        category_order: int,
        recent_categories: list[str],
        category_counts: dict[str, int],
    ) -> int:
        score = int(question.get("source_count", 0)) * 10
        if category in focus_categories:
            score += 25
        if company and any(company.lower() in str(source.get("title", "")).lower() for source in question.get("sources", [])):
            score += 30
        score += _source_signal(question.get("sources", []))
        if question.get("follow_ups"):
            score += 5
        text = str(question.get("question", ""))
        if any(token in text for token in ("怎么", "如何", "为什么", "how", "why")):
            score += 3
        score += _recency_bonus(str(question.get("latest_source_at", "")))
        score += max(0, 32 - category_order * 8)

        if recent_categories:
            if category == recent_categories[-1]:
                score -= 18
            elif category in recent_categories[-2:]:
                score -= 8

        if category_counts.get(category, 0) > 0:
            score -= category_counts[category] * 6

        return score

    def _lookup_answer_card(self, category: str, question: str) -> dict[str, Any]:
        card = find_answer_card(question, category=category, index=self._answer_card_index)
        if card is not None:
            return card
        return find_answer_card(question, category=category, index=self._fallback_answer_card_index) or {}


def _normalize_bank_categories(bank: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    normalized: dict[str, list[dict[str, Any]]] = {}
    for category in bank.get("categories", []):
        items = category.get("clusters") or category.get("questions", [])
        for item in items:
            resolved_category = classify_question(str(item.get("question", "")))
            resolved_item = dict(item)
            resolved_item["category"] = resolved_category
            resolved_item["category_label"] = CATEGORY_LABELS[resolved_category]
            resolved_item["stage"] = STAGE_BY_CATEGORY[resolved_category]
            normalized.setdefault(resolved_category, []).append(resolved_item)
    for items in normalized.values():
        items.sort(
            key=lambda item: (
                str(item.get("latest_source_at", "")),
                int(item.get("source_count", 0)),
                str(item.get("question", "")),
            ),
            reverse=True,
        )
    return normalized


def render_plan(plan: list[PlannedQuestion]) -> str:
    lines = [
        "# Mock Interview",
        "",
        f"- questions: {len(plan)}",
        "- source: recent real interview posts",
        "- next_step: answer naturally, then run /review",
        "",
    ]
    for index, item in enumerate(plan, 1):
        lines.append(f"{index}. [{item.stage_label} | {item.category_label}] {item.question}")
        lines.append(f"   - signal: {item.source_count} source(s)")
        if item.latest_source_at:
            lines.append(f"   - latest: {item.latest_source_at[:10]}")
        if item.answer_status:
            lines.append(f"   - answer-card: {item.answer_status}")
        if item.follow_ups:
            lines.append(f"   - likely_follow_up: {' / '.join(item.follow_ups)}")
    return "\n".join(lines).rstrip() + "\n"


def _resolve_focus_categories(topics: list[str]) -> set[str]:
    categories: set[str] = set()
    for topic in topics:
        lowered = str(topic or "").lower()
        for key, values in FOCUS_RULES.items():
            if key in lowered:
                categories.update(values)
    return categories


def _stage_targets_for_round(round_index: int, interview_style: str) -> list[str]:
    template_groups = ROUND_STAGE_TARGETS.get(interview_style, ROUND_STAGE_TARGETS["standard"])
    return list(template_groups.get(round_index, template_groups[1]))


def _categories_for_stage(stage: str, focus_categories: set[str], project_slot: int) -> list[str]:
    if stage == "opening":
        return ["opening"]
    if stage == "coding":
        foundation_focus = [item for item in FOUNDATION_CATEGORIES if item in focus_categories]
        return _unique(["coding", *foundation_focus, *CODING_CATEGORIES])
    if stage == "foundations":
        foundation_focus = [item for item in FOUNDATION_CATEGORIES if item in focus_categories]
        return _unique([*foundation_focus, *FOUNDATION_CATEGORIES])
    return _project_slot_categories(project_slot, focus_categories)


def _project_slot_categories(project_slot: int, focus_categories: set[str]) -> list[str]:
    project_focus = [item for item in PROJECT_FOCUSABLE_CATEGORIES if item in focus_categories]
    if project_slot <= 0:
        return _unique(
            [
                *PROJECT_CATEGORY_GROUPS["background"],
                *project_focus,
                *PROJECT_CATEGORY_GROUPS["architecture"],
                *PROJECT_CATEGORY_GROUPS["outcomes"],
            ]
        )
    if project_slot == 1:
        return _unique(
            [
                *PROJECT_CATEGORY_GROUPS["architecture"],
                *project_focus,
                *PROJECT_CATEGORY_GROUPS["background"],
                *PROJECT_CATEGORY_GROUPS["outcomes"],
            ]
        )
    if project_slot == 2:
        return _unique(
            [
                *project_focus,
                *PROJECT_CATEGORY_GROUPS["architecture"],
                *PROJECT_CATEGORY_GROUPS["outcomes"],
                *PROJECT_CATEGORY_GROUPS["background"],
            ]
        )
    return _unique(
        [
            *PROJECT_CATEGORY_GROUPS["outcomes"],
            *project_focus,
            *PROJECT_CATEGORY_GROUPS["architecture"],
            *PROJECT_CATEGORY_GROUPS["background"],
        ]
    )


def _unique(values: list[str]) -> list[str]:
    result = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _recency_bonus(value: str) -> int:
    if not value:
        return 0
    try:
        days = (datetime.now() - datetime.fromisoformat(value)).days
    except ValueError:
        return 0
    if days <= 30:
        return 30
    if days <= 90:
        return 20
    if days <= 180:
        return 10
    return 0


def _source_signal(sources: list[dict[str, Any]]) -> int:
    score = 0
    for source in sources:
        title = str(source.get("title", "")).lower()
        if any(keyword in title for keyword in ("一面", "二面", "三面", "面经", "offer", "oc")):
            score += 4
    return min(score, 12)


def _is_usable_question(text: str, *, category: str = "") -> bool:
    value = str(text or "").strip()
    if category == "coding":
        return len(value) >= 2
    if len(value) < 6:
        return value in {"自我介绍", "项目介绍", "介绍一下你自己"}
    if value.startswith(("而是", "这是", "这种", "一般", "通常", "常见", "本质上", "实际项目里", "很多")):
        return False
    if value.endswith(("？", "?")):
        return True
    if "问题" in value:
        return True
    if any(value.startswith(prefix) for prefix in ("什么", "如何", "怎么", "为什么", "请", "介绍", "如果")):
        return True
    if any(token in value for token in ("什么", "如何", "怎么", "为什么", "区别", "是否", "有没有")):
        return True
    if value.endswith(("吗", "呢", "呀")):
        return True
    return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a realistic interview plan from the question bank.")
    parser.add_argument("--user-id", default="demo")
    parser.add_argument("--round", dest="round_index", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--company", default="")
    parser.add_argument("--position", default="")
    parser.add_argument("--focus-topic", action="append", dest="focus_topics", default=[])
    parser.add_argument("--max-questions", type=int, default=8)
    parser.add_argument("--all-history", action="store_true", help="Use the full bank instead of the recent bank.")
    parser.add_argument("--json", action="store_true", help="Print plan as JSON instead of markdown.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    session = InterviewSession(
        user_id=args.user_id,
        round_index=args.round_index,
        company=args.company,
        position=args.position,
        focus_topics=args.focus_topics,
    )
    plan = InterviewPlanner(recent=not args.all_history).plan(session, max_questions=args.max_questions)
    if args.json:
        print(json.dumps([asdict(item) for item in plan], ensure_ascii=False, indent=2))
    else:
        print(render_plan(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
