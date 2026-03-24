"""Structured interview goal state used by selection and phrasing."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from copilot.interview.session import InterviewSession
from copilot.profile import parse_candidate_projects

PROJECT_PHASE_SEQUENCE = [
    "background",
    "architecture",
    "tradeoff",
    "evaluation",
    "reflection",
]

PROJECT_PHASE_CATEGORIES = {
    "background": ["project_background", "project_scope", "project", "opening"],
    "architecture": [
        "project_architecture",
        "agent_architecture",
        "rag_retrieval",
        "prompt_context",
        "project_data",
        "code_agent",
    ],
    "tradeoff": ["project_challenges"],
    "evaluation": ["project_evaluation", "project_deployment"],
    "reflection": ["project_challenges", "project_scope", "project"],
}

PROJECT_PHASE_DIMENSIONS = {
    "background": {"background"},
    "architecture": {"architecture", "implementation"},
    "tradeoff": {"tradeoff"},
    "evaluation": {"evaluation"},
}


@dataclass(slots=True)
class InterviewGoalState:
    covered_categories: list[str] = field(default_factory=list)
    weak_categories: list[str] = field(default_factory=list)
    unresolved_points: list[str] = field(default_factory=list)
    recommended_focus: list[str] = field(default_factory=list)
    authenticity_status: str = "unknown"
    evidence_status: str = "unknown"
    profile_signals: list[str] = field(default_factory=list)
    project_names: list[str] = field(default_factory=list)
    discussed_projects: list[str] = field(default_factory=list)
    active_project: str = ""
    active_project_keywords: list[str] = field(default_factory=list)
    project_focus_mode: str = "floating"
    discussed_project_count: int = 0
    undiscussed_projects: list[str] = field(default_factory=list)
    project_turn_counts: dict[str, int] = field(default_factory=dict)
    active_project_turns: int = 0
    project_coverage: dict[str, list[str]] = field(default_factory=dict)
    active_project_phase: str = ""
    next_project_phase: str = ""
    exhausted_projects: list[str] = field(default_factory=list)
    project_switch_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_goal_state(
    *,
    session: InterviewSession,
    trace: list[Any],
    pending_policy_reason: str = "",
) -> InterviewGoalState:
    covered_categories: list[str] = []
    weak_categories: list[str] = []
    unresolved_points: list[str] = []
    profile_signals = _extract_profile_signals(session.candidate_profile)
    project_cards = parse_candidate_projects(session.candidate_profile)
    project_names = [str(item.get("name", "")).strip() for item in project_cards if str(item.get("name", "")).strip()]

    answers = []
    mentioned_project_scores: dict[str, int] = {}
    project_turn_counts: dict[str, int] = {}
    project_coverage: dict[str, list[str]] = {}
    trace_projects: list[str] = []
    trace_categories: list[str] = []
    for entry in trace:
        review = getattr(entry, "review", None)
        category = getattr(review, "category", "") if review is not None else ""
        trace_categories.append(category)
        if category and category not in covered_categories:
            covered_categories.append(category)
        answer = str(getattr(entry, "answer", "") or "")
        if getattr(entry, "follow_up_answer", ""):
            answer += "\n" + str(getattr(entry, "follow_up_answer", ""))
        if answer.strip():
            answers.append(answer)
            _accumulate_project_mentions(project_cards, answer, mentioned_project_scores)
        resolved_project = _resolve_entry_project(
            project_cards=project_cards,
            answer_text=answer,
            question_text=str(getattr(entry, "question", "") or ""),
        )
        trace_projects.append(resolved_project)
        if resolved_project:
            project_turn_counts[resolved_project] = project_turn_counts.get(resolved_project, 0) + 1
            dimension = _category_to_project_dimension(category)
            if dimension:
                coverage = project_coverage.setdefault(resolved_project, [])
                if dimension not in coverage:
                    coverage.append(dimension)
        if review is not None and float(getattr(review, "overall_score", 5.0)) < 3.4 and category:
            if category not in weak_categories:
                weak_categories.append(category)
        policy_reason = str(getattr(entry, "policy_reason", "") or "")
        if policy_reason and policy_reason not in {"answer_is_good_enough", "no_follow_up_available"}:
            unresolved_points.append(policy_reason)

    if pending_policy_reason and pending_policy_reason not in unresolved_points:
        unresolved_points.append(pending_policy_reason)

    combined_answers = "\n".join(answers)
    authenticity_status = _authenticity_status(combined_answers)
    evidence_status = _evidence_status(combined_answers)
    active_project = _resolve_active_project(project_cards, mentioned_project_scores)
    discussed_projects = [name for name in project_names if mentioned_project_scores.get(name, 0) > 0]
    undiscussed_projects = [name for name in project_names if name not in discussed_projects]
    exhausted_projects = [
        name
        for name, dimensions in project_coverage.items()
        if len(dimensions) >= 4 and project_turn_counts.get(name, 0) >= 3
    ]
    active_project_name = active_project.get("name", "")
    active_project_turns = _count_trailing_project_turns(trace_projects, active_project_name)
    active_project_phase = _resolve_active_project_phase(
        trace_projects=trace_projects,
        trace_categories=trace_categories,
        active_project_name=active_project_name,
    )
    next_project_phase = _resolve_next_project_phase(project_coverage.get(active_project_name, []))
    project_switch_required = bool(
        active_project_name
        and active_project_turns >= 2
        and (
            undiscussed_projects
            or active_project_name in exhausted_projects
        )
    )
    recommended_focus = _recommended_focus(
        covered_categories=covered_categories,
        weak_categories=weak_categories,
        unresolved_points=unresolved_points,
        profile_signals=profile_signals,
        authenticity_status=authenticity_status,
        evidence_status=evidence_status,
    )

    return InterviewGoalState(
        covered_categories=covered_categories,
        weak_categories=weak_categories,
        unresolved_points=unresolved_points[:5],
        recommended_focus=recommended_focus[:5],
        authenticity_status=authenticity_status,
        evidence_status=evidence_status,
        profile_signals=profile_signals[:6],
        project_names=project_names[:4],
        discussed_projects=discussed_projects[:4],
        active_project=active_project_name,
        active_project_keywords=list(active_project.get("keywords", []))[:12] if active_project else [],
        project_focus_mode="anchored" if active_project_name else "floating",
        discussed_project_count=len(discussed_projects),
        undiscussed_projects=undiscussed_projects[:4],
        project_turn_counts={key: int(value) for key, value in list(project_turn_counts.items())[:4]},
        active_project_turns=active_project_turns,
        project_coverage={key: value[:6] for key, value in list(project_coverage.items())[:4]},
        active_project_phase=active_project_phase,
        next_project_phase=next_project_phase,
        exhausted_projects=exhausted_projects[:4],
        project_switch_required=project_switch_required,
    )


def _extract_profile_signals(profile: str) -> list[str]:
    lowered = str(profile or "").lower()
    signals: list[str] = []
    mapping = [
        ("agent", "agent_architecture"),
        ("multi-agent", "agent_architecture"),
        ("rag", "rag_retrieval"),
        ("retrieval", "rag_retrieval"),
        ("memory", "agent_architecture"),
        ("workflow", "project_architecture"),
        ("webworker", "project_architecture"),
        ("python", "python_system"),
        ("deploy", "project_deployment"),
        ("evaluation", "project_evaluation"),
    ]
    for token, category in mapping:
        if token in lowered and category not in signals:
            signals.append(category)
    return signals


def _authenticity_status(text: str) -> str:
    if not text.strip():
        return "unknown"
    strong_tokens = ("我负责", "我做了", "上线", "落地", "踩坑", "tradeoff", "取舍", "为什么这么做")
    if any(token in text for token in strong_tokens):
        return "verified"
    if any(token in text for token in ("项目", "场景", "例如", "比如")):
        return "partial"
    return "unknown"


def _evidence_status(text: str) -> str:
    if not text.strip():
        return "unknown"
    if re.search(r"(\d+ms|\d+%|qps|延迟|时延|命中率|准确率|成本)", text, re.IGNORECASE):
        return "strong"
    if any(token in text for token in ("线上", "指标", "数据集", "用户反馈", "耗时", "错误率")):
        return "partial"
    return "weak"


def _recommended_focus(
    *,
    covered_categories: list[str],
    weak_categories: list[str],
    unresolved_points: list[str],
    profile_signals: list[str],
    authenticity_status: str,
    evidence_status: str,
) -> list[str]:
    focus: list[str] = []
    if "opening" not in covered_categories:
        focus.append("opening")
    if authenticity_status != "verified":
        focus.extend(["project_scope", "project_background", "project_architecture"])
    if evidence_status in {"unknown", "weak"}:
        focus.extend(["project_evaluation", "project_deployment"])
    for category in profile_signals:
        if category not in covered_categories:
            focus.append(category)
    focus.extend(weak_categories)
    if "need_execution_detail" in unresolved_points:
        focus.append("project_architecture")
    if "need_system_fallback_strategy" in unresolved_points:
        focus.append("project_deployment")
    return _unique(focus)


def _resolve_entry_project(
    *,
    project_cards: list[dict[str, Any]],
    answer_text: str,
    question_text: str,
) -> str:
    best_name = ""
    best_score = 0
    combined = f"{question_text}\n{answer_text}"
    lowered = combined.lower()
    for card in project_cards:
        name = str(card.get("name", "")).strip()
        if not name:
            continue
        score = 0
        for keyword in card.get("keywords", []):
            token = str(keyword or "").strip()
            if len(token) < 2:
                continue
            if token.lower() in lowered:
                score += 3 if token == name else 1
        if score > best_score:
            best_name = name
            best_score = score
    return best_name if best_score > 0 else ""


def _category_to_project_dimension(category: str) -> str:
    mapping = {
        "opening": "background",
        "project_background": "background",
        "project_scope": "background",
        "project": "background",
        "project_architecture": "architecture",
        "agent_architecture": "architecture",
        "rag_retrieval": "architecture",
        "prompt_context": "architecture",
        "project_data": "implementation",
        "code_agent": "implementation",
        "project_challenges": "tradeoff",
        "project_deployment": "evaluation",
        "project_evaluation": "evaluation",
    }
    return mapping.get(str(category or ""), "")


def _phase_for_category(category: str) -> str:
    normalized = str(category or "")
    for phase, categories in PROJECT_PHASE_CATEGORIES.items():
        if normalized in categories:
            return phase
    return ""


def _resolve_active_project_phase(
    *,
    trace_projects: list[str],
    trace_categories: list[str],
    active_project_name: str,
) -> str:
    if not active_project_name:
        return ""
    for project_name, category in zip(reversed(trace_projects), reversed(trace_categories)):
        if project_name != active_project_name:
            continue
        phase = _phase_for_category(category)
        if phase:
            return phase
    return ""


def _resolve_next_project_phase(dimensions: list[str]) -> str:
    covered = set(dimensions)
    for phase in PROJECT_PHASE_SEQUENCE:
        required = PROJECT_PHASE_DIMENSIONS.get(phase, set())
        if required and covered.intersection(required):
            continue
        if required:
            return phase
    return "reflection"


def preferred_categories_for_phase(phase: str) -> list[str]:
    return list(PROJECT_PHASE_CATEGORIES.get(str(phase or ""), []))


def _count_trailing_project_turns(trace_projects: list[str], active_project_name: str) -> int:
    if not active_project_name:
        return 0
    count = 0
    for name in reversed(trace_projects):
        if name != active_project_name:
            break
        count += 1
    return count


def _accumulate_project_mentions(
    project_cards: list[dict[str, Any]],
    text: str,
    scores: dict[str, int],
) -> None:
    lowered = str(text or "").lower()
    for card in project_cards:
        name = str(card.get("name", "")).strip()
        if not name:
            continue
        for keyword in card.get("keywords", []):
            token = str(keyword or "").strip()
            if len(token) < 2:
                continue
            if token.lower() in lowered:
                scores[name] = scores.get(name, 0) + (3 if token == name else 1)


def _resolve_active_project(
    project_cards: list[dict[str, Any]],
    scores: dict[str, int],
) -> dict[str, Any]:
    if not project_cards:
        return {}
    best_card = None
    best_score = -1
    for card in project_cards:
        name = str(card.get("name", "")).strip()
        score = scores.get(name, 0)
        if score > best_score:
            best_card = card
            best_score = score
    if best_card is not None and best_score > 0:
        return best_card
    return project_cards[0]


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = [
    "InterviewGoalState",
    "PROJECT_PHASE_SEQUENCE",
    "build_goal_state",
    "preferred_categories_for_phase",
]
