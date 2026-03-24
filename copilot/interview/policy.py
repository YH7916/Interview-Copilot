"""Thin runtime policy for the live interview harness."""

from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass(slots=True)
class PolicyDecision:
    action: str
    reason: str
    missing_points: list[str] = field(default_factory=list)


def decide_next_action(
    *,
    has_follow_ups: bool,
    answer_text: str,
    depth_score: int,
    evidence_score: int,
    overall_score: float,
    question: str = "",
    category: str = "",
) -> PolicyDecision:
    normalized_answer = answer_text.strip()
    if is_clarification_request(normalized_answer):
        return PolicyDecision(action="clarify", reason="candidate_requested_clarification")

    missing_points = _infer_missing_points(
        question=question,
        category=category,
        answer_text=normalized_answer,
    )
    if has_follow_ups and missing_points:
        return PolicyDecision(
            action="follow_up",
            reason=missing_points[0],
            missing_points=missing_points,
        )

    if not has_follow_ups:
        return PolicyDecision(action="advance", reason="no_follow_up_available")

    answer_length = len(normalized_answer)
    if answer_length < 28 and not _looks_compact_but_complete(normalized_answer):
        return PolicyDecision(action="follow_up", reason="need_deeper_reasoning")
    if depth_score <= 2:
        return PolicyDecision(action="follow_up", reason="need_deeper_reasoning")
    if evidence_score <= 1 and _expects_real_world_details(question=question, category=category):
        return PolicyDecision(action="follow_up", reason="need_concrete_evidence")
    if overall_score < 2.8:
        return PolicyDecision(action="follow_up", reason="need_deeper_reasoning")
    return PolicyDecision(action="advance", reason="answer_is_good_enough")


def should_follow_up(
    *,
    has_follow_ups: bool,
    answer_text: str,
    depth_score: int,
    evidence_score: int,
    overall_score: float,
    question: str = "",
    category: str = "",
) -> bool:
    return decide_next_action(
        has_follow_ups=has_follow_ups,
        answer_text=answer_text,
        depth_score=depth_score,
        evidence_score=evidence_score,
        overall_score=overall_score,
        question=question,
        category=category,
    ).action == "follow_up"


def is_clarification_request(answer_text: str) -> bool:
    normalized = str(answer_text or "").strip().lower()
    if not normalized:
        return False

    patterns = (
        "您指的是",
        "你指的是",
        "哪个项目",
        "哪个场景",
        "哪一个项目",
        "是指",
        "具体是指",
        "能具体说一下题意吗",
        "可以明确一下",
        "麻烦确认一下",
        "which project",
        "which scenario",
    )
    if any(token in normalized for token in patterns):
        return True

    return normalized.endswith(("?", "？")) and any(
        token in normalized for token in ("哪个", "哪一个", "是指", "题意", "具体", "which")
    )


def _infer_missing_points(*, question: str, category: str, answer_text: str) -> list[str]:
    normalized_question = str(question or "").lower()
    normalized_answer = str(answer_text or "").lower()
    normalized_category = str(category or "").lower()

    if _is_opening_question(normalized_question, normalized_category):
        return _opening_missing_points(answer_text)

    if "webworker" in normalized_question:
        return _webworker_missing_points(normalized_answer)

    if "调度" in question and "策略" in question:
        return _dispatch_missing_points(normalized_answer)

    if "记忆" in question or "memory" in normalized_question:
        return _memory_missing_points(normalized_answer)

    return []


def _opening_missing_points(answer_text: str) -> list[str]:
    missing: list[str] = []
    if not _has_school_context(answer_text):
        missing.append("missing_school_context")
    if not _has_stage_context(answer_text):
        missing.append("missing_current_stage")
    if not _has_graduation_context(answer_text):
        missing.append("missing_graduation_timeline")
    return missing[:2]


def _webworker_missing_points(answer_text: str) -> list[str]:
    if not any(token in answer_text for token in ("比如", "例如", "项目", "场景")):
        return ["need_project_example"]
    if not any(
        token in answer_text
        for token in ("主线程", "消息", "通信", "postmessage", "推理", "计算", "wasm", "共享", "transferable")
    ):
        return ["need_execution_detail"]
    return []


def _dispatch_missing_points(answer_text: str) -> list[str]:
    if any(token in answer_text for token in ("报错", "提示用户")) and not any(
        token in answer_text for token in ("重试", "降级", "回退", "fallback", "兜底", "超时", "熔断", "限流")
    ):
        return ["need_system_fallback_strategy"]
    if not any(
        token in answer_text
        for token in ("优先级", "队列", "并发", "路由", "调度", "分发", "超时", "重试", "预算", "仲裁")
    ):
        return ["need_execution_detail"]
    return []


def _memory_missing_points(answer_text: str) -> list[str]:
    if any(token in answer_text for token in ("短期", "长期", "滑动窗口", "持久化")) and not any(
        token in answer_text for token in ("写入", "召回", "检索", "触发", "更新", "压缩", "遗忘", "淘汰")
    ):
        return ["need_execution_detail"]
    return []


def _is_opening_question(question: str, category: str) -> bool:
    if category == "opening":
        return True
    return any(token in question for token in ("自我介绍", "介绍一下你自己", "介绍一下自己"))


def _has_school_context(answer_text: str) -> bool:
    return bool(re.search(r"(大学|学院|school|college|zju|浙大)", answer_text, re.IGNORECASE))


def _has_stage_context(answer_text: str) -> bool:
    return bool(
        re.search(
            r"(大一|大二|大三|大四|本科|硕士|研究生|博士|应届|毕业年级|研一|研二|研三)",
            answer_text,
            re.IGNORECASE,
        )
    )


def _has_graduation_context(answer_text: str) -> bool:
    return bool(re.search(r"(20\d{2}\s*年?.{0,2}毕业|预计.{0,4}毕业|毕业时间)", answer_text, re.IGNORECASE))


def _looks_compact_but_complete(answer_text: str) -> bool:
    return sum(
        1
        for token in ("比如", "例如", "因为", "所以", "负责", "项目", "主线程", "长期", "短期", "fallback")
        if token in answer_text
    ) >= 2


def _expects_real_world_details(*, question: str, category: str) -> bool:
    normalized_question = str(question or "").lower()
    normalized_category = str(category or "").lower()
    if normalized_category in {
        "project",
        "project_architecture",
        "project_data",
        "project_evaluation",
        "project_deployment",
        "project_challenges",
        "agent_architecture",
        "rag_retrieval",
    }:
        return True
    return any(token in normalized_question for token in ("项目", "设计", "架构", "调度", "记忆", "webworker", "rag"))


__all__ = ["PolicyDecision", "decide_next_action", "is_clarification_request", "should_follow_up"]
