"""Structured evaluation helpers for answer-card-based interview review."""

from __future__ import annotations

import json
from typing import Any, Callable

from copilot.llm import call_text, parse_json_response

SCORE_DIMENSIONS = [
    "accuracy",
    "clarity",
    "depth",
    "evidence",
    "structure",
]

JudgeFn = Callable[[str], str]

JUDGE_PROMPT = """你是一名严格的 AI Agent 面试官。请根据参考答案和常见失分点，评估候选人的回答。
评分维度：
1. accuracy: 是否答对核心事实与关键链路。
2. clarity: 表达是否清晰，是否有明确主线。
3. depth: 是否讲到设计取舍、边界条件和继续追问能展开的层次。
4. evidence: 是否结合项目、指标、失败案例或真实实现细节。
5. structure: 是否先给结论再展开，回答层次是否稳定。

输入：
- 问题：{question}
- 参考答案：{expected_answer}
- 常见失分点：{pitfalls}
- 候选人回答：{actual_answer}

请严格输出 JSON：
{{"accuracy_score": 1-5, "clarity_score": 1-5, "depth_score": 1-5, "evidence_score": 1-5, "structure_score": 1-5, "reason": "一句到两句点评"}}"""


def evaluate_answer(
    question: str,
    expected_answer: str,
    actual_answer: str,
    *,
    pitfalls: list[str] | None = None,
    judge_fn: JudgeFn | None = None,
) -> dict[str, Any]:
    prompt = JUDGE_PROMPT.format(
        question=question,
        expected_answer=expected_answer or "暂无完整参考答案，请根据题意和失分点严格判断。",
        pitfalls=json.dumps(pitfalls or [], ensure_ascii=False),
        actual_answer=actual_answer,
    )
    judge = judge_fn or (lambda payload: call_text(payload, task="judge"))

    try:
        payload = parse_json_response(judge(prompt))
    except Exception as exc:
        return _fallback_result(question, actual_answer, exc)

    result = {
        "question": question,
        "expected": expected_answer,
        "actual": actual_answer,
        "accuracy_score": _clamp_score(payload.get("accuracy_score")),
        "clarity_score": _clamp_score(payload.get("clarity_score")),
        "depth_score": _clamp_score(payload.get("depth_score")),
        "evidence_score": _clamp_score(payload.get("evidence_score")),
        "structure_score": _clamp_score(payload.get("structure_score")),
        "reason": str(payload.get("reason", "")).strip(),
    }
    result["overall_score"] = round(
        (
            result["accuracy_score"]
            + result["clarity_score"]
            + result["depth_score"]
            + result["evidence_score"]
            + result["structure_score"]
        )
        / 5,
        1,
    )
    if not result["reason"]:
        result["reason"] = "回答还可以继续加强结论、细节和证据支撑。"
    return result


def render_review_summary(results: list[dict[str, Any]]) -> str:
    return render_review_summary_with_drill(results)


def render_review_summary_with_drill(
    results: list[dict[str, Any]],
    drill_plan: list[dict[str, Any]] | None = None,
) -> str:
    if not results:
        return "No structured review results."

    average = round(sum(item["overall_score"] for item in results) / len(results), 1)
    lines = [
        "# Interview Review",
        "",
        f"- reviewed: {len(results)}",
        f"- average_overall: {average}/5",
        "",
    ]
    for index, item in enumerate(results, 1):
        lines.append(f"{index}. {item['question']}")
        lines.append(f"   - overall: {item['overall_score']}/5")
        lines.append(f"   - accuracy: {item['accuracy_score']}/5")
        lines.append(f"   - reason: {item['reason']}")

    drills = drill_plan if drill_plan is not None else build_drill_plan(results)
    if drills:
        lines.extend(["", "## Next Drill", ""])
        for index, item in enumerate(drills, 1):
            lines.append(f"{index}. {item['question']}")
            lines.append(f"   - focus: {item['focus']}")
            lines.append(f"   - reason: {item['reason']}")
    return "\n".join(lines).rstrip() + "\n"


def build_drill_plan(results: list[dict[str, Any]], *, limit: int = 3) -> list[dict[str, Any]]:
    if not results:
        return []

    ranked = sorted(
        results,
        key=lambda item: (
            float(item.get("overall_score", 0.0)),
            int(item.get("evidence_score", 0)),
            int(item.get("depth_score", 0)),
            int(item.get("clarity_score", 0)),
        ),
    )
    drills: list[dict[str, Any]] = []
    for item in ranked:
        question = str(item.get("question", "")).strip()
        if not question or any(existing["question"] == question for existing in drills):
            continue
        drills.append(
            {
                "question": question,
                "focus": _drill_focus(item),
                "reason": str(item.get("reason", "")).strip()
                or "Strengthen this answer with clearer structure and stronger evidence.",
            }
        )
        if len(drills) >= limit:
            break
    return drills


def _drill_focus(item: dict[str, Any]) -> str:
    evidence = int(item.get("evidence_score", 0))
    depth = int(item.get("depth_score", 0))
    clarity = int(item.get("clarity_score", 0))
    accuracy = int(item.get("accuracy_score", 0))

    if evidence <= min(depth, clarity, accuracy):
        return "add concrete evidence, metrics, and implementation detail"
    if depth <= min(clarity, accuracy):
        return "go one layer deeper on design trade-offs and edge cases"
    if clarity <= accuracy:
        return "tighten the structure: lead with conclusion, then expand"
    return "fix the core technical explanation before optimizing delivery"


def _clamp_score(value: Any) -> int:
    try:
        return max(1, min(5, int(value)))
    except Exception:
        return 1


def _fallback_result(question: str, actual_answer: str, err: Exception) -> dict[str, Any]:
    score = 2 if len(actual_answer.strip()) >= 40 else 1
    return {
        "question": question,
        "expected": "",
        "actual": actual_answer,
        "accuracy_score": score,
        "clarity_score": score,
        "depth_score": score,
        "evidence_score": 1,
        "structure_score": score,
        "overall_score": float(score),
        "reason": f"裁判模型失败，使用保守兜底评分: {err}",
    }
