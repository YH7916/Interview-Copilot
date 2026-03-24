"""Interview orchestration: planning, live harness runtime, review, and trace."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from copilot.interview.evaluation import build_drill_plan, evaluate_answer, render_review_summary_with_drill
from copilot.interview.interviewer import LLMInterviewer
from copilot.interview.policy import decide_next_action, is_clarification_request, should_follow_up
from copilot.interview.planner import InterviewPlanner, PlannedQuestion, render_plan
from copilot.interview.selector import LLMQuestionSelector
from copilot.interview.session import InterviewSession
from copilot.interview.state import InterviewGoalState, build_goal_state
from copilot.interview.trace import TraceTurn, create_interview_trace, save_interview_trace
from copilot.knowledge.answer_cards import build_answer_card_index, find_answer_card, load_answer_cards_or_empty


@dataclass(slots=True)
class ReviewedQuestion:
    question: str
    category: str
    stage: str
    answer_status: str
    candidate_answer: str
    reference_answer: str
    pitfalls: list[str]
    accuracy_score: int
    clarity_score: int
    depth_score: int
    evidence_score: int
    structure_score: int
    overall_score: float
    reason: str


@dataclass(slots=True)
class InterviewTraceEntry:
    question: str
    answer: str
    policy_action: str = ""
    policy_reason: str = ""
    follow_up: str = ""
    follow_up_answer: str = ""
    review: ReviewedQuestion | None = None


class InterviewRunner:
    def __init__(
        self,
        planner: InterviewPlanner | None = None,
        tracker: Any | None = None,
        *,
        recent: bool = True,
    ):
        self.planner = planner or InterviewPlanner(recent=recent)
        self.tracker = tracker
        self.answer_cards = load_answer_cards_or_empty(recent=recent)
        self.answer_card_index = build_answer_card_index(self.answer_cards)

    def plan(self, session: InterviewSession, max_questions: int = 8) -> list[PlannedQuestion]:
        return self.planner.plan(session, max_questions=max_questions)

    def review_question_answer(self, item: PlannedQuestion, candidate_answer: str) -> ReviewedQuestion:
        return self._review_question(item, candidate_answer)

    def summarize_reviewed(
        self,
        reviewed: list[ReviewedQuestion],
        *,
        persist: bool = True,
    ) -> dict[str, Any]:
        return self._finalize(reviewed, persist=persist)

    def review_answers(
        self,
        plan: list[PlannedQuestion],
        answers: list[str],
        *,
        persist: bool = True,
    ) -> dict[str, Any]:
        reviewed = [
            self._review_question(item, answers[index])
            for index, item in enumerate(plan)
            if index < len(answers) and answers[index].strip()
        ]
        return self._finalize(reviewed, persist=persist)

    def review_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        persist: bool = True,
    ) -> dict[str, Any]:
        reviewed = []
        for pair in extract_question_answer_pairs(messages):
            card = find_answer_card(pair["question"], index=self.answer_card_index)
            reviewed.append(
                self._review_reference(
                    question=pair["question"],
                    category=str((card or {}).get("category", "")),
                    stage=str((card or {}).get("stage", "")),
                    answer_status=str((card or {}).get("status", "missing")),
                    candidate_answer=pair["answer"],
                    reference_answer=str((card or {}).get("answer", "")),
                    pitfalls=list((card or {}).get("pitfalls", [])),
                )
            )
        return self._finalize(reviewed, persist=persist)

    def _review_question(self, item: PlannedQuestion, candidate_answer: str) -> ReviewedQuestion:
        return self._review_reference(
            question=item.question,
            category=item.category,
            stage=item.stage,
            answer_status=item.answer_status,
            candidate_answer=candidate_answer,
            reference_answer=item.reference_answer,
            pitfalls=item.pitfalls,
        )

    @staticmethod
    def _review_reference(
        *,
        question: str,
        category: str,
        stage: str,
        answer_status: str,
        candidate_answer: str,
        reference_answer: str,
        pitfalls: list[str],
    ) -> ReviewedQuestion:
        result = evaluate_answer(
            question,
            reference_answer,
            candidate_answer,
            pitfalls=pitfalls,
        )
        return ReviewedQuestion(
            question=question,
            category=category,
            stage=stage,
            answer_status=answer_status,
            candidate_answer=candidate_answer,
            reference_answer=reference_answer,
            pitfalls=pitfalls,
            accuracy_score=int(result["accuracy_score"]),
            clarity_score=int(result["clarity_score"]),
            depth_score=int(result["depth_score"]),
            evidence_score=int(result["evidence_score"]),
            structure_score=int(result["structure_score"]),
            overall_score=float(result["overall_score"]),
            reason=str(result["reason"]),
        )

    def _finalize(self, reviewed: list[ReviewedQuestion], *, persist: bool) -> dict[str, Any]:
        results = [asdict(item) for item in reviewed]
        if persist and results and self.tracker is not None:
            self.tracker.update(results)
        average = round(sum(item["overall_score"] for item in results) / max(len(results), 1), 1)
        drill_plan = build_drill_plan(results)
        return {
            "count": len(results),
            "average_overall": average,
            "results": results,
            "drill_plan": drill_plan,
            "summary": render_review_summary_with_drill(results, drill_plan=drill_plan),
        }


class InterviewHarness:
    def __init__(
        self,
        runner: InterviewRunner,
        session: InterviewSession,
        *,
        max_questions: int = 6,
        topic: str = "",
        trace_dir: Path | None = None,
        interviewer: LLMInterviewer | None = None,
        selector: LLMQuestionSelector | None = None,
    ):
        self.runner = runner
        self.session = session
        self.plan = runner.plan(session, max_questions=max_questions)
        self.topic = topic
        self.trace_dir = trace_dir
        self.interviewer = interviewer or LLMInterviewer()
        self.selector = selector or LLMQuestionSelector()
        self.current_index = 0
        self.pending_follow_up = ""
        self.pending_answer = ""
        self.pending_policy_reason = ""
        self._rendered_questions: dict[int, str] = {}
        self.trace: list[InterviewTraceEntry] = []
        self.goal_state = InterviewGoalState()
        self.trace_record = create_interview_trace(user_id=session.user_id, topic=topic)
        self.trace_path: Path | None = None
        self.completed = not self.plan

    def open(self) -> str:
        if not self.plan:
            return "No interview questions available right now. Try /ingest 7 first."
        self._refresh_goal_state()
        self._select_question_for_current_slot()
        return "\n".join(
            [
                "# Live Interview",
                "",
                f"- mode: {self.session.mode}",
                f"- questions: {len(self.plan)}",
                "- rule: answer naturally, I may ask follow-ups based on your answer",
                "",
                self._render_question_block(self.plan[self.current_index], index=self.current_index + 1),
            ]
        )

    def reply(self, answer: str) -> str:
        if self.completed:
            return "Interview already finished. Run /review for the full report or /interview to start a new one."

        answer = answer.strip()
        if not answer:
            return "Please answer the current question first."
        current = self.plan[self.current_index]
        if is_clarification_request(answer):
            self._refresh_goal_state()
            return self._render_clarification(current, answer)

        self.session.next_turn()
        if self.pending_follow_up:
            combined_answer = f"{self.pending_answer}\n补充回答：{answer}"
            review = self.runner.review_question_answer(current, combined_answer)
            decision = decide_next_action(
                has_follow_ups=True,
                answer_text=combined_answer,
                depth_score=review.depth_score,
                evidence_score=review.evidence_score,
                overall_score=review.overall_score,
                question=current.question,
                category=current.category,
            )
            self.trace.append(
                InterviewTraceEntry(
                    question=current.question,
                    answer=self.pending_answer,
                    policy_action="follow_up",
                    policy_reason=self.pending_policy_reason or decision.reason,
                    follow_up=self.pending_follow_up,
                    follow_up_answer=answer,
                    review=review,
                )
            )
            self.trace_record.add_turn(
                TraceTurn(
                    index=len(self.trace_record.turns) + 1,
                    stage=current.stage,
                    category=current.category,
                    question=current.question,
                    answer=self.pending_answer,
                    follow_up=self.pending_follow_up,
                    follow_up_answer=answer,
                    policy_action="follow_up",
                    policy_reason=self.pending_policy_reason or decision.reason,
                    review=asdict(review),
                )
            )
            self.pending_follow_up = ""
            self.pending_answer = ""
            self.pending_policy_reason = ""
            self._refresh_goal_state()
            return self._advance()

        review = self.runner.review_question_answer(current, answer)
        decision = decide_next_action(
            has_follow_ups=bool(current.follow_ups),
            answer_text=review.candidate_answer,
            depth_score=review.depth_score,
            evidence_score=review.evidence_score,
            overall_score=review.overall_score,
            question=current.question,
            category=current.category,
        )
        if decision.action == "follow_up":
            self.pending_answer = answer
            self.pending_policy_reason = decision.reason
            self._refresh_goal_state()
            fallback_follow_up = self._build_follow_up_prompt(current, answer, decision.reason)
            self.pending_follow_up = self.interviewer.render_follow_up(
                item=current,
                session=self.session,
                candidate_answer=answer,
                policy_reason=decision.reason,
                goal_state=self.goal_state,
                fallback=fallback_follow_up,
                history=self._history_for_prompt(),
            )
            return "\n".join(
                [
                    "继续追问一个细节：",
                    self.pending_follow_up,
                ]
            )

        self.trace.append(
            InterviewTraceEntry(
                question=current.question,
                answer=answer,
                policy_action=decision.action,
                policy_reason=decision.reason,
                review=review,
            )
        )
        self.trace_record.add_turn(
            TraceTurn(
                index=len(self.trace_record.turns) + 1,
                stage=current.stage,
                category=current.category,
                question=current.question,
                answer=answer,
                policy_action=decision.action,
                policy_reason=decision.reason,
                review=asdict(review),
            )
        )
        self._refresh_goal_state()
        return self._advance()

    def review(self, *, persist: bool = False) -> dict[str, Any]:
        reviewed = [entry.review for entry in self.trace if entry.review is not None]
        return self.runner.summarize_reviewed(reviewed, persist=persist) if reviewed else {
            "count": 0,
            "average_overall": 0.0,
            "results": [],
            "drill_plan": [],
            "summary": "No completed interview turns yet.",
        }

    def export_trace(self) -> list[dict[str, Any]]:
        return [
            {
                "question": entry.question,
                "answer": entry.answer,
                "policy_action": entry.policy_action,
                "policy_reason": entry.policy_reason,
                "follow_up": entry.follow_up,
                "follow_up_answer": entry.follow_up_answer,
                "review": asdict(entry.review) if entry.review else None,
            }
            for entry in self.trace
        ]

    def _advance(self) -> str:
        self.current_index += 1
        if self.current_index >= len(self.plan):
            self.completed = True
            report = self.review(persist=False)
            self.trace_record.finalize(report)
            self.trace_path = save_interview_trace(
                self.trace_record,
                **({"trace_dir": self.trace_dir} if self.trace_dir else {}),
            )
            return "\n".join(
                [
                    "Interview finished.",
                    f"- reviewed: {report['count']}",
                    f"- average_overall: {report['average_overall']}/5",
                    f"- trace: {self.trace_path}" if self.trace_path else "",
                    "- next_step: run /review for the full report",
                ]
            ).replace("\n\n", "\n")
        self._refresh_goal_state()
        self._select_question_for_current_slot()
        return self._render_question_block(self.plan[self.current_index], index=self.current_index + 1)

    @staticmethod
    def _should_follow_up(item: PlannedQuestion, review: ReviewedQuestion) -> bool:
        return should_follow_up(
            has_follow_ups=bool(item.follow_ups),
            answer_text=review.candidate_answer,
            depth_score=review.depth_score,
            evidence_score=review.evidence_score,
            overall_score=review.overall_score,
            question=item.question,
            category=item.category,
        )

    @staticmethod
    def _render_question(item: PlannedQuestion, *, index: int) -> str:
        lines = [
            f"[Q{index}] [{item.stage_label} | {item.category_label}] {item.question}",
            f"- signal: {item.source_count} source(s)",
        ]
        if item.latest_source_at:
            lines.append(f"- latest: {item.latest_source_at[:10]}")
        return "\n".join(lines)

    def _render_clarification(self, item: PlannedQuestion, candidate_message: str) -> str:
        fallback = self._render_clarification_fallback(item)
        return self.interviewer.render_clarification(
            item=item,
            session=self.session,
            candidate_message=candidate_message,
            goal_state=self.goal_state,
            fallback=fallback,
            history=self._history_for_prompt(),
        )

    def _render_clarification_fallback(self, item: PlannedQuestion) -> str:
        question = item.question
        if "调度" in question and "策略" in question:
            return (
                "可以结合你最熟悉、最有细节的那个项目来回答。"
                "我更想听的是任务怎么分发、优先级怎么定，以及超时或失败时怎么兜底。"
            )
        if "自我介绍" in question:
            return "就按 1 分钟左右讲你的背景、项目亮点，以及你为什么想做 AI Agent 就可以。"
        if "记忆" in question:
            return (
                "就拿你最熟悉的 agent 项目来讲，重点说短期记忆、长期记忆，"
                "以及写入和召回策略分别怎么设计。"
            )
        return "可以结合你最熟悉、最有细节的那个项目来回答，我更关注你的实际设计和取舍。"

    def _build_follow_up_prompt(
        self,
        item: PlannedQuestion,
        answer: str,
        reason: str,
    ) -> str:
        if reason == "missing_graduation_timeline":
            return "方便再补充一下你的预计毕业时间吗？"
        if reason == "missing_school_context":
            return "再补充一下你的学校和专业背景。"
        if reason == "missing_current_stage":
            return "再补充一下你现在的培养阶段，比如本科几年级或硕士阶段。"
        if reason == "need_project_example":
            return "别只讲概念，拿一个你亲手做过的项目展开一下：场景是什么、你负责什么、最后效果怎样？"
        if reason == "need_execution_detail":
            question = str(item.question or "").lower()
            if "webworker" in question:
                return (
                    "如果结合你刚才提到的项目，WebWorker 具体承载了哪些计算？"
                    "它和主线程怎么通信，为什么要这样拆？"
                )
            if "记忆" in item.question:
                return (
                    "具体一点：什么信息只放在上下文窗口里，什么信息会写入长期记忆？"
                    "写入、更新和召回分别在什么时机触发？"
                )
            if "调度" in item.question and "策略" in item.question:
                return "你就拿一个具体项目讲：任务怎么分发和排队，优先级怎么定，资源冲突时怎么处理？"
            return "再往下讲一层，重点说清楚具体设计、模块拆分和关键执行细节。"
        if reason == "need_system_fallback_strategy":
            return (
                "除了给用户报错，你们系统层面还有什么兜底策略？"
                "比如超时重试、降级、回退默认流程，或者切换更轻量的方案。"
            )
        if reason == "need_deeper_reasoning":
            return self._deeper_reasoning_follow_up(item)
        if reason == "need_concrete_evidence":
            return self._evidence_follow_up(item)

        return self._default_follow_up_fallback(item, answer)

    @staticmethod
    def _polish_follow_up_text(follow_up: str, answer: str) -> str:
        text = str(follow_up or "").strip()
        if not text:
            return "可以结合一个真实项目，再展开一下你的设计思路、取舍和最终结果。"
        if text.endswith(("?", "？")):
            return text
        if re.search(r"(讲讲|介绍|谈谈|展开|补充)", text):
            return text
        if any(token in answer for token in ("项目", "比如", "例如")):
            return f"结合你刚才提到的项目，{text}"
        return text

    @staticmethod
    def _deeper_reasoning_follow_up(item: PlannedQuestion) -> str:
        question = str(item.question or "")
        category = str(item.category or "")
        if category == "opening":
            return "可以再用 1 分钟把你的背景和两个最有分量的项目亮点捋一下，别只给结论。"
        if "记忆" in question or category == "agent_architecture":
            return "再往下展开一层，重点说你为什么这样划分记忆层次，这个取舍带来了什么好处和代价。"
        if "调度" in question or "策略" in question or category == "project_architecture":
            return "可以再往下拆一层，说一下你当时的设计考虑、优先级取舍，以及为什么没选另一种方案。"
        if "评测" in question or "评估" in question or category == "project_evaluation":
            return "再展开一点，说一下你评测集怎么切分、标签怎么定，以及你怎么判断这套评估是真的有用。"
        return "可以再往下展开一层，重点说一下你的设计取舍和为什么这么做。"

    @staticmethod
    def _evidence_follow_up(item: PlannedQuestion) -> str:
        question = str(item.question or "")
        lowered = question.lower()
        if "webworker" in lowered:
            return "最好带一个具体结果，比如卡不卡、动画流畅度怎么变化，或者你怎么判断主线程压力确实下来了。"
        if "评测" in question or "评估" in question:
            return "最好带一个实际指标或对比结果，比如重复率、命中缺口的比例，或者评测结果怎么反向修改了系统。"
        return "最好带一个真实项目细节或指标，比如时延、命中率、错误率，或者成本变化。"

    def _default_follow_up_fallback(self, item: PlannedQuestion, answer: str) -> str:
        if item.follow_ups:
            return self._polish_follow_up_text(item.follow_ups[0], answer)
        category = str(item.category or "")
        if category == "project_background":
            return "可以结合一个真实项目的场景再讲一下，说清楚你负责什么、为什么要做，最后效果怎样。"
        if category in {"project_architecture", "agent_architecture", "rag_retrieval"}:
            return "可以结合一个真实项目，再展开一下你的设计思路、模块拆分和关键取舍。"
        if category in {"project_evaluation", "project_deployment", "project_challenges"}:
            return "再补充一下你是怎么判断效果、怎么处理难点，以及最后的结果是什么。"
        return "可以结合一个真实项目，再展开一下你的设计思路、取舍和最终结果。"

    def _history_for_prompt(self) -> list[dict[str, str]]:
        history = []
        for entry in self.trace[-2:]:
            history.append(
                {
                    "question": entry.question,
                    "category": entry.review.category if entry.review else "",
                    "answer": entry.answer,
                    "follow_up": entry.follow_up,
                    "follow_up_answer": entry.follow_up_answer,
                    "policy_reason": entry.policy_reason,
                    "overall_score": (
                        f"{entry.review.overall_score:.1f}" if entry.review is not None else ""
                    ),
                }
            )
        return history

    def _select_question_for_current_slot(self) -> None:
        if self.current_index >= len(self.plan):
            return
        current = self.plan[self.current_index]
        if self.current_index == 0 and current.stage in {"opening", "coding"}:
            return
        remaining = self.plan[self.current_index :]
        if len(remaining) <= 1:
            return
        chosen_offset = self.selector.select_next_question(
            session=self.session,
            candidates=remaining,
            goal_state=self.goal_state,
            history=self._history_for_prompt(),
        )
        if chosen_offset <= 0:
            return
        chosen_index = self.current_index + chosen_offset
        self.plan[self.current_index], self.plan[chosen_index] = self.plan[chosen_index], self.plan[self.current_index]

    def _render_question_block(self, item: PlannedQuestion, *, index: int) -> str:
        rendered_question = self._rendered_questions.get(index)
        if not rendered_question:
            fallback = item.question
            rendered_question = self.interviewer.render_question(
                item=item,
                session=self.session,
                index=index,
                total_questions=len(self.plan),
                goal_state=self.goal_state,
                history=self._history_for_prompt(),
                fallback=fallback,
            )
            self._rendered_questions[index] = rendered_question

        lines = [
            f"[Q{index}] [{item.stage_label} | {item.category_label}] {rendered_question}",
            f"- signal: {item.source_count} source(s)",
        ]
        if item.latest_source_at:
            lines.append(f"- latest: {item.latest_source_at[:10]}")
        return "\n".join(lines)

    def _refresh_goal_state(self) -> None:
        self.goal_state = build_goal_state(
            session=self.session,
            trace=self.trace,
            pending_policy_reason=self.pending_policy_reason,
        )


def extract_question_answer_pairs(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    pairs = []
    pending_question = ""

    for message in messages:
        role = str(message.get("role", ""))
        content = _message_text(message).strip()
        if not content:
            continue

        if role == "assistant":
            question = _pick_question_text(content)
            if question:
                pending_question = question
            continue

        if role == "user" and pending_question:
            pairs.append({"question": pending_question, "answer": content})
            pending_question = ""

    return pairs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a lightweight interview plan and review loop.")
    parser.add_argument("--user-id", default="demo")
    parser.add_argument("--round", dest="round_index", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--company", default="")
    parser.add_argument("--position", default="")
    parser.add_argument("--focus-topic", action="append", dest="focus_topics", default=[])
    parser.add_argument("--max-questions", type=int, default=5)
    parser.add_argument("--all-history", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-persist", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    runner = InterviewRunner(recent=not args.all_history)
    session = InterviewSession(
        user_id=args.user_id,
        round_index=args.round_index,
        company=args.company,
        position=args.position,
        focus_topics=args.focus_topics,
    )
    plan = runner.plan(session, max_questions=args.max_questions)

    print(render_plan(plan))
    print("")

    answers = []
    for index, item in enumerate(plan, 1):
        print(f"[Q{index}] {item.question}")
        answers.append(input("Your answer> ").strip())
        print("")

    report = runner.review_answers(plan, answers, persist=not args.no_persist)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(report["summary"])
    return 0


def _pick_question_text(text: str) -> str:
    lines = [line.strip(" -") for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if "?" in line or "？" in line:
            return line
    if not lines:
        return ""
    return lines[-1] if len(lines[-1]) <= 120 else lines[-1][:120]


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        return " ".join(item.get("text", "") for item in content if isinstance(item, dict))
    return str(content or "")


if __name__ == "__main__":
    raise SystemExit(main())
