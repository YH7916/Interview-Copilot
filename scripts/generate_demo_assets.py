"""Generate reproducible demo assets for Interview Copilot."""

from __future__ import annotations

import json
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from copilot.interview.evaluation import build_drill_plan, render_review_summary_with_drill
from copilot.interview.orchestrator import InterviewHarness, ReviewedQuestion
from copilot.interview.planner import PlannedQuestion
from copilot.interview.session import InterviewSession
from copilot.prep import render_interview_prep_pack

DEMO_DIR = ROOT / "examples" / "demo"
DEMO_RESUME_PATH = DEMO_DIR / "demo_resume.typ"
FIXED_TIMESTAMP = "2026-03-21T00:00:00"


class DemoInterviewer:
    """Minimal interviewer surface for deterministic demo generation."""

    def render_question(self, *, fallback: str, **_: object) -> str:
        return fallback

    def render_follow_up(
        self,
        *,
        item: PlannedQuestion,
        policy_reason: str,
        fallback: str,
        **_: object,
    ) -> str:
        reason_prefix = {
            "need_deeper_reasoning": "Let's go one layer deeper.",
            "need_concrete_evidence": "Make this more concrete.",
            "need_execution_detail": "Give me the implementation detail here.",
            "need_system_fallback_strategy": "Describe the actual fallback path.",
        }.get(policy_reason, "")
        if item.follow_ups:
            return f"{reason_prefix} {item.follow_ups[0]}".strip()
        return fallback

    def render_clarification(self, *, fallback: str, **_: object) -> str:
        return fallback


class DemoSelector:
    """Keep the demo question order fixed."""

    def select_next_question(self, **_: object) -> int:
        return 0


class DemoRunner:
    """Small runner that feeds deterministic review results into the harness."""

    def __init__(self, plan: list[PlannedQuestion]):
        self._plan = plan

    def plan(self, session: InterviewSession, max_questions: int = 6) -> list[PlannedQuestion]:
        del session
        return self._plan[:max_questions]

    def review_question_answer(self, item: PlannedQuestion, candidate_answer: str) -> ReviewedQuestion:
        lower_answer = candidate_answer.lower()
        result = _review_payload(item.question, lower_answer)
        return ReviewedQuestion(
            question=item.question,
            category=item.category,
            stage=item.stage,
            answer_status=item.answer_status,
            candidate_answer=candidate_answer,
            reference_answer=item.reference_answer,
            pitfalls=item.pitfalls,
            accuracy_score=result["accuracy_score"],
            clarity_score=result["clarity_score"],
            depth_score=result["depth_score"],
            evidence_score=result["evidence_score"],
            structure_score=result["structure_score"],
            overall_score=result["overall_score"],
            reason=result["reason"],
        )

    def summarize_reviewed(self, reviewed: list[ReviewedQuestion], *, persist: bool = False) -> dict[str, object]:
        del persist
        results = [asdict(item) for item in reviewed]
        average = round(sum(item["overall_score"] for item in results) / max(len(results), 1), 1)
        drill_plan = build_drill_plan(results)
        return {
            "count": len(results),
            "average_overall": average,
            "results": results,
            "drill_plan": drill_plan,
            "summary": render_review_summary_with_drill(results, drill_plan=drill_plan),
        }


def generate_demo_assets(output_dir: Path = DEMO_DIR) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not DEMO_RESUME_PATH.exists():
        raise FileNotFoundError(f"Demo resume not found: {DEMO_RESUME_PATH}")
    prep_path = output_dir / "demo_prep.md"
    prep_trace_path = output_dir / "demo_prep_trace.json"
    transcript_path = output_dir / "demo_mock_interview.md"
    review_path = output_dir / "demo_review.md"
    interview_trace_path = output_dir / "demo_interview_trace.json"

    with tempfile.TemporaryDirectory(prefix="interview-copilot-demo-") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        prep_text = render_interview_prep_pack(
            topic="agent",
            company="ByteDance",
            position="AI Agent Intern",
            target_description="Need agent orchestration, retrieval, evaluation, tracing, and strong project storytelling.",
            candidate_profile_path=str(DEMO_RESUME_PATH),
            recent=False,
            max_questions=6,
            trace_dir=tmp_dir,
        )
        prep_trace_source = _extract_trace_path(prep_text)
        if prep_trace_source is None:
            raise RuntimeError("Prep output did not contain a trace path.")

        prep_trace = json.loads(prep_trace_source.read_text(encoding="utf-8"))
        prep_trace = _sanitize_demo_value(prep_trace)
        prep_trace["session_id"] = "demo-prep"
        prep_trace["started_at"] = FIXED_TIMESTAMP
        prep_trace["completed_at"] = FIXED_TIMESTAMP
        prep_trace_path.write_text(json.dumps(prep_trace, ensure_ascii=False, indent=2), encoding="utf-8")

        prep_text = prep_text.replace(str(prep_trace_source), "demo_prep_trace.json")
        prep_path.write_text(_render_prep_markdown(prep_text), encoding="utf-8")

        harness = InterviewHarness(
            runner=DemoRunner(_build_demo_plan()),
            session=InterviewSession(
                user_id="demo",
                focus_topics=["agent", "retrieval", "evaluation"],
                candidate_profile=DEMO_RESUME_PATH.read_text(encoding="utf-8"),
                interview_style="auto",
            ),
            max_questions=4,
            topic="agent",
            trace_dir=tmp_dir,
            interviewer=DemoInterviewer(),
            selector=DemoSelector(),
        )

        transcript_lines: list[str] = []
        opening = harness.open()
        transcript_lines.extend(_dialogue_block("Copilot", opening))

        answers = [
            (
                "I am a sophomore in computer science at Zhejiang University graduating in 2028. "
                "My main project is Interview Copilot, a retrieval-augmented interview harness built on top of nanobot, "
                "and I also built CuteGo to keep browser-side inference responsive with WebWorker.",
                None,
            ),
            (
                "Nanobot keeps long-term memory, while copilot owns explicit runtime state and replayable traces. "
                "That keeps routing policy inspectable and avoids hiding control flow inside another memory layer.",
                None,
            ),
            (
                "We start with project-aware routing and then fall back to a simpler path when the signal is weak.",
                (
                    "We keep an explicit goal state with active project, phase, and recent weak points. "
                    "If confidence is low or a project has already been pushed far enough, the harness switches to another project "
                    "or a different interview dimension instead of forcing more shallow follow-ups."
                ),
            ),
            (
                "Each answer is scored on accuracy, clarity, depth, evidence, and structure. "
                "The weakest answers are converted into a next-drill list so the next session starts from the real gap instead of a random new question.",
                None,
            ),
        ]

        for answer, follow_up_answer in answers:
            transcript_lines.extend(_dialogue_block("You", answer))
            response = harness.reply(answer)
            transcript_lines.extend(_dialogue_block("Copilot", response))
            if follow_up_answer and not response.startswith("[Q") and "Interview finished." not in response:
                transcript_lines.extend(_dialogue_block("You", follow_up_answer))
                response = harness.reply(follow_up_answer)
                transcript_lines.extend(_dialogue_block("Copilot", response))

        review = harness.review(persist=False)
        if harness.trace_path is None:
            raise RuntimeError("Interview trace was not created.")

        interview_trace = _sanitize_demo_value(json.loads(harness.trace_path.read_text(encoding="utf-8")))
        interview_trace["session_id"] = "demo-interview"
        interview_trace["started_at"] = FIXED_TIMESTAMP
        interview_trace["completed_at"] = FIXED_TIMESTAMP
        interview_trace_path.write_text(
            json.dumps(interview_trace, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        transcript_path.write_text(
            _render_transcript_markdown(transcript_lines, review),
            encoding="utf-8",
        )
        review_path.write_text(
            _render_review_markdown(review["summary"]),
            encoding="utf-8",
        )

    return {
        "prep": str(prep_path),
        "prep_trace": str(prep_trace_path),
        "transcript": str(transcript_path),
        "review": str(review_path),
        "interview_trace": str(interview_trace_path),
    }


def _build_demo_plan() -> list[PlannedQuestion]:
    return [
        PlannedQuestion(
            stage="opening",
            stage_label="Opening",
            category="project_background",
            category_label="Opening and Background",
            question="Tell me about yourself and why you are applying for AI agent internships.",
            follow_ups=[],
            source_count=5,
            latest_source_at="2026-03-20T10:00:00",
            answer_status="grounded",
            reference_answer="Lead with school stage, strongest project, second supporting project, and why AI agent work is a fit.",
            pitfalls=[
                "Too generic and not tied to real projects.",
                "No clear connection between background and target role.",
            ],
        ),
        PlannedQuestion(
            stage="project",
            stage_label="Project Deep Dive",
            category="agent_architecture",
            category_label="Agent Architecture and Skills",
            question="How did you design the memory boundary in Interview Copilot?",
            follow_ups=["What belongs in nanobot memory versus copilot state and trace?"],
            source_count=4,
            latest_source_at="2026-03-20T10:00:00",
            answer_status="grounded",
            reference_answer="Explain the split between long-term memory, explicit runtime state, and replayable traces, and why that keeps policy observable.",
            pitfalls=[
                "Mixing long-term memory with runtime control.",
                "Only naming short-term and long-term memory without write/read policy.",
            ],
        ),
        PlannedQuestion(
            stage="project",
            stage_label="Project Deep Dive",
            category="project_architecture",
            category_label="Project Architecture and Runtime",
            question="How do scheduling and fallback work in the runtime?",
            follow_ups=["Give me the concrete switching rule when one project trail is exhausted or weak."],
            source_count=3,
            latest_source_at="2026-03-20T10:00:00",
            answer_status="grounded",
            reference_answer="Describe project-aware routing, active phase tracking, switching thresholds, and the degrade path when evidence is weak.",
            pitfalls=[
                "Only saying the system reports an error to the user.",
                "No concrete routing or degrade rule.",
            ],
        ),
        PlannedQuestion(
            stage="project",
            stage_label="Project Deep Dive",
            category="project_evaluation",
            category_label="Evaluation and Review",
            question="How do you review answers and turn them into the next drill?",
            follow_ups=["Which signals are strongest enough to trigger a targeted drill?"],
            source_count=4,
            latest_source_at="2026-03-20T10:00:00",
            answer_status="grounded",
            reference_answer="Explain structured scoring, weakness extraction, drill generation, and why the next session starts from the weakest answer.",
            pitfalls=[
                "No explicit scoring dimensions.",
                "Review exists but does not change the next session.",
            ],
        ),
    ]


def _review_payload(question: str, candidate_answer: str) -> dict[str, object]:
    if question.startswith("Tell me about yourself"):
        return {
            "accuracy_score": 4,
            "clarity_score": 4,
            "depth_score": 4,
            "evidence_score": 4,
            "structure_score": 4,
            "overall_score": 4.0,
            "reason": "Clear background, strong project anchor, and a good role fit.",
        }
    if question.startswith("How did you design the memory boundary"):
        return {
            "accuracy_score": 4,
            "clarity_score": 4,
            "depth_score": 4,
            "evidence_score": 4,
            "structure_score": 5,
            "overall_score": 4.2,
            "reason": "Good boundary definition and rationale. One concrete failure case would make it even stronger.",
        }
    if question.startswith("How do scheduling and fallback work"):
        if "switches to another project" not in candidate_answer:
            return {
                "accuracy_score": 3,
                "clarity_score": 3,
                "depth_score": 2,
                "evidence_score": 2,
                "structure_score": 3,
                "overall_score": 2.6,
                "reason": "The direction is right, but the actual routing rule and fallback threshold are still too thin.",
            }
        return {
            "accuracy_score": 4,
            "clarity_score": 4,
            "depth_score": 3,
            "evidence_score": 3,
            "structure_score": 4,
            "overall_score": 3.6,
            "reason": "Routing policy is clear. Add one concrete threshold or failure example to make it more convincing.",
        }
    return {
        "accuracy_score": 5,
        "clarity_score": 4,
        "depth_score": 4,
        "evidence_score": 4,
        "structure_score": 5,
        "overall_score": 4.4,
        "reason": "Strong review loop with explicit scoring and a clear handoff into the next drill.",
    }


def _extract_trace_path(text: str) -> Path | None:
    for line in text.splitlines():
        if line.startswith("- trace: "):
            return Path(line.split(": ", 1)[1].strip())
    return None


def _dialogue_block(speaker: str, content: str) -> list[str]:
    normalized = _sanitize_demo_text(content)
    normalized = re.sub(
        r"- trace:\s+[A-Za-z]:[\\/].*?\.json",
        "- trace: demo_interview_trace.json",
        normalized,
    )
    lines = [f"{speaker}: {line}" if index == 0 else f"  {line}" for index, line in enumerate(normalized.splitlines())]
    return lines + [""]


def _render_prep_markdown(prep_text: str) -> str:
    prep_text = _sanitize_demo_text(prep_text)
    return "\n".join(
        [
            "# Demo Prep Run",
            "",
            "Command:",
            "",
            "```text",
            '/prep agent --resume "examples/demo/demo_resume.typ" --company ByteDance --position "AI Agent Intern" --target "Need agent orchestration, retrieval, evaluation, tracing, and strong project storytelling."',
            "```",
            "",
            prep_text.strip(),
            "",
            "Artifacts:",
            "",
            "- prep trace: [demo_prep_trace.json](demo_prep_trace.json)",
        ]
    ).rstrip() + "\n"


def _render_transcript_markdown(transcript_lines: list[str], review: dict[str, object]) -> str:
    summary = _sanitize_demo_text(str(review["summary"]).strip())
    summary = summary.replace("## Next Drill", "### Next Drill")
    return "\n".join(
        [
            "# Demo Mock Interview",
            "",
            "Command:",
            "",
            "```text",
            '/interview agent --resume "examples/demo/demo_resume.typ"',
            "```",
            "",
            "Transcript:",
            "",
            "```text",
            *transcript_lines,
            "```",
            "",
            "Review Snapshot:",
            "",
            summary,
            "",
            "Artifacts:",
            "",
            "- review: [demo_review.md](demo_review.md)",
            "- interview trace: [demo_interview_trace.json](demo_interview_trace.json)",
        ]
    ).rstrip() + "\n"


def _render_review_markdown(summary: str) -> str:
    summary = _sanitize_demo_text(summary)
    return "\n".join(
        [
            "# Demo Review",
            "",
            summary.strip(),
            "",
            "Artifact:",
            "",
            "- interview trace: [demo_interview_trace.json](demo_interview_trace.json)",
        ]
    ).rstrip() + "\n"


def _sanitize_demo_text(text: str) -> str:
    replacements = {
        "鎴戞兂缁х画杩介棶涓€涓偣锛?": "One follow-up:",
        "琛ュ厖鍥炵瓟锛?": "Follow-up answer: ",
    }
    value = str(text)
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value


def _sanitize_demo_value(value: object) -> object:
    if isinstance(value, dict):
        return {key: _sanitize_demo_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_demo_value(item) for item in value]
    if isinstance(value, str):
        return _sanitize_demo_text(value)
    return value


def main() -> int:
    assets = generate_demo_assets()
    for name, path in assets.items():
        print(f"{name}: {Path(path).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
