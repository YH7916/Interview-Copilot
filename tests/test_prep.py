import json
from pathlib import Path

from copilot.prep import render_interview_prep_pack


def test_render_interview_prep_pack_includes_key_sections(tmp_path):
    profile = "\n".join(
        [
            "Candidate Snapshot",
            "- Education: Zhejiang University",
            "- Stage: Undergraduate, class of 2028",
            "- Project 1: Interview Copilot | Ownership: built the runtime and retrieval flow | Tech: Python, RAG, agent | Deep Dive: orchestration, trace, evaluation",
            "- Project 2: CuteGo | Ownership: built frontend inference flow | Tech: TypeScript, WebWorker | Deep Dive: runtime isolation, responsiveness",
        ]
    )

    result = render_interview_prep_pack(
        topic="agent",
        company="ByteDance",
        position="AI Agent Intern",
        candidate_profile=profile,
        recent=False,
        max_questions=6,
        trace_dir=tmp_path,
    )

    assert "# Interview Prep Pack" in result
    assert "- trace:" in result
    assert "## Candidate Snapshot" in result
    assert "## Lead With These Projects" in result
    assert "Interview Copilot" in result
    assert "## Target Gap Analysis" in result
    assert "## Training Plan" in result
    assert "## Seed Questions" in result


def test_render_interview_prep_pack_surfaces_common_evidence_gaps(tmp_path):
    profile = "\n".join(
        [
            "Candidate Snapshot",
            "- Project 1: Mock Agent | Tech: Python, tool, workflow",
        ]
    )

    result = render_interview_prep_pack(
        topic="agent",
        candidate_profile=profile,
        recent=False,
        max_questions=4,
        trace_dir=tmp_path,
    )

    assert "Prepare 2-3 concrete metrics" in result
    assert "Prepare one trade-off story" in result


def test_render_interview_prep_pack_highlights_role_gaps(tmp_path):
    profile = "\n".join(
        [
            "Candidate Snapshot",
            "- Project 1: CuteGo | Ownership: built frontend inference flow | Tech: TypeScript, WebWorker",
        ]
    )

    result = render_interview_prep_pack(
        topic="agent",
        position="AI Agent Intern",
        target_description="Need agent orchestration, retrieval, evaluation, and tracing experience.",
        candidate_profile=profile,
        recent=False,
        max_questions=4,
        trace_dir=tmp_path,
    )

    assert "Weak or missing in your profile today" in result
    assert "retrieval" in result.lower()
    assert "evaluation" in result.lower()


def test_render_interview_prep_pack_writes_trace_file(tmp_path):
    result = render_interview_prep_pack(
        topic="agent",
        company="ByteDance",
        position="AI Agent Intern",
        target_description="Need agent orchestration and retrieval experience.",
        candidate_profile="Candidate Snapshot\n- Project 1: Interview Copilot | Ownership: built runtime | Tech: Python, agent",
        recent=False,
        max_questions=4,
        trace_dir=tmp_path,
    )

    trace_line = next(line for line in result.splitlines() if line.startswith("- trace: "))
    trace_path = trace_line.split(": ", 1)[1].strip()
    payload = json.loads(Path(trace_path).read_text(encoding="utf-8"))

    assert payload["session_id"].startswith("prep-")
    assert payload["target"]["company"] == "ByteDance"
    assert payload["seed_questions"]
