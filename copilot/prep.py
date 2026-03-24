"""Interview preparation pack builders."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from copilot.interview.planner import InterviewPlanner, PlannedQuestion
from copilot.interview.session import InterviewSession
from copilot.interview.trace import create_prep_trace, save_prep_trace
from copilot.profile import load_candidate_profile, parse_candidate_projects


def render_interview_prep_pack(
    *,
    user_id: str = "nanobot",
    topic: str = "",
    company: str = "",
    position: str = "",
    target_description: str = "",
    max_questions: int = 8,
    recent: bool = True,
    candidate_profile: str = "",
    candidate_profile_path: str = "",
    trace_dir: Path | None = None,
) -> str:
    """Render a structured prep pack from profile and target signals."""
    profile = load_candidate_profile(
        profile_text=candidate_profile,
        profile_path=candidate_profile_path,
    )
    profile_lines = _render_candidate_snapshot(profile)
    session = InterviewSession(
        user_id=user_id,
        company=company,
        position=position,
        focus_topics=[topic] if topic else [],
        candidate_profile=profile,
        interview_style="auto",
    )
    plan = InterviewPlanner(recent=recent).plan(session, max_questions=max_questions)
    projects = parse_candidate_projects(profile)
    project_lines = _render_project_priorities(projects)
    loop_lines = _render_question_loops(plan)
    gap_lines = _render_target_gap_analysis(profile, topic, company, position, target_description)
    evidence_lines = _render_evidence_gaps(profile, projects)
    training_lines = _render_training_plan(topic, company, position, target_description, profile, projects, plan)
    seed_lines = _render_seed_questions(plan)

    prep_trace = create_prep_trace(user_id=user_id, topic=topic or "general")
    prep_trace.target = {
        "company": company,
        "position": position,
        "target_description": target_description,
        "target_signals": _extract_signal_set(topic, company, position, target_description),
    }
    prep_trace.profile_summary = profile_lines
    prep_trace.project_priorities = project_lines
    prep_trace.target_gap_analysis = gap_lines
    prep_trace.evidence_gaps = evidence_lines
    prep_trace.training_plan = training_lines
    prep_trace.seed_questions = seed_lines
    prep_trace.finalize()
    trace_path = save_prep_trace(prep_trace, **({"trace_dir": trace_dir} if trace_dir else {}))

    lines = [
        "# Interview Prep Pack",
        "",
        f"- topic: {topic or 'general'}",
        f"- company: {company or 'unspecified'}",
        f"- position: {position or 'unspecified'}",
        f"- target_signals: {_render_target_signals(topic, company, position, target_description)}",
        f"- trace: {trace_path}",
        "",
        "## Candidate Snapshot",
        *profile_lines,
        "",
        "## Lead With These Projects",
        *project_lines,
        "",
        "## Likely Interview Loops",
        *loop_lines,
        "",
        "## Target Gap Analysis",
        *gap_lines,
        "",
        "## Evidence To Prepare",
        *evidence_lines,
        "",
        "## Training Plan",
        *training_lines,
        "",
        "## Seed Questions",
        *seed_lines,
    ]
    return "\n".join(lines).strip() + "\n"


def _render_candidate_snapshot(profile: str) -> list[str]:
    snapshot_lines = [line.strip() for line in str(profile or "").splitlines() if line.strip()]
    if not snapshot_lines:
        return ["- No candidate profile loaded yet. Provide resume text or use --resume."]
    return [f"- {line[2:]}" if line.startswith("- ") else f"- {line}" for line in snapshot_lines[:6]]


def _render_project_priorities(projects: list[dict[str, Any]]) -> list[str]:
    if not projects:
        return ["- No structured project cards detected yet."]

    lines: list[str] = []
    for index, project in enumerate(projects[:3], 1):
        name = str(project.get("name", "") or f"Project {index}").strip()
        ownership = str(project.get("ownership", "")).strip()
        summary = str(project.get("summary", "")).strip()
        deep_dive_points = list(project.get("deep_dive_points", []) or [])
        reason_bits: list[str] = []
        if ownership:
            reason_bits.append(f"ownership: {ownership}")
        if deep_dive_points:
            reason_bits.append(f"deep dive: {', '.join(deep_dive_points[:3])}")
        elif summary:
            reason_bits.append(summary)
        else:
            reason_bits.append("prepare a clean end-to-end story for this project")
        lines.append(f"- {name}: {' | '.join(reason_bits)}")
    return lines


def _render_question_loops(plan: list[PlannedQuestion]) -> list[str]:
    if not plan:
        return ["- No interview plan available yet."]

    counts = Counter(item.category_label for item in plan)
    return [f"- {label}: {count} planned question(s)" for label, count in counts.most_common(4)]


def _render_evidence_gaps(profile: str, projects: list[dict[str, Any]]) -> list[str]:
    lowered = str(profile or "").lower()
    lines: list[str] = []

    if not re.search(r"\b\d+(\.\d+)?(%|ms|s|qps|x)\b", lowered):
        lines.append("- Prepare 2-3 concrete metrics: latency, accuracy, throughput, adoption, or iteration speed.")

    if not any(str(project.get("ownership", "")).strip() for project in projects):
        lines.append("- Clarify ownership: what you designed, built, debugged, and decided yourself.")

    if not any(
        token in lowered
        for token in ("tradeoff", "fallback", "retry", "degrade", "evaluation", "deploy", "monitor", "latency")
    ):
        lines.append("- Prepare one trade-off story and one failure/fallback story for your main project.")

    if not any(
        token in lowered
        for token in ("retrieval", "rag", "memory", "tool", "agent", "workflow", "webworker", "rerank")
    ):
        lines.append("- Add architecture language around pipeline, tool use, memory, or async/runtime boundaries.")

    if not lines:
        lines.append("- Your profile already contains decent evidence density. Focus on sharper, shorter delivery.")
    return lines


def _render_training_plan(
    topic: str,
    company: str,
    position: str,
    target_description: str,
    profile: str,
    projects: list[dict[str, Any]],
    plan: list[PlannedQuestion],
) -> list[str]:
    lead_project = str(projects[0].get("name", "")).strip() if projects else "your strongest project"
    target_bits = ", ".join(bit for bit in (company, position, topic) if bit)
    target_text = target_bits or "your target role"
    missing_signals = _missing_target_signals(profile, topic, company, position, target_description)

    steps = [
        f"1. Rewrite your 90-second self-introduction for {target_text}.",
        f"2. Build a 3-minute deep-dive story for {lead_project}.",
        "3. Prepare concrete evidence for ownership, trade-offs, failure handling, and outcomes.",
        "4. Drill the seed questions below until each answer can be delivered in 1-2 minutes.",
    ]
    if missing_signals:
        steps.append(f"5. Add explicit stories for these missing role signals: {', '.join(missing_signals[:4])}.")
    if plan:
        steps.append(f"{len(steps) + 1}. Run a focused mock interview starting from the {plan[0].stage_label} stage.")
    return [f"- {step}" for step in steps]


def _render_seed_questions(plan: list[PlannedQuestion]) -> list[str]:
    if not plan:
        return ["- No seed questions available yet."]
    return [f"- {item.question}" for item in plan[:8]]


def _render_target_signals(topic: str, company: str, position: str, target_description: str) -> str:
    signals = list(_extract_signal_set(topic, company, position, target_description))
    return ", ".join(signals) if signals else "general interview prep"


def _render_target_gap_analysis(
    profile: str,
    topic: str,
    company: str,
    position: str,
    target_description: str,
) -> list[str]:
    target_signals = list(_extract_signal_set(topic, company, position, target_description))
    if not target_signals:
        return ["- No explicit target role signals provided yet. Add company, position, or a short JD summary."]

    aligned = [signal for signal in target_signals if _profile_matches_signal(profile, signal)]
    missing = [signal for signal in target_signals if signal not in aligned]

    lines = [f"- Target role is likely to emphasize: {', '.join(target_signals[:6])}."]
    if aligned:
        lines.append(f"- Already visible in your profile: {', '.join(aligned[:5])}.")
    if missing:
        lines.append(f"- Weak or missing in your profile today: {', '.join(missing[:5])}.")
        lines.append("- Use prep stories and mock interviews to make those signals explicit in your answers.")
    else:
        lines.append("- Your current profile already overlaps well with the target role. Focus on delivery and evidence density.")
    return lines


def _missing_target_signals(
    profile: str,
    topic: str,
    company: str,
    position: str,
    target_description: str,
) -> list[str]:
    return [
        signal
        for signal in _extract_signal_set(topic, company, position, target_description)
        if not _profile_matches_signal(profile, signal)
    ]


def _extract_signal_set(*parts: str) -> list[str]:
    signals: list[str] = []
    merged = " ".join(part for part in parts if part).lower()
    mapping = [
        ("agent", "agent"),
        ("rag", "rag"),
        ("retrieval", "retrieval"),
        ("memory", "memory"),
        ("frontend", "frontend"),
        ("webworker", "webworker"),
        ("python", "python"),
        ("backend", "backend"),
        ("eval", "evaluation"),
        ("trace", "tracing"),
        ("orchestrat", "orchestration"),
        ("tool", "tool-use"),
        ("workflow", "workflow"),
        ("system design", "system-design"),
        ("deploy", "deployment"),
        ("infra", "infrastructure"),
    ]
    for token, label in mapping:
        if token in merged and label not in signals:
            signals.append(label)
    return signals


def _profile_matches_signal(profile: str, signal: str) -> bool:
    lowered = str(profile or "").lower()
    signal_tokens = {
        "agent": ("agent", "multi-agent", "tool", "workflow"),
        "rag": ("rag", "retrieval", "rerank", "embedding"),
        "retrieval": ("retrieval", "rag", "rerank", "bm25", "embedding"),
        "memory": ("memory", "long-term", "short-term"),
        "frontend": ("frontend", "react", "webworker", "typescript"),
        "webworker": ("webworker", "worker", "wasm"),
        "python": ("python",),
        "backend": ("backend", "service", "api", "python"),
        "evaluation": ("evaluation", "eval", "metric", "trace"),
        "tracing": ("trace", "tracing", "observability"),
        "orchestration": ("orchestration", "workflow", "planner", "routing"),
        "tool-use": ("tool", "function call", "mcp"),
        "workflow": ("workflow", "state", "step", "phase"),
        "system-design": ("architecture", "system", "pipeline"),
        "deployment": ("deploy", "latency", "monitor"),
        "infrastructure": ("infra", "deployment", "monitor", "service"),
    }
    return any(token in lowered for token in signal_tokens.get(signal, (signal,)))


__all__ = ["render_interview_prep_pack"]
