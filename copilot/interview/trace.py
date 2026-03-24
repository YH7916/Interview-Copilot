"""Structured trace for live interview sessions."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = ROOT / "data" / "traces"


@dataclass(slots=True)
class TraceTurn:
    index: int
    stage: str
    category: str
    question: str
    answer: str
    policy_action: str
    policy_reason: str
    follow_up: str = ""
    follow_up_answer: str = ""
    review: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InterviewTrace:
    session_id: str
    user_id: str
    topic: str
    started_at: str
    completed_at: str = ""
    turns: list[TraceTurn] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def add_turn(self, turn: TraceTurn) -> None:
        self.turns.append(turn)

    def finalize(self, summary: dict[str, Any]) -> None:
        self.completed_at = datetime.now().isoformat(timespec="seconds")
        self.summary = summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "topic": self.topic,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "turns": [asdict(item) for item in self.turns],
            "summary": self.summary,
        }


@dataclass(slots=True)
class PrepTrace:
    session_id: str
    user_id: str
    topic: str
    started_at: str
    completed_at: str = ""
    target: dict[str, Any] = field(default_factory=dict)
    profile_summary: list[str] = field(default_factory=list)
    project_priorities: list[str] = field(default_factory=list)
    target_gap_analysis: list[str] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)
    training_plan: list[str] = field(default_factory=list)
    seed_questions: list[str] = field(default_factory=list)

    def finalize(self) -> None:
        self.completed_at = datetime.now().isoformat(timespec="seconds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_interview_trace(*, user_id: str, topic: str) -> InterviewTrace:
    started_at = datetime.now().isoformat(timespec="seconds")
    session_id = f"{_slug(user_id)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return InterviewTrace(
        session_id=session_id,
        user_id=user_id,
        topic=topic,
        started_at=started_at,
    )


def create_prep_trace(*, user_id: str, topic: str) -> PrepTrace:
    started_at = datetime.now().isoformat(timespec="seconds")
    session_id = f"prep-{_slug(user_id)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return PrepTrace(
        session_id=session_id,
        user_id=user_id,
        topic=topic,
        started_at=started_at,
    )


def save_interview_trace(trace: InterviewTrace, *, trace_dir: Path = TRACE_DIR) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{trace.session_id}.json"
    path.write_text(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_prep_trace(trace: PrepTrace, *, trace_dir: Path = TRACE_DIR) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{trace.session_id}.json"
    path.write_text(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _slug(value: str) -> str:
    lowered = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
    return lowered.strip("-") or "interview"


__all__ = [
    "InterviewTrace",
    "PrepTrace",
    "TraceTurn",
    "TRACE_DIR",
    "create_interview_trace",
    "create_prep_trace",
    "save_interview_trace",
    "save_prep_trace",
]
