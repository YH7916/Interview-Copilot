"""Interview session state shared by different frontends and channels."""

from __future__ import annotations

from dataclasses import dataclass, field

from copilot.interview.modes import InterviewMode


@dataclass(slots=True)
class InterviewSession:
    user_id: str
    mode: InterviewMode = InterviewMode.FORMAL
    round_index: int = 1
    company: str = ""
    position: str = ""
    focus_topics: list[str] = field(default_factory=list)
    candidate_profile: str = ""
    interview_style: str = "auto"
    turns: int = 0

    def next_turn(self) -> None:
        self.turns += 1
