"""Interview modes with distinct behaviour policies."""

from __future__ import annotations

from enum import StrEnum


class InterviewMode(StrEnum):
    FORMAL = "formal"
    COACH = "coach"
    REVIEW = "review"
