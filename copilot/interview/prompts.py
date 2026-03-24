"""Prompt helpers for different interview modes."""

from __future__ import annotations

from copilot.interview.modes import InterviewMode


def build_interview_system_prompt(mode: InterviewMode) -> str:
    if mode == InterviewMode.COACH:
        return (
            "You are an interview coach. Ask targeted questions, expose weak spots, "
            "and give concise actionable feedback after each answer."
        )
    if mode == InterviewMode.REVIEW:
        return (
            "You are an interview reviewer. Summarize strengths, weaknesses, and next steps "
            "based on the completed interview."
        )
    return (
        "You are a realistic interviewer for AI and agent roles. Ask one question at a time, "
        "keep the flow natural, and avoid revealing hidden evaluation criteria."
    )
