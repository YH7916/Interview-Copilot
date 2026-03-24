"""Helpers for loading a candidate profile from text or a local source file."""

from __future__ import annotations

import re
from pathlib import Path

from copilot.profile.extractor import build_candidate_profile_summary


def load_candidate_profile(
    *,
    profile_text: str = "",
    profile_path: str = "",
    max_chars: int = 4000,
) -> str:
    text = str(profile_text or "").strip()
    if text:
        return normalize_candidate_profile(text, max_chars=max_chars)

    path_value = str(profile_path or "").strip()
    if not path_value:
        return ""

    path = Path(path_value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Candidate profile file not found: {path}")
    if path.suffix.lower() not in {".txt", ".md", ".typ"}:
        raise ValueError(
            f"Unsupported candidate profile file type: {path.suffix or '<none>'}. "
            "Use .txt, .md, or .typ."
        )

    return normalize_candidate_profile(
        path.read_text(encoding="utf-8", errors="ignore"),
        max_chars=max_chars,
    )


def normalize_candidate_profile(text: str, *, max_chars: int = 4000) -> str:
    lines = []
    for raw_line in str(text or "").replace("\r", "\n").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            lines.append(line)

    normalized = "\n".join(lines)
    if not normalized:
        return ""

    extracted = build_candidate_profile_summary(normalized, max_chars=max_chars)
    if extracted:
        return extracted
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 4].rstrip() + "\n..."
