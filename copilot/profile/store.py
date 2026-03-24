"""Persistent store for recurring interview weak points."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from copilot.llm import call_text

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "data" / "memory" / "weakness_log.md"

EXTRACT_PROMPT = """Extract 2-4 concise recurring weak points from the evaluation results.
Return markdown bullet points. Each bullet should name the weakness and give a brief reason.

Evaluation results:
{eval_json}
"""


class WeaknessTracker:
    _lock = threading.Lock()

    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def update(self, eval_results: list[dict]) -> str:
        bullets = self._extract(eval_results)
        average = sum(item.get("accuracy_score", 0) for item in eval_results) / max(len(eval_results), 1)
        self._append(_format_entry(average, bullets))
        return bullets

    def get_context(self) -> str:
        if not self.log_path.exists():
            return ""
        text = self.log_path.read_text(encoding="utf-8").strip()
        if not text:
            return ""
        return "## Candidate profile notes\n\n" + text

    def _extract(self, eval_results: list[dict]) -> str:
        weak_items = [item for item in eval_results if item.get("accuracy_score", 5) < 5]
        if not weak_items:
            return "- No persistent weak points detected in this round."

        try:
            prompt = EXTRACT_PROMPT.format(eval_json=json.dumps(weak_items, ensure_ascii=False, indent=2))
            return call_text(prompt, task="analysis")
        except Exception:
            return "\n".join(
                f"- {_score_icon(item.get('accuracy_score', 0))} "
                f"**{item.get('question', '?')[:30]}**: {item.get('reason', '')}"
                for item in weak_items
            )

    def _append(self, entry: str) -> None:
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(entry + "\n")


def _format_entry(average: float, bullets: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"## [{timestamp}] Profile update {_average_icon(average)} (avg accuracy: {average:.1f}/5)\n\n{bullets}\n"


def _score_icon(score: float) -> str:
    return "warning" if score <= 3 else "note"


def _average_icon(score: float) -> str:
    if score < 3:
        return "high"
    if score < 4:
        return "medium"
    return "low"
