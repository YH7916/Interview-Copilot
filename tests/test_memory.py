from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

def _make_log_path() -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="copilot-test-", suffix=".md", delete=False)
    handle.close()
    return Path(handle.name)


class TestWeaknessTracker:
    def _tracker(self):
        from copilot.profile.store import WeaknessTracker

        return WeaknessTracker(log_path=_make_log_path())

    def test_get_context_empty(self):
        tracker = self._tracker()
        tracker.log_path.unlink(missing_ok=True)
        assert tracker.get_context() == ""

    def test_get_context_returns_text(self):
        tracker = self._tracker()
        tracker.log_path.write_text("- Weak on RAG retrieval details", encoding="utf-8")

        context = tracker.get_context()
        assert "Candidate profile notes" in context
        assert "RAG" in context

    def test_update_writes_entry(self):
        tracker = self._tracker()
        eval_results = [{"question": "What is RAG?", "accuracy_score": 2, "reason": "Missing retrieval flow."}]

        with patch("copilot.profile.store.call_text", return_value="- Weak on RAG retrieval flow"):
            tracker.update(eval_results)

        content = tracker.log_path.read_text(encoding="utf-8")
        assert "Profile update" in content
        assert "RAG" in content

    def test_update_fallback_on_api_error(self):
        tracker = self._tracker()
        eval_results = [{"question": "Explain concurrency", "accuracy_score": 3, "reason": "Too shallow."}]

        with patch("copilot.profile.store.call_text", side_effect=RuntimeError("timeout")):
            tracker.update(eval_results)

        content = tracker.log_path.read_text(encoding="utf-8")
        assert "Explain concurrency" in content


class TestTokenWindowManager:
    def _manager(self, **kwargs):
        from copilot.profile.window import TokenWindowManager

        return TokenWindowManager(**kwargs)

    def test_add_and_get(self):
        manager = self._manager(max_turns=3)
        manager.add_turn("hello", "hi")

        messages = manager.get_window()
        assert messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

    def test_max_turns_eviction(self):
        manager = self._manager(max_turns=2)
        manager.add_turn("Q1", "A1")
        manager.add_turn("Q2", "A2")
        manager.add_turn("Q3", "A3")

        contents = [item["content"] for item in manager.get_window()]
        assert "Q1" not in contents
        assert "Q3" in contents
        assert manager.turn_count == 2

    def test_token_eviction(self):
        manager = self._manager(max_turns=10, max_tokens=10)
        manager.add_turn("A" * 50, "B" * 50)
        manager.add_turn("C" * 50, "D" * 50)

        assert manager.turn_count <= 1

    def test_clear(self):
        manager = self._manager()
        manager.add_turn("Q", "A")
        manager.clear()

        assert manager.turn_count == 0
        assert manager.get_window() == []


class TestReviewAgent:
    def _agent(self):
        from copilot.profile.review import ReviewAgent
        from copilot.profile.store import WeaknessTracker

        tracker = WeaknessTracker(log_path=_make_log_path())
        return ReviewAgent(tracker=tracker), tracker

    def test_analyze_writes_to_tracker(self):
        agent, tracker = self._agent()
        messages = [
            {"role": "assistant", "content": "Please explain GIL."},
            {"role": "user", "content": "I do not really understand it."},
        ]

        with patch("copilot.interview.orchestrator.InterviewRunner.review_messages", return_value={"count": 1, "average_overall": 3.0, "results": [{"question": "Please explain GIL.", "accuracy_score": 3, "reason": "Too shallow", "overall_score": 3.0}], "summary": "# Interview Review\n\n1. Please explain GIL.\n   - overall: 3.0/5\n   - accuracy: 3/5\n   - reason: Too shallow"}):
            summary = asyncio.run(agent.analyze(messages))

        assert "Interview Review" in summary

    def test_analyze_fallback_on_api_error(self):
        agent, tracker = self._agent()
        messages = [{"role": "user", "content": "这个问题我不知道怎么答。"}]

        with patch("copilot.profile.review.call_text", side_effect=RuntimeError("net err")):
            asyncio.run(agent.analyze(messages))

        content = tracker.log_path.read_text(encoding="utf-8")
        assert "Interview review" in content
