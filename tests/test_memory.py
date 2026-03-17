"""单元测试：copilot 记忆层 (WeaknessTracker, TokenWindowManager, ReviewAgent)"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# WeaknessTracker
# ──────────────────────────────────────────────────────────────────────────────

class TestWeaknessTracker:
    def _tracker(self, tmp_path: Path):
        from copilot.memory.weakness_tracker import WeaknessTracker
        return WeaknessTracker(log_path=tmp_path / "weakness_log.md")

    def test_get_context_empty(self, tmp_path):
        t = self._tracker(tmp_path)
        assert t.get_context() == ""

    def test_get_context_returns_text(self, tmp_path):
        t = self._tracker(tmp_path)
        t.log_path.write_text("- ❌ **RAG**: 未能说明检索流程", encoding="utf-8")
        ctx = t.get_context()
        assert "错题本" in ctx
        assert "RAG" in ctx

    def test_update_writes_entry(self, tmp_path):
        """update() 必须写入 markdown 条目（mock LLM 调用）。"""
        t = self._tracker(tmp_path)
        eval_results = [
            {"question": "什么是 RAG?", "accuracy_score": 2, "reason": "未说明检索流程"},
        ]
        with patch("copilot.memory.weakness_tracker._call_llm", return_value="- ❌ **RAG**: 未知"):
            t.update(eval_results)

        content = t.log_path.read_text(encoding="utf-8")
        assert "评测记录" in content
        assert "RAG" in content

    def test_update_all_perfect_score(self, tmp_path):
        """全部满分时不调用 LLM，仅写入满分提示。"""
        t = self._tracker(tmp_path)
        eval_results = [{"question": "Q1", "accuracy_score": 5, "reason": "perfect"}]
        with patch("copilot.memory.weakness_tracker._call_llm") as mock_llm:
            t.update(eval_results)
            mock_llm.assert_not_called()

        assert "满分" in t.log_path.read_text(encoding="utf-8")

    def test_update_fallback_on_api_error(self, tmp_path):
        """LLM 调用失败时，降级为从 reason 字段拼接。"""
        t = self._tracker(tmp_path)
        eval_results = [{"question": "并发原理", "accuracy_score": 3, "reason": "答错了"}]
        with patch("copilot.memory.weakness_tracker._call_llm", side_effect=RuntimeError("timeout")):
            t.update(eval_results)

        content = t.log_path.read_text(encoding="utf-8")
        assert "并发原理" in content


# ──────────────────────────────────────────────────────────────────────────────
# TokenWindowManager
# ──────────────────────────────────────────────────────────────────────────────

class TestTokenWindowManager:
    def _mgr(self, **kw):
        from copilot.memory.short_term import TokenWindowManager
        return TokenWindowManager(**kw)

    def test_add_and_get(self):
        mgr = self._mgr(max_turns=3)
        mgr.add_turn("你好", "你好！")
        msgs = mgr.get_window()
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "你好"}
        assert msgs[1] == {"role": "assistant", "content": "你好！"}

    def test_max_turns_eviction(self):
        mgr = self._mgr(max_turns=2)
        mgr.add_turn("Q1", "A1")
        mgr.add_turn("Q2", "A2")
        mgr.add_turn("Q3", "A3")  # 应弹出 Q1/A1
        assert mgr.turn_count == 2
        window = mgr.get_window()
        assert any("Q3" in m["content"] for m in window)
        assert not any("Q1" in m["content"] for m in window)

    def test_token_eviction(self):
        mgr = self._mgr(max_turns=10, max_tokens=10)  # 极小 token 限制
        mgr.add_turn("A" * 50, "B" * 50)  # ~25 tokens, 超出限制
        mgr.add_turn("C" * 50, "D" * 50)
        # 窗口里只保留不超限的轮次
        assert mgr.turn_count <= 1

    def test_clear(self):
        mgr = self._mgr()
        mgr.add_turn("Q", "A")
        mgr.clear()
        assert mgr.turn_count == 0
        assert mgr.get_window() == []


# ──────────────────────────────────────────────────────────────────────────────
# ReviewAgent
# ──────────────────────────────────────────────────────────────────────────────

class TestReviewAgent:
    def _agent(self, tmp_path: Path):
        from copilot.memory.weakness_tracker import WeaknessTracker
        from copilot.memory.review_agent import ReviewAgent
        tracker = WeaknessTracker(log_path=tmp_path / "weakness_log.md")
        return ReviewAgent(tracker=tracker), tracker

    def test_analyze_writes_to_tracker(self, tmp_path):
        agent, tracker = self._agent(tmp_path)
        messages = [
            {"role": "assistant", "content": "请解释什么是 GIL？"},
            {"role": "user", "content": "不知道，GIL 是什么？"},
        ]
        with patch("copilot.memory.weakness_tracker._call_llm", return_value="- ❌ **GIL**: 不了解"):
            asyncio.run(agent.analyze(messages))

        content = tracker.log_path.read_text(encoding="utf-8")
        assert "ReviewAgent" in content or "面试复盘" in content

    def test_analyze_fallback_on_api_error(self, tmp_path):
        """API 失败时，依据关键词做降级提取，不抛异常。"""
        agent, tracker = self._agent(tmp_path)
        messages = [
            {"role": "user", "content": "不知道，我不清楚这个问题"},
        ]
        with patch("copilot.memory.weakness_tracker._call_llm", side_effect=RuntimeError("net err")):
            asyncio.run(agent.analyze(messages))

        assert tracker.log_path.exists()
