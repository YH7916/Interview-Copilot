"""复盘 Agent — 面试结束后异步分析对话，自动更新错题本。"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from copilot.memory.weakness_tracker import WeaknessTracker, _call_llm

logger = logging.getLogger(__name__)

_REVIEW_PROMPT = """\
你是面试复盘分析师。阅读以下模拟面试对话，提炼候选人暴露的知识薄弱点。

【对话记录】
{dialogue}

【输出规则】
- 表现良好则仅输出：- ✅ 本次面试表现良好，无明显薄弱点
- 否则输出 2-4 条列表项：`- ❌ **知识点**: 一句话说明缺失`（基本正确用⚠️，明显错误用❌）
- 不输出额外内容"""

_WEAK_KEYWORDS = ("不知道", "不清楚", "不太确定", "不对", "错了", "抱歉", "理解有误")


class ReviewAgent:
    """非阻塞复盘：在后台分析对话 → 提炼薄弱点 → 写入错题本。"""

    def __init__(self, tracker: WeaknessTracker | None = None):
        self.tracker = tracker or WeaknessTracker()

    def analyze_background(self, messages: list[dict]) -> None:
        """Fire-and-forget：不阻塞用户交互。"""
        asyncio.create_task(self._analyze_async(messages))

    async def analyze(self, messages: list[dict]) -> str:
        """阻塞版本，返回提炼结果。用于测试或手动调用。"""
        return await asyncio.to_thread(self._analyze_sync, messages)

    # ── 内部 ────────────────────────────────────────────────────

    async def _analyze_async(self, messages: list[dict]) -> None:
        try:
            await asyncio.to_thread(self._analyze_sync, messages)
            logger.info("ReviewAgent: 错题本已更新")
        except Exception:
            logger.exception("ReviewAgent: 分析失败")

    def _analyze_sync(self, messages: list[dict]) -> str:
        dialogue = _format_dialogue(messages)

        try:
            bullets = _call_llm(_REVIEW_PROMPT.format(dialogue=dialogue))
        except Exception as e:
            bullets = _fallback_extract(messages, e)

        # 根据内容推断严重程度（而非硬编码 avg=0.0）
        icon = _infer_severity(bullets)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"## [{ts}] 🤖 面试复盘 {icon}\n\n{bullets}\n"

        self.tracker._append(entry)
        return bullets


# ── 工具函数 ────────────────────────────────────────────────────


def _format_dialogue(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = "面试官" if m.get("role") == "assistant" else "候选人"
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        if content:
            lines.append(f"[{role}]: {content[:300]}")
    return "\n".join(lines)


def _fallback_extract(messages: list[dict], err: Exception) -> str:
    """LLM 不可用时，基于关键词做降级提取。"""
    hints = [
        f"- ⚠️ **{m.get('content', '?')[:40]}**: 候选人表达了不确定"
        for m in messages
        if m.get("role") == "user" and any(kw in (m.get("content") or "") for kw in _WEAK_KEYWORDS)
    ]
    return "\n".join(hints[:4]) or f"- ⚠️ 自动分析失败({err})，请人工复盘"


def _infer_severity(bullets: str) -> str:
    """从薄弱点标记推断严重程度图标。"""
    if "✅" in bullets and "❌" not in bullets and "⚠️" not in bullets:
        return "🟢"
    if bullets.count("❌") >= 2:
        return "🔴"
    return "🟡"
