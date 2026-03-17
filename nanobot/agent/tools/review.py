"""TriggerReviewTool — 让面试官 bot 在对话结束时主动触发 Review Agent。"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from nanobot.agent.tools.base import Tool

logger = logging.getLogger(__name__)


class TriggerReviewTool(Tool):
    """
    面试官 bot 在单轮模拟面试结束后调用此 tool，
    后台异步分析对话历史并更新错题本。
    """

    def __init__(self) -> None:
        self._history: list[dict] = []

    @property
    def name(self) -> str:
        return "trigger_interview_review"

    @property
    def description(self) -> str:
        return (
            "在单轮模拟面试结束时调用。后台自动分析本次对话，"
            "提取候选人薄弱点并更新错题本（weakness_log.md）。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "session_summary": {
                    "type": "string",
                    "description": "本次面试的一句话简短总结，如'候选人完成了 RAG 和并发编程两个专题的模拟面试'",
                }
            },
            "required": ["session_summary"],
        }

    def set_history(self, messages: list[dict]) -> None:
        """由 AgentLoop 在每轮结束时注入当前对话历史。"""
        self._history = messages

    async def execute(self, session_summary: str, **kwargs: Any) -> str:
        try:
            from copilot.memory.review_agent import ReviewAgent
            agent = ReviewAgent()
            # 启动后台任务，不阻塞当前对话
            asyncio.create_task(agent._analyze_async(list(self._history)))
            logger.info("ReviewAgent triggered: %s", session_summary)
            return f"✅ 已触发面试复盘分析：{session_summary}\n错题本将在后台自动更新。"
        except Exception as e:
            logger.warning("TriggerReviewTool failed: %s", e)
            return f"⚠️ 复盘触发失败（{e}），请手动运行 `python evals/run_eval.py`。"
