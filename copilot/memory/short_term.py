"""
TokenWindowManager — 短期记忆（滑动窗口）

维护最近 N 轮对话，超过 token 阈值时弹出最旧一轮。
面向 copilot 业务层；nanobot 系统级滑动窗口由 MemoryConsolidator 负责。
"""
from __future__ import annotations

from collections import deque
from typing import NamedTuple

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


class Turn(NamedTuple):
    user: str
    assistant: str


class TokenWindowManager:
    """
    保留最近 max_turns 轮、且总 Token 数不超过 max_tokens 的对话窗口。
    
    设计意图：
    AI 记忆不是越多越好，太久远的消息会消耗多余 Token 且干扰判断。
    这个类作为独立的辅助工具，可以放在任何需要“滑动窗口”记忆的地方（如业务插件）。
    """

    def __init__(self, max_turns: int = 5, max_tokens: int = 4_000):
        self.max_turns = max_turns   # 窗口最大轮数（一问一答为一轮）
        self.max_tokens = max_tokens # 窗口最大 Token 容纳量（粗略估算）
        # 使用 deque (双端队列) 方便高效地从左侧（旧消息）进行弹出
        self._window: deque[Turn] = deque()

    # ── 公开接口 ────────────────────────────────────────────────

    def add_turn(self, user: str, assistant: str) -> None:
        """
        添加一对【问、答】进入窗口。
        如果总数超标，先剔除最老的轮次；如果 Token 超标，继续剔除。
        """
        self._window.append(Turn(user, assistant))
        
        # 约束 1：限制最大对话轮数，不让历史堆积太长
        while len(self._window) > self.max_turns:
            self._window.popleft()
            
        # 约束 2：限制 Token 长度，防止撑爆模型上下文导致 400 错误
        while self._window and self._count_tokens() > self.max_tokens:
            self._window.popleft()

    def get_window(self) -> list[dict]:
        """
        将对象内部的 Turn 集合转换成 OpenAI 兼容的消息格式。
        供 llm.chat(messages=...) 直接使用。
        """
        msgs = []
        for turn in self._window:
            msgs.append({"role": "user", "content": turn.user})
            msgs.append({"role": "assistant", "content": turn.assistant})
        return msgs

    @property
    def turn_count(self) -> int:
        """返回目前窗口里有几轮对话。"""
        return len(self._window)

    def clear(self) -> None:
        """清空所有短期记忆。"""
        self._window.clear()

    # ── 内部方法 ────────────────────────────────────────────────

    def _count_tokens(self) -> int:
        """
        使用 tiktoken cl100k_base 统计 Token 数量。
        """
        return sum(len(_enc.encode(t.user)) + len(_enc.encode(t.assistant)) for t in self._window)
