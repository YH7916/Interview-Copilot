"""短期记忆 — Token 滑动窗口，保持上下文在模型限制内。"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


class Turn(NamedTuple):
    user: str
    assistant: str


class TokenWindowManager:
    """保留最近 max_turns 轮对话，且总 Token 不超过 max_tokens。

    超限时自动弹出最旧的轮次（FIFO）。
    """

    def __init__(self, max_turns: int = 5, max_tokens: int = 4_000):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._window: deque[Turn] = deque()

    def add_turn(self, user: str, assistant: str) -> None:
        self._window.append(Turn(user, assistant))
        while len(self._window) > self.max_turns:
            self._window.popleft()
        while self._window and self._count_tokens() > self.max_tokens:
            self._window.popleft()

    def get_window(self) -> list[dict]:
        """转为 OpenAI messages 格式。"""
        msgs = []
        for t in self._window:
            msgs.append({"role": "user", "content": t.user})
            msgs.append({"role": "assistant", "content": t.assistant})
        return msgs

    @property
    def turn_count(self) -> int:
        return len(self._window)

    def clear(self) -> None:
        self._window.clear()

    def _count_tokens(self) -> int:
        """tiktoken cl100k_base 精确计数。"""
        return sum(len(_enc.encode(t.user)) + len(_enc.encode(t.assistant)) for t in self._window)
