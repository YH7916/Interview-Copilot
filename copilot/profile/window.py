"""Short-term conversation window limited by turns and tokens."""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


class Turn(NamedTuple):
    user: str
    assistant: str


class TokenWindowManager:
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
        messages = []
        for turn in self._window:
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": turn.assistant})
        return messages

    @property
    def turn_count(self) -> int:
        return len(self._window)

    def clear(self) -> None:
        self._window.clear()

    def _count_tokens(self) -> int:
        return sum(len(_enc.encode(turn.user)) + len(_enc.encode(turn.assistant)) for turn in self._window)
