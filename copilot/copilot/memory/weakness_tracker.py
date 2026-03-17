"""错题本 — 长期记忆层，持久化面试薄弱点并注入 System Prompt。"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from copilot.config import get_dashscope_api_key

_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = _ROOT / "data" / "memory" / "weakness_log.md"

_EXTRACT_PROMPT = """\
你是面试表现分析师。根据评测数据，提炼候选人的核心薄弱点。

【评测数据（JSON）】
{eval_json}

【输出规则】
- 仅输出 2-4 条 Markdown 列表项
- 格式：`- ❌ **知识点**: 一句话说明缺失` (accuracy≤3用❌, =4用⚠️, 5不出现)
- 不输出任何额外内容"""


class WeaknessTracker:
    """每次评测后提炼薄弱点 → 写入 Markdown → 下次 Session 注入 AI 上下文。"""

    _lock = threading.Lock()  # 保护并发写入（多个 ReviewAgent 同时分析时）

    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def update(self, eval_results: list[dict]) -> None:
        """提炼薄弱点并追加到错题本。"""
        bullets = self._extract(eval_results)
        avg = sum(r.get("accuracy_score", 0) for r in eval_results) / max(len(eval_results), 1)
        self._append(_format_entry(avg, bullets))

    def get_context(self) -> str:
        """读取错题本，返回可直接注入 System Prompt 的文本。空则返回 ''。"""
        if not self.log_path.exists():
            return ""
        text = self.log_path.read_text(encoding="utf-8").strip()
        if not text:
            return ""
        return (
            "## 📋 候选人历史薄弱点（错题本）\n\n"
            "以下为历次面试暴露的知识盲区，请主动引导：\n\n" + text
        )

    # ── 内部 ────────────────────────────────────────────────────

    def _extract(self, eval_results: list[dict]) -> str:
        weak = [r for r in eval_results if r.get("accuracy_score", 5) < 5]
        if not weak:
            return "- ✅ 本次全部满分，无明显薄弱点"
        try:
            return _call_llm(
                _EXTRACT_PROMPT.format(eval_json=json.dumps(weak, ensure_ascii=False, indent=2))
            )
        except Exception:
            # 降级：直接拼接 reason
            return "\n".join(
                f"- {'❌' if r.get('accuracy_score', 0) <= 3 else '⚠️'} "
                f"**{r.get('question', '?')[:30]}**: {r.get('reason', '')}"
                for r in weak
            )

    def _append(self, entry: str) -> None:
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(entry + "\n")


# ── 工具函数（供 ReviewAgent 等模块复用）────────────────────────


def _format_entry(avg: float, bullets: str) -> str:
    icon = "🔴" if avg < 3 else "🟡" if avg < 4 else "🟢"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"## [{ts}] 评测记录 {icon} (avg accuracy: {avg:.1f}/5)\n\n{bullets}\n"


def _call_llm(prompt: str) -> str:
    """同步调用 DashScope Qwen。"""
    import dashscope

    dashscope.api_key = get_dashscope_api_key()
    resp = dashscope.Generation.call(model="qwen-max", prompt=prompt, result_format="text")
    return resp.output.text.strip()
