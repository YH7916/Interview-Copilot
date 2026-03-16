"""
WeaknessTracker — 面试专属错题本 (长期记忆层)

职责：
  1. update(eval_results)  — 每次 eval 后，让 Qwen 从打分数据中提取薄弱知识点，
                             以带日期的 Markdown 条目追加到 weakness_log.md
  2. get_context()         — 读取 weakness_log.md，返回注入 System Prompt 的字符串

存储路径: data/memory/weakness_log.md (项目根目录下)
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
WEAKNESS_LOG_PATH = BASE_DIR / "data" / "memory" / "weakness_log.md"

# ---------------------------------------------------------------
# Review Prompt: 让 Qwen 从打分结果里提炼薄弱点
# ---------------------------------------------------------------
_REVIEW_PROMPT = """
你是一个面试表现分析师。根据候选人这次模拟面试的评测数据，提炼出他的核心薄弱点。

【评测数据（JSON）】
{eval_json}

【输出要求】
仅输出 2-4 条 Markdown 列表项，每项格式：
- ❌ 或 ⚠️ **知识点名称**: 一句话说明具体缺失或错误

规则：
- accuracy_score ≤ 3 用 ❌，4 用 ⚠️，5 不出现在列表中
- 不要任何额外解释，只输出列表项
"""


class WeaknessTracker:
    """面试薄弱点追踪器 — 将每次模拟面试的弱项持久化，供下次 System Prompt 注入。"""

    def __init__(self, log_path: Path | None = None):
        self.log_path = log_path or WEAKNESS_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def update(self, eval_results: list[dict]) -> None:
        """
        eval_results 示例：
        [
          {"question": "...", "expected": "...", "actual": "...",
           "accuracy_score": 3, "faithfulness_score": 5, "reason": "..."},
          ...
        ]
        """
        weak_items = self._extract_weaknesses(eval_results)
        avg_acc = sum(r.get("accuracy_score", 0) for r in eval_results) / max(len(eval_results), 1)
        entry = self._format_entry(avg_acc, weak_items)
        self._append(entry)

    def get_context(self) -> str:
        """返回用于注入 System Prompt 的薄弱点摘要，若无记录则返回空串。"""
        if not self.log_path.exists():
            return ""
        content = self.log_path.read_text(encoding="utf-8").strip()
        if not content:
            return ""
        return (
            "## 📋 候选人历史薄弱点（错题本）\n\n"
            "以下是该候选人在历次模拟面试中暴露的知识盲区，请在回答时特别关注并主动引导：\n\n"
            + content
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _extract_weaknesses(self, eval_results: list[dict]) -> str:
        """调用 Qwen 从打分数据中提炼薄弱点列表（markdown bullet points）。"""
        # 过滤掉满分项，减少无效 API 调用
        weak = [r for r in eval_results if r.get("accuracy_score", 5) < 5]
        if not weak:
            return "- ✅ 本次评测全部满分，无明显薄弱点"

        try:
            import dashscope

            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                config = json.loads(
                    (Path.home() / ".nanobot" / "config.json").read_text(encoding="utf-8")
                )
                api_key = config.get("providers", {}).get("dashscope", {}).get("apiKey")
            dashscope.api_key = api_key

            prompt = _REVIEW_PROMPT.format(eval_json=json.dumps(weak, ensure_ascii=False, indent=2))
            resp = dashscope.Generation.call(model="qwen-max", prompt=prompt, result_format="text")
            return resp.output.text.strip()
        except Exception as e:
            # 降级：直接从 reason 字段拼接
            lines = []
            for r in weak:
                score = r.get("accuracy_score", "?")
                icon = "❌" if score <= 3 else "⚠️"
                q = r.get("question", "未知问题")[:30]
                reason = r.get("reason", "")
                lines.append(f"- {icon} **{q}**: {reason}")
            return "\n".join(lines) if lines else f"- ⚠️ 薄弱点提取失败: {e}"

    @staticmethod
    def _format_entry(avg_acc: float, weak_items: str) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        count_emoji = "🔴" if avg_acc < 3 else "🟡" if avg_acc < 4 else "🟢"
        return (
            f"## [{ts}] 评测记录 {count_emoji} (avg accuracy: {avg_acc:.1f}/5)\n\n"
            f"{weak_items}\n"
        )

    def _append(self, entry: str) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
