"""
WeaknessTracker — 面试错题本（长期记忆层）

职责：
  1. update(eval_results) — 负责把每次面试或评测的原始扣分项，转化为 Markdown 日志追加到文件中。
  2. get_context()       — 负责读取日志，提供一个“摘要”给 System Prompt，让 AI 知道历史弱点。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

# 获取项目根目录 (Interview-Copilot/)
_BASE = Path(__file__).resolve().parents[2]
# 定义错题本存储路径：data/memory/weakness_log.md
LOG_PATH = _BASE / "data" / "memory" / "weakness_log.md"

# 提炼薄弱点的 Prompt 模板
_PROMPT = """\
你是面试表现分析师。根据评测数据，提炼候选人的核心薄弱点。

【评测数据（JSON）】
{eval_json}

【输出规则】
- 仅输出 2-4 条 Markdown 列表项
- 格式：`- ❌ **知识点**: 一句话说明缺失` (accuracy≤3用❌, =4用⚠️, 5不出现)
- 不输出任何额外内容"""


class WeaknessTracker:
    """持久化面试薄弱点，每次 Session 启动时通过 System Prompt 自动注入 AI 的脑子。"""

    def __init__(self, log_path: Path = LOG_PATH):
        # 初始化存储路径，并确保其父目录 (data/memory/) 存在
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 公开接口：供外部调用 ───────────────────────────────────────────

    def update(self, eval_results: list[dict]) -> None:
        """
        核心方法：传入评测结果列表，自动提炼并记录薄弱点。
        eval_results 格式预期包含 accuracy_score, question, reason 等字段。
        """
        # 1. 调用 LLM 提炼 Markdown 格式的薄弱点列表
        bullets = self._extract(eval_results)
        # 2. 计算本次面试的平均准确率，用于显示红/黄/绿灯
        avg = sum(r.get("accuracy_score", 0) for r in eval_results) / max(len(eval_results), 1)
        # 3. 按照带时间戳的格式，将内容追加到文件末尾
        self._append(_format_entry(avg, bullets))

    def get_context(self) -> str:
        """
        读取错题本文件，封装成用于注入 System Prompt 的 context 字符串。
        如果文件不存在或为空，返回空字符串，不影响 AI 正常生成。
        """
        if not self.log_path.exists():
            return ""
        text = self.log_path.read_text(encoding="utf-8").strip()
        if not text:
            return ""
        # 封装成 Markdown 章节，方便 AI 理解这是历史背景
        return (
            "## 📋 候选人历史薄弱点（错题本）\n\n"
            "以下为历次模拟面试中暴露的知识盲区，请在回答时特别关注并主动引导：\n\n"
            + text
        )

    # ── 内部逻辑方法 (Private-like) ──────────────────────────────────────

    def _extract(self, eval_results: list[dict]) -> str:
        """调用 Qwen 模型，从海量的 JSON 评测数据中精炼出几条核心弱点。"""
        # 过滤：只关心不到 5 分（满分）的项目
        weak = [r for r in eval_results if r.get("accuracy_score", 5) < 5]
        if not weak:
            return "- ✅ 本次全部满分，无明显薄弱点"

        try:
            # 正常路径：请求 AI 进行提炼
            return _call_llm(_PROMPT.format(eval_json=json.dumps(weak, ensure_ascii=False, indent=2)))
        except Exception as e:
            # 降级路径：API 挂了时，不报错，直接拼接原始的 reason 字段
            return "\n".join(
                f"- {'❌' if r.get('accuracy_score', 0) <= 3 else '⚠️'} "
                f"**{r.get('question','?')[:30]}**: {r.get('reason','')}"
                for r in weak
            ) or f"- ⚠️ 薄弱点提取失败: {e}"

    def _append(self, entry: str) -> None:
        """简单、稳健的文件追加方法。"""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")


# ── 工具函数 ──────────────────────────────────────────────────

def _format_entry(avg: float, bullets: str) -> str:
    icon = "🔴" if avg < 3 else "🟡" if avg < 4 else "🟢"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"## [{ts}] 评测记录 {icon} (avg accuracy: {avg:.1f}/5)\n\n{bullets}\n"


def _call_llm(prompt: str) -> str:
    """调用 DashScope Qwen，从 config 或环境变量读取 API Key。"""
    import dashscope

    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY") or _read_config_key()
    resp = dashscope.Generation.call(model="qwen-max", prompt=prompt, result_format="text")
    return resp.output.text.strip()


def _read_config_key() -> str | None:
    try:
        cfg = json.loads((Path.home() / ".nanobot" / "config.json").read_text(encoding="utf-8"))
        return cfg.get("providers", {}).get("dashscope", {}).get("apiKey")
    except Exception:
        return None
