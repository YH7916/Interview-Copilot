"""
ReviewAgent — 面试复盘分析器

职责：
  在单轮面试结束后的“垃圾时间”，静静地分析刚才所有的对话历史，
  提炼出候选人表现出的薄弱环节，并调用 WeaknessTracker 写入错题本。
"""
from __future__ import annotations # 启用 Python 3.7+ 的类型注解特性，例如 `list[dict]`

import asyncio # 用于异步操作，例如在后台运行分析任务
import logging # 用于记录日志，方便调试和监控
from datetime import datetime # 用于获取当前时间，标记错题本条目

# 复用 WeaknessTracker 的核心逻辑
from copilot.memory.weakness_tracker import WeaknessTracker, _call_llm

logger = logging.getLogger(__name__) # 获取当前模块的日志记录器


def _infer_severity(bullets: str) -> str:
    """根据薄弱点标记推断严重程度。"""
    if "✅" in bullets and "❌" not in bullets and "⚠️" not in bullets:
        return "🟢"
    if bullets.count("❌") >= 2:
        return "🔴"
    return "🟡"

# 分析对话的专用 Prompt
_PROMPT = """\
你是一个面试复盘分析师。请阅读以下模拟面试对话，提炼候选人暴露出的知识薄弱点。

【对话记录】
{dialogue}

【输出规则】
- 若候选人表现良好，仅输出：- ✅ 本次面试表现良好，无明显薄弱点
- 否则输出 2-4 条 Markdown 列表项，格式：
  `- ❌ **知识点名称**: 一句话说明具体缺失` (回答基本正确用⚠️，明显错误用❌)
- 不输出任何额外内容"""


class ReviewAgent:
    """复盘 Agent：通过异步执行，不让繁重的总结工作阻塞用户的交互。"""

    def __init__(self, tracker: WeaknessTracker | None = None):
        # 默认关联内置的错题本追踪器
        self.tracker = tracker or WeaknessTracker() # 初始化 WeaknessTracker 实例，如果未提供则创建一个新的

    def analyze_background(self, messages: list[dict]) -> None:
        """
        非阻塞接口：在后台开启一个新的 asyncio 协程运行。
        适用于面试结束后，你想立刻回个“再见”，让分析任务在后台跑。
        """
        asyncio.create_task(self._analyze_async(messages))

    async def analyze(self, messages: list[dict]) -> str:
        """
        阻塞接口：等待分析完成并返回结果。通常用于手动调用或单元测试。
        """
        return await asyncio.to_thread(self._analyze_sync, messages)

    # ── 内部逻辑 ────────────────────────────────────────────────────

    async def _analyze_async(self, messages: list[dict]) -> None:
        """协程包装：防止同步调用卡死事件循环。"""
        try:
            # 推送到线程池运行同步的 LLM 调用
            await asyncio.to_thread(self._analyze_sync, messages)
            logger.info("ReviewAgent: 错题本已更新 (%s)", self.tracker.log_path)
        except Exception:
            logger.exception("ReviewAgent: 分析失败")

    def _analyze_sync(self, messages: list[dict]) -> str:
        """同步分析逻辑：格式化对话 -> 调模型 -> 保存结果。"""
        # 1. 把复杂的消息列表转换成纯文本对白
        dialogue = _format_dialogue(messages)
        try:
            # 2. 调用 Qwen 提炼薄弱点
            bullets = _call_llm(_PROMPT.format(dialogue=dialogue))
        except Exception as e:
            # 3. 容错：如果网络不通，用简单的正则/关键字模拟提取
            bullets = _fallback_extract(messages, e)

        # 4. 格式化日志条目
        icon = _infer_severity(bullets)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"## [{ts}] 🤖 面试复盘 {icon}\n\n{bullets}\n"
        
        # 5. 持久化
        self.tracker._append(entry)
        return bullets


# ── 工具函数 ──────────────────────────────────────────────────────

def _format_dialogue(messages: list[dict]) -> str:
    """将消息列表 (List[Dict]) 精简为对话记录，去除干扰信息。"""
    lines = []
    for m in messages:
        role = "面试官" if m.get("role") == "assistant" else "候选人"
        content = m.get("content", "")
        # 处理多模态或复杂 content 列表，提取文本部分
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        if content:
            # 每一行截断，防止单条消息太长浪费消耗 Token
            lines.append(f"[{role}]: {content[:300]}")
    return "\n".join(lines)


def _fallback_extract(messages: list[dict], err: Exception) -> str:
    """兜底策略：当 LLM 故障时，搜索对话中候选人说“不懂/不知道”的话作为弱点凭据。"""
    WEAK_KEYWORDS = ("不知道", "不清楚", "不太确定", "不对", "错了", "抱歉", "理解有误")
    hints = [
        f"- ⚠️ **{m.get('content','?')[:40]}**: 候选人在此处表达了不确定或错误"
        for m in messages
        if m.get("role") == "user"
        and any(kw in (m.get("content") or "") for kw in WEAK_KEYWORDS)
    ]
    return "\n".join(hints[:4]) or f"- ⚠️ 自动分析失败({err})，请人工复盘"
