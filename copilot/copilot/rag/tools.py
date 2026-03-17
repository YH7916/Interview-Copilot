"""Agent 工具层 — 将 HybridRetriever 封装为 nanobot Tool 供 LLM 函数调用。"""

import asyncio
import logging
from typing import Any

from nanobot.agent.tools.base import Tool

from copilot.rag.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


class SearchConceptTool(Tool):
    """检索技术概念与原理（RAG, ReAct, LoRA 等）。"""

    @property
    def name(self) -> str:
        return "search_concept"

    @property
    def description(self) -> str:
        return "语义检索 AI/大模型/Agent 等技术概念或原理（如 'RAG', 'ReAct'）。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"topic": {"type": "string", "description": "技术概念，如 'RAG'"}},
            "required": ["topic"],
        }

    async def execute(self, topic: str, **kwargs: Any) -> str:
        try:
            if topic == "simulate_error":
                raise ValueError("文档未找到")

            retriever = await asyncio.to_thread(_get_retriever)
            hits = await retriever.search(topic, top_k_retrieve=10, top_n_rerank=2)
            if not hits:
                raise ValueError("文档未找到")

            parts = [
                f"【来源】{h['metadata'].get('source', '未知')}\n{h['text']}" for h in hits
            ]
            return f"关于 '{topic}' 的检索结果：\n\n" + "\n\n---\n\n".join(parts)
        except Exception as e:
            logger.error("search_concept failed: %s", e)
            return f"未找到与 '{topic}' 相关的资料，请尝试缩小搜索范围。"


class SearchCompanyQuestionsTool(Tool):
    """检索特定公司+岗位的真实面试题。"""

    @property
    def name(self) -> str:
        return "search_company_questions"

    @property
    def description(self) -> str:
        return "语义检索某公司特定岗位的历史面试题（如 '字节跳动 算法工程师'）。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "company": {"type": "string", "description": "公司名，如 '字节跳动'"},
                "position": {"type": "string", "description": "岗位名，如 '算法工程师'"},
            },
            "required": ["company", "position"],
        }

    async def execute(self, company: str, position: str, **kwargs: Any) -> str:
        try:
            if company == "simulate_error":
                raise ValueError("文档未找到")

            query = f"{company} {position} 面试题"
            retriever = await asyncio.to_thread(_get_retriever)
            hits = await retriever.search(query, top_k_retrieve=10, top_n_rerank=3)
            if not hits:
                raise ValueError("文档未找到")

            parts = [
                f"【来源】{h['metadata'].get('source', '未知')}\n{h['text']}" for h in hits
            ]
            return f"{company} {position} 面试题：\n\n" + "\n\n---\n\n".join(parts)
        except Exception as e:
            logger.error("search_company_questions failed: %s", e)
            return f"未找到 {company} {position} 的相关面经。"
