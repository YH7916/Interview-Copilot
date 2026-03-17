import asyncio
import logging
from typing import Any

from nanobot.agent.tools.base import Tool

from copilot.rag.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

_hybrid_retriever: HybridRetriever | None = None


def _get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


class SearchConceptTool(Tool):
    """Tool to search for technical concepts and principles."""

    @property
    def name(self) -> str:
        return "search_concept"

    @property
    def description(self) -> str:
        return "专门用于语义检索具体的AI、大模型、Agent等技术概念或原理（如 'RAG', 'ReAct'）。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "要搜索的技术概念的主题，例如 'RAG'",
                }
            },
            "required": ["topic"],
        }

    async def execute(self, topic: str, **kwargs: Any) -> str:
        try:
            if topic == "simulate_error":
                raise ValueError("文档未找到")

            retriever = await asyncio.to_thread(_get_hybrid_retriever)
            results_data = await retriever.search(topic, top_k_retrieve=10, top_n_rerank=2)

            results = []
            for item in results_data:
                snippet = item.get("text", "")
                meta = item.get("metadata", {})
                source = meta.get("source", "未知来源")
                results.append(f"【来源文件】: {source}\n【相关内容】: ...{snippet}...")

            if not results:
                raise ValueError("文档未找到")

            return f"为您检索到以下关于 '{topic}' 的高度相关资料：\n\n" + "\n\n---\n\n".join(results)

        except Exception as e:
            logger.error(f"【DEBUG 抓虫】真正的报错原因是: {repr(e)}")
            return f"未在面经库中找到与 '{topic}' 相关的八股文或概念解答，请尝试询问通用技术问题或缩减搜索范围。"


class SearchCompanyQuestionsTool(Tool):
    """Tool to search for real interview questions from specific companies."""

    @property
    def name(self) -> str:
        return "search_company_questions"

    @property
    def description(self) -> str:
        return "专门用于语义检索某公司特定岗位的真实历史面试题（如 '字节跳动', '算法工程师'）。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "company": {
                    "type": "string",
                    "description": "公司名称，如 '字节跳动'",
                },
                "position": {
                    "type": "string",
                    "description": "岗位名称，如 '算法工程师'",
                }
            },
            "required": ["company", "position"],
        }

    async def execute(self, company: str, position: str, **kwargs: Any) -> str:
        try:
            if company == "simulate_error":
                raise ValueError("文档未找到")

            search_query = f"{company} {position} 面试题"
            retriever = await asyncio.to_thread(_get_hybrid_retriever)
            results_data = await retriever.search(search_query, top_k_retrieve=10, top_n_rerank=3)

            results = []
            for item in results_data:
                snippet = item.get("text", "")
                meta = item.get("metadata", {})
                source = meta.get("source", "未知来源")
                results.append(f"【来源文件】: {source}\n【面试题片段】: ...{snippet}...")

            if not results:
                raise ValueError("文档未找到")

            return f"为您检索到以下关于 {company} {position} 的高匹配度面试题：\n\n" + "\n\n---\n\n".join(results)

        except Exception as e:
            logger.error(f"【DEBUG 抓虫】真正的报错原因是: {repr(e)}")
            return f"未在面经库中找到 {company} {position} 职位的相关面经，请尝试询问通用技术问题或缩减搜索范围。"
