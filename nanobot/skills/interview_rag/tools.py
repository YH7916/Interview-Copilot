import os
from pathlib import Path
from typing import Any
import logging
from nanobot.agent.tools.base import Tool

# --- 新增：配置日志和资料库路径 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 获取项目根目录下的 data/knowledge_base 路径
# Path(__file__).resolve() 是当前 tools.py 的路径，往上退 4 层正好是项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "data" / "knowledge_base"


class SearchConceptTool(Tool):
    """Tool to search for technical concepts and principles."""

    @property
    def name(self) -> str:
        return "search_concept"

    @property
    def description(self) -> str:
        # 文案升级：既然资料是大模型/Agent方向的，提示词也要跟着改
        return "专门用于检索具体的AI、大模型、Agent等技术概念或原理（如 'RAG', 'ReAct'）。"

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
            # 1. 保留之前的错误模拟防线
            if topic == "simulate_error":
                raise ValueError("文档未找到")

            # 2. 检查你是不是忘了建文件夹
            if not KNOWLEDGE_BASE_DIR.exists():
                return f"系统提示：知识库目录 {KNOWLEDGE_BASE_DIR} 不存在，请开发者先配置资料。"

            # 3. 核心：遍历读取 Markdown 文件寻找关键词 (轻量级 RAG 雏形)
            results = []
            for md_file in KNOWLEDGE_BASE_DIR.rglob("*.md"):
                try:
                    content = md_file.read_text(encoding="utf-8", errors="ignore")
                    if topic.lower() in content.lower():
                        # 找到关键词后，截取它前后的一段文本（模拟文本分块 Chunking）
                        idx = content.lower().find(topic.lower())
                        start = max(0, idx - 150)  # 往前截取 150 字符
                        end = min(len(content), idx + 800)  # 往后截取 800 字符
                        snippet = content[start:end]
                        
                        results.append(f"【来源文件】: {md_file.name}\n【相关内容】: ...{snippet}...")
                        
                        # 为防止塞爆大模型上下文，最多只取 2 个相关文件的片段
                        if len(results) >= 2:
                            break
                except Exception as e:
                    logger.warning(f"读取文件 {md_file} 时出错: {e}")

            # 4. 如果遍历完所有文件都没找到，触发我们的兜底防线
            if not results:
                raise ValueError("文档未找到")

            return f"为您检索到以下关于 '{topic}' 的参考资料：\n\n" + "\n\n---\n\n".join(results)

        except Exception as e:
            print(f"【DEBUG 抓虫】真正的报错原因是: {repr(e)}")
            return f"未在面经库中找到与 '{topic}' 相关的八股文或概念解答，请尝试询问通用技术问题或缩减搜索范围。"


class SearchCompanyQuestionsTool(Tool):
    """Tool to search for real interview questions from specific companies."""

    @property
    def name(self) -> str:
        return "search_company_questions"

    @property
    def description(self) -> str:
        return "专门用于检索某公司特定岗位的真实历史面试题（如 '字节跳动', '算法工程师'）。"

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

            if not KNOWLEDGE_BASE_DIR.exists():
                 return f"系统提示：知识库目录 {KNOWLEDGE_BASE_DIR} 不存在，请开发者先配置资料。"

            results = []
            search_keyword = company.lower() # 简化逻辑，先只搜公司名
            
            for md_file in KNOWLEDGE_BASE_DIR.rglob("*.md"):
                try:
                    content = md_file.read_text(encoding="utf-8", errors="ignore")
                    if search_keyword in content.lower():
                        idx = content.lower().find(search_keyword)
                        start = max(0, idx - 100)
                        end = min(len(content), idx + 800)
                        snippet = content[start:end]
                        results.append(f"【来源文件】: {md_file.name}\n【面试题片段】: ...{snippet}...")
                        if len(results) >= 2:
                            break
                except Exception as e:
                    pass

            if not results:
                raise ValueError("文档未找到")
                
            return f"为您检索到以下关于 {company} {position} 的面试题：\n\n" + "\n\n---\n\n".join(results)

        except Exception as e:
            print(f"【DEBUG 抓虫】真正的报错原因是: {repr(e)}")
            return f"未在面经库中找到 {company} {position} 职位的相关面经，请尝试询问通用技术问题或缩减搜索范围。"