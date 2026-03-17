"""向量数据库引擎 — ChromaDB 初始化、递归分块、Embedding 入库。"""

import logging
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from copilot.config import get_dashscope_api_key

logger = logging.getLogger(__name__)

# 路径约定：engine.py → rag/ → copilot/ → 项目根
_ROOT = Path(__file__).resolve().parent.parent.parent
KNOWLEDGE_BASE_DIR = _ROOT / "data" / "knowledge_base"
VECTOR_DB_DIR = _ROOT / "data" / "vector_db"

_collection = None  # 全局单例，避免重复初始化


# ── Chunking ────────────────────────────────────────────────────


def _recursive_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    separators: list[str] | None = None,
) -> list[str]:
    """按 Markdown 层级递归切分，相邻块保留 overlap 字符防止语义断裂。

    分隔符优先级: ## 标题 > ### 子标题 > 空行 > 换行 > 标点 > 空格
    """
    if separators is None:
        separators = ["\n## ", "\n### ", "\n\n", "\n", "。", ".", "，", " "]

    if not text or len(text) <= chunk_size:
        return [text] if text and text.strip() else []

    for sep in separators:
        parts = text.split(sep)
        if len(parts) <= 1:
            continue

        chunks: list[str] = []
        current = parts[0]

        for part in parts[1:]:
            candidate = current + sep + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                # 取上一段末尾作为下一段的 overlap 前缀
                tail = current[-overlap:] if overlap and len(current) > overlap else ""
                current = (tail + sep + part) if tail else (sep + part)

        if current.strip():
            chunks.append(current.strip())

        # 超长块用下一级分隔符继续拆
        remaining = separators[separators.index(sep) + 1 :]
        final: list[str] = []
        for c in chunks:
            if len(c) > chunk_size and remaining:
                final.extend(_recursive_chunk(c, chunk_size, overlap, remaining))
            else:
                final.append(c)
        return final

    # 兜底：强制按字符截断
    return [
        text[i : i + chunk_size].strip()
        for i in range(0, len(text), chunk_size - overlap)
        if text[i : i + chunk_size].strip()
    ]


# ── ChromaDB ────────────────────────────────────────────────────


def get_chroma_collection():
    """懒加载 ChromaDB 集合；首次调用时自动索引知识库。"""
    global _collection
    if _collection is not None:
        return _collection

    api_key = get_dashscope_api_key()
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 或 ~/.nanobot/config.json")

    ef = OpenAIEmbeddingFunction(
        api_key=api_key,
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="text-embedding-v3",
    )

    VECTOR_DB_DIR.parent.mkdir(parents=True, exist_ok=True)
    try:
        client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        _collection = client.get_or_create_collection(name="interview_rag", embedding_function=ef)
    except Exception as e:
        raise ValueError(f"向量数据库初始化失败: {e}") from e

    # 空库时自动入库
    if _collection.count() == 0 and KNOWLEDGE_BASE_DIR.exists():
        _ingest_knowledge_base(_collection)

    return _collection


def _ingest_knowledge_base(collection) -> None:
    """扫描 knowledge_base/ 下所有 Markdown 并分块入库。"""
    logger.info("正在索引知识库...")
    documents, metadatas, ids = [], [], []
    counter = 0

    for md in KNOWLEDGE_BASE_DIR.rglob("*.md"):
        try:
            content = md.read_text(encoding="utf-8", errors="ignore")
            for chunk in _recursive_chunk(content, chunk_size=500, overlap=100):
                if len(chunk) > 30:
                    documents.append(chunk)
                    metadatas.append({"source": md.name})
                    ids.append(f"doc_{counter}")
                    counter += 1
        except Exception as e:
            logger.warning("读取 %s 失败: %s", md.name, e)

    if not documents:
        return

    # DashScope Embedding API 每批最多约 25 条，安全起见用 10
    batch = 10
    for i in range(0, len(documents), batch):
        collection.add(
            documents=documents[i : i + batch],
            metadatas=metadatas[i : i + batch],
            ids=ids[i : i + batch],
        )
    logger.info("索引完成，共 %d 个文本块", len(documents))
