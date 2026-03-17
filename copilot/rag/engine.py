import logging
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from copilot.config import get_dashscope_api_key

logger = logging.getLogger(__name__)

# 获取项目根目录，由于现在位置是 copilot/rag/engine.py
# __file__ -> engine.py
# parent -> rag/
# parent.parent -> copilot/
# parent.parent.parent -> Interview-Copilot/
BASE_DIR = Path(__file__).resolve().parent.parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "data" / "knowledge_base"
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db"

_collection = None


def _recursive_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    separators: list[str] | None = None,
) -> list[str]:
    """递归分块：按 Markdown 结构层级切分，相邻块之间保留 overlap 字符防止语义断裂。"""
    if separators is None:
        separators = ["\n## ", "\n### ", "\n\n", "\n", "。", ".", "，", " "]

    if not text or len(text) <= chunk_size:
        return [text] if text and text.strip() else []

    # 尝试用当前优先级最高的分隔符切分
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks: list[str] = []
            current = parts[0]
            for part in parts[1:]:
                candidate = current + sep + part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current.strip():
                        chunks.append(current.strip())
                    # overlap: 保留上一个 chunk 末尾的 overlap 个字符作为下一个 chunk 的开头
                    tail = current[-overlap:] if overlap and len(current) > overlap else ""
                    current = tail + sep + part if tail else sep + part
            if current.strip():
                chunks.append(current.strip())

            # 如果某个 chunk 仍然超长，用下一级分隔符递归切分
            final: list[str] = []
            remaining_seps = separators[separators.index(sep) + 1:]
            for chunk in chunks:
                if len(chunk) > chunk_size and remaining_seps:
                    final.extend(_recursive_chunk(chunk, chunk_size, overlap, remaining_seps))
                else:
                    final.append(chunk)
            return final

    # 所有分隔符都试完了，强制按字符截断
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


def get_chroma_collection():
    global _collection
    if _collection is not None:
        return _collection
        
    api_key = get_dashscope_api_key()
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或 ~/.nanobot/config.json 以启用云端向量化")
        
    # 阿里云百炼 (DashScope) 兼容 OpenAI 的 Embedding 接口
    openai_ef = OpenAIEmbeddingFunction(
        api_key=api_key,
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="text-embedding-v3"
    )

    VECTOR_DB_DIR.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 使用持久化客户端存储向量数据到本地文件系统
        client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        _collection = client.get_or_create_collection(
            name="interview_rag", 
            embedding_function=openai_ef
        )
    except Exception as e:
        logger.error(f"初始化 ChromaDB 失败: {e}")
        raise ValueError(f"向量数据库初始化失败: {e}")

    # 初始化知识库数据 (简单的 Chunking)
    if _collection.count() == 0 and KNOWLEDGE_BASE_DIR.exists():
        logger.info("初始化知识库向量数据，这可能需要一点时间...")
        documents = []
        metadatas = []
        ids = []
        
        doc_id_counter = 0
        for md_file in KNOWLEDGE_BASE_DIR.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8", errors="ignore")
                # 递归分块：按 Markdown 结构切分 + overlap 防止语义断裂
                paragraphs = [c for c in _recursive_chunk(content, chunk_size=500, overlap=100) if len(c) > 30]

                for p in paragraphs:
                    documents.append(p)
                    metadatas.append({"source": md_file.name})
                    ids.append(f"doc_{doc_id_counter}")
                    doc_id_counter += 1
            except Exception as e:
                logger.warning(f"读取文件 {md_file} 时出错: {e}")
                
        if documents:
            # 批量添加以避免超过API限制 (DashScope text-embedding limit typically 25, we use 10 for safety)
            batch_size = 10
            try:
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i+batch_size]
                    batch_metas = metadatas[i:i+batch_size]
                    batch_ids = ids[i:i+batch_size]
                    _collection.add(
                        documents=batch_docs,
                        metadatas=batch_metas,
                        ids=batch_ids
                    )
                logger.info(f"成功将 {len(documents)} 个文本块写入向量数据库！")
            except Exception as e:
                logger.error(f"向向量数据库写入数据失败: {e}")
                raise ValueError(f"向量数据写入失败: {e}")
            
    return _collection
