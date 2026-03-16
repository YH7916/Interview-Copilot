import os
import logging
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

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

def get_chroma_collection():
    global _collection
    if _collection is not None:
        return _collection
        
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量以启用云端向量化")
        
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
                # 按照段落简单切分
                paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 30]
                
                # 如果文件全是对白没空行，退化为按长度截断
                if not paragraphs:
                    chunk_size = 500
                    paragraphs = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size) if len(content[i:i+chunk_size].strip()) > 30]

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
