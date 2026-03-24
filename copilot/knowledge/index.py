"""Vector index helpers for the local interview knowledge base."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from copilot.config import get_embedding_settings

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE_DIR = ROOT / "data" / "knowledge_base"
VECTOR_DB_DIR = ROOT / "data" / "vector_db"
COLLECTION_NAME = "interview_rag"
EMBEDDING_BATCH_SIZE = 10

_collection = None


def get_chroma_collection():
    global _collection
    if _collection is not None:
        return _collection

    _collection = _create_collection()
    if _collection.count() == 0 and KNOWLEDGE_BASE_DIR.exists():
        _ingest_knowledge_base(_collection)
    return _collection


def rebuild_chroma_collection():
    global _collection

    client = _create_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    _collection = _create_collection(client)
    if KNOWLEDGE_BASE_DIR.exists():
        _ingest_knowledge_base(_collection)
    return _collection


def _recursive_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    separators: list[str] | None = None,
) -> list[str]:
    if separators is None:
        separators = ["\n## ", "\n### ", "\n\n", "\n", "、", ".", "。", " "]

    if not text or len(text) <= chunk_size:
        return [text] if text and text.strip() else []

    for separator in separators:
        parts = text.split(separator)
        if len(parts) <= 1:
            continue

        chunks: list[str] = []
        current = parts[0]

        for part in parts[1:]:
            candidate = current + separator + part
            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current.strip():
                chunks.append(current.strip())

            tail = current[-overlap:] if overlap and len(current) > overlap else ""
            current = f"{tail}{separator}{part}" if tail else f"{separator}{part}"

        if current.strip():
            chunks.append(current.strip())

        remaining = separators[separators.index(separator) + 1 :]
        final: list[str] = []
        for chunk in chunks:
            if len(chunk) > chunk_size and remaining:
                final.extend(_recursive_chunk(chunk, chunk_size, overlap, remaining))
            else:
                final.append(chunk)
        return final

    step = max(chunk_size - overlap, 1)
    return [
        text[i : i + chunk_size].strip()
        for i in range(0, len(text), step)
        if text[i : i + chunk_size].strip()
    ]


def _create_collection(client=None):
    client = client or _create_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_create_embedding_function(),
    )


def _create_embedding_function():
    settings = get_embedding_settings()
    if not settings["api_key"]:
        provider = settings["provider"].upper()
        raise ValueError(f"Missing {provider} API key for embeddings.")

    return OpenAIEmbeddingFunction(
        api_key=settings["api_key"],
        api_base=settings["base_url"],
        model_name=settings["model"],
    )


def _create_client():
    VECTOR_DB_DIR.parent.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(VECTOR_DB_DIR))


def _ingest_knowledge_base(collection) -> None:
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []
    ids: list[str] = []

    for index, (source, chunk) in enumerate(_iter_knowledge_chunks()):
        documents.append(chunk)
        metadatas.append({"source": source})
        ids.append(f"doc_{index}")

    if not documents:
        return

    for start in range(0, len(documents), EMBEDDING_BATCH_SIZE):
        end = start + EMBEDDING_BATCH_SIZE
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    logger.info("Indexed %d knowledge chunks.", len(documents))


def _iter_knowledge_chunks() -> Iterator[tuple[str, str]]:
    for markdown_file in KNOWLEDGE_BASE_DIR.rglob("*.md"):
        try:
            content = markdown_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.warning("Failed to read %s: %s", markdown_file.name, exc)
            continue

        for chunk in _recursive_chunk(content, chunk_size=500, overlap=100):
            if len(chunk) > 30:
                yield markdown_file.name, chunk
