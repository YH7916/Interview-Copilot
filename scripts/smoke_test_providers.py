from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from copilot.config import get_embedding_settings, get_text_settings
from copilot.knowledge.retrieval import rerank
from copilot.llm import call_text


def test_chat() -> dict:
    settings = get_text_settings("chat")
    try:
        result = call_text("Reply with exactly: ok", task="chat", max_tokens=8)
        return {"ok": True, "settings": settings, "response": result}
    except Exception as exc:
        return {"ok": False, "settings": settings, "error": repr(exc)}


def test_embedding() -> dict:
    settings = get_embedding_settings()
    try:
        client = OpenAI(api_key=settings["api_key"], base_url=settings["base_url"])
        response = client.embeddings.create(model=settings["model"], input="agent interview")
        return {
            "ok": True,
            "settings": settings,
            "embedding_length": len(response.data[0].embedding),
        }
    except Exception as exc:
        return {"ok": False, "settings": settings, "error": repr(exc)}


async def test_rerank() -> dict:
    try:
        result = await rerank("agent", ["agent loop design", "database sharding"], top_n=1)
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def main() -> int:
    results = {
        "chat": test_chat(),
        "embedding": test_embedding(),
        "rerank": asyncio.run(test_rerank()),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
