"""Small configuration helpers shared by copilot modules."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULTS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "chat_model": "gpt-5.4",
        "analysis_model": "gpt-5.4",
        "rewrite_model": "gpt-5.4",
        "judge_model": "gpt-5.4",
        "embedding_model": "text-embedding-3-small",
        "rerank_model": None,
        "rerank_url": None,
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "chat_model": "qwen-max",
        "analysis_model": "qwen-max",
        "rewrite_model": "qwen-turbo",
        "judge_model": "qwen-max",
        "embedding_model": "text-embedding-v4",
        "rerank_model": "qwen3-rerank",
        "rerank_url": "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    },
}


def get_nanobot_config_path() -> Path:
    return Path.home() / ".nanobot" / "config.json"


def load_nanobot_config(path: Path | None = None) -> dict[str, Any]:
    try:
        return json.loads((path or get_nanobot_config_path()).read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def get_openai_api_key() -> str | None:
    return get_provider_api_key("openai")


def get_dashscope_api_key() -> str | None:
    return get_provider_api_key("dashscope")


def get_provider_api_key(provider: str) -> str | None:
    provider = normalize_provider(provider)
    env_name = "OPENAI_API_KEY" if provider == "openai" else "DASHSCOPE_API_KEY"
    return os.environ.get(env_name) or _provider_config(provider).get("apiKey")


def get_text_settings(task: str = "chat", provider: str | None = None) -> dict[str, Any]:
    task = normalize_task(task)
    chosen = normalize_provider(provider or _copilot_provider("textProvider") or resolve_provider())
    settings = get_provider_settings(chosen)
    return {
        "provider": settings["provider"],
        "api_key": settings["api_key"],
        "base_url": settings["base_url"],
        "model": settings[f"{task}_model"],
    }


def get_embedding_settings(provider: str | None = None) -> dict[str, Any]:
    chosen = normalize_provider(
        provider or _copilot_provider("embeddingProvider") or resolve_provider()
    )
    settings = get_provider_settings(chosen)
    return {
        "provider": settings["provider"],
        "api_key": settings["api_key"],
        "base_url": settings["base_url"],
        "model": settings["embedding_model"],
    }


def get_rerank_settings(provider: str | None = None) -> dict[str, Any]:
    chosen = normalize_provider(
        provider
        or _copilot_provider("rerankProvider")
        or os.environ.get("COPILOT_RERANK_PROVIDER")
        or _default_rerank_provider()
    )
    settings = get_provider_settings(chosen)
    return {
        "provider": settings["provider"],
        "api_key": settings["api_key"],
        "model": settings["rerank_model"],
        "url": settings["rerank_url"],
    }


def get_provider_settings(provider: str | None = None) -> dict[str, Any]:
    name = resolve_provider(provider)
    defaults = DEFAULTS[name]
    raw = _provider_config(name)
    return {
        "provider": name,
        "api_key": get_provider_api_key(name),
        "base_url": raw.get("apiBase") or raw.get("baseUrl") or defaults["base_url"],
        "chat_model": _model_override(raw, "chat_model") or defaults["chat_model"],
        "analysis_model": _model_override(raw, "analysis_model") or defaults["analysis_model"],
        "rewrite_model": _model_override(raw, "rewrite_model") or defaults["rewrite_model"],
        "judge_model": _model_override(raw, "judge_model") or defaults["judge_model"],
        "embedding_model": _model_override(raw, "embedding_model") or defaults["embedding_model"],
        "rerank_model": _model_override(raw, "rerank_model") or defaults["rerank_model"],
        "rerank_url": raw.get("rerankUrl") or defaults["rerank_url"],
    }


def resolve_provider(provider: str | None = None) -> str:
    if provider:
        return normalize_provider(provider)
    if os.environ.get("COPILOT_PROVIDER"):
        return normalize_provider(os.environ["COPILOT_PROVIDER"])
    if get_openai_api_key():
        return "openai"
    if get_dashscope_api_key():
        return "dashscope"
    return "openai"


def normalize_provider(provider: str | None) -> str:
    value = (provider or "").strip().lower()
    return value if value in DEFAULTS else "openai"


def normalize_task(task: str) -> str:
    value = "chat" if task in {"default", "text"} else task.strip().lower()
    if value not in {"chat", "analysis", "rewrite", "judge"}:
        raise ValueError(f"Unsupported task: {task}")
    return value


def _default_rerank_provider() -> str:
    return "dashscope" if get_dashscope_api_key() else resolve_provider()


def _provider_config(provider: str) -> dict[str, Any]:
    providers = load_nanobot_config().get("providers", {})
    value = providers.get(provider, {})
    return value if isinstance(value, dict) else {}


def _model_override(config: dict[str, Any], key: str) -> str | None:
    camel = "".join(part.capitalize() if index else part for index, part in enumerate(key.split("_")))
    return config.get(key) or config.get(camel)


def _copilot_provider(key: str) -> str | None:
    value = load_nanobot_config().get("copilot", {}).get(key)
    return value if isinstance(value, str) else None
