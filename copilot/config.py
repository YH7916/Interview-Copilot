"""
统一配置工具 — 集中管理 DashScope API Key 的读取逻辑

优先级：环境变量 DASHSCOPE_API_KEY > ~/.nanobot/config.json > None
"""
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_dashscope_api_key() -> str | None:
    """读取 DashScope API Key，优先环境变量，其次 nanobot 配置文件。"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if api_key:
        return api_key
    try:
        cfg = json.loads((Path.home() / ".nanobot" / "config.json").read_text(encoding="utf-8"))
        return cfg.get("providers", {}).get("dashscope", {}).get("apiKey")
    except Exception:
        return None
