"""统一配置入口 — 所有模块通过此处读取 API Key，避免重复逻辑。"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def get_dashscope_api_key() -> str | None:
    """优先级：环境变量 > ~/.nanobot/config.json"""
    key = os.environ.get("DASHSCOPE_API_KEY")
    if key:
        return key
    try:
        cfg_path = Path.home() / ".nanobot" / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return cfg.get("providers", {}).get("dashscope", {}).get("apiKey")
    except Exception:
        return None
