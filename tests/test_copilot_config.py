from __future__ import annotations

import json

from copilot.config import (
    get_provider_api_key,
    get_rerank_settings,
    get_text_settings,
    load_nanobot_config,
    resolve_provider,
)


def test_resolve_provider_prefers_openai_when_available(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(
        "copilot.config.load_nanobot_config",
        lambda path=None: {
            "providers": {
                "openai": {"apiKey": "openai-key"},
                "dashscope": {"apiKey": "dashscope-key"},
            }
        },
    )

    assert resolve_provider() == "openai"
    assert get_text_settings()["provider"] == "openai"


def test_provider_api_key_falls_back_to_nanobot_config(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "copilot.config.load_nanobot_config",
        lambda path=None: {"providers": {"openai": {"apiKey": "file-key"}}},
    )

    assert get_provider_api_key("openai") == "file-key"


def test_rerank_prefers_dashscope_when_available(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(
        "copilot.config.load_nanobot_config",
        lambda path=None: {
            "providers": {
                "openai": {"apiKey": "openai-key"},
                "dashscope": {"apiKey": "dashscope-key"},
            }
        },
    )

    settings = get_rerank_settings()
    assert settings["provider"] == "dashscope"
    assert settings["model"] == "gte-rerank"


def test_text_settings_support_model_overrides_from_config(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "copilot.config.load_nanobot_config",
        lambda path=None: {
            "providers": {
                "openai": {
                    "apiKey": "openai-key",
                    "chatModel": "my-chat-model",
                    "embeddingModel": "my-embedding-model",
                }
            }
        },
    )

    assert get_text_settings("chat")["model"] == "my-chat-model"
    assert get_text_settings("analysis")["model"] == "gpt-5.4"
    assert get_text_settings("rewrite")["model"] == "gpt-5.4"


def test_capability_provider_overrides(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(
        "copilot.config.load_nanobot_config",
        lambda path=None: {
            "copilot": {
                "textProvider": "openai",
                "embeddingProvider": "dashscope",
                "rerankProvider": "dashscope",
            },
            "providers": {
                "openai": {"apiKey": "openai-key"},
                "dashscope": {"apiKey": "dashscope-key"},
            },
        },
    )

    assert get_text_settings()["provider"] == "openai"
    assert get_rerank_settings()["provider"] == "dashscope"


def test_load_nanobot_config_accepts_utf8_bom(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"providers": {"openai": {"apiKey": "file-key"}}}),
        encoding="utf-8-sig",
    )

    assert load_nanobot_config(config_path)["providers"]["openai"]["apiKey"] == "file-key"
