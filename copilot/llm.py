"""Small helpers for calling OpenAI-compatible text models."""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI, OpenAI

from copilot.config import get_text_settings


def create_client(*, task: str = "chat", provider: str | None = None) -> tuple[OpenAI, dict[str, Any]]:
    settings = get_text_settings(task=task, provider=provider)
    if not settings["api_key"]:
        raise ValueError(f"Missing API key for provider '{settings['provider']}'.")
    client = OpenAI(api_key=settings["api_key"], base_url=settings["base_url"])
    return client, settings


def create_async_client(
    *,
    task: str = "chat",
    provider: str | None = None,
) -> tuple[AsyncOpenAI, dict[str, Any]]:
    settings = get_text_settings(task=task, provider=provider)
    if not settings["api_key"]:
        raise ValueError(f"Missing API key for provider '{settings['provider']}'.")
    client = AsyncOpenAI(api_key=settings["api_key"], base_url=settings["base_url"])
    return client, settings


def call_text(
    prompt: str,
    *,
    task: str = "chat",
    model: str | None = None,
    provider: str | None = None,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    client, settings = create_client(task=task, provider=provider)
    response = client.chat.completions.create(
        **_build_request(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model or settings["model"],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )
    return _extract_text(response)


async def call_text_async(
    prompt: str,
    *,
    task: str = "chat",
    model: str | None = None,
    provider: str | None = None,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    client, settings = create_async_client(task=task, provider=provider)
    response = await client.chat.completions.create(
        **_build_request(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model or settings["model"],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )
    return _extract_text(response)


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return json.loads(cleaned)


def _build_request(
    *,
    prompt: str,
    system_prompt: str | None,
    model: str,
    temperature: float | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(prompt, system_prompt),
    }
    if temperature is not None:
        request["temperature"] = temperature
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    return request


def _build_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _extract_text(response: Any) -> str:
    return (response.choices[0].message.content or "").strip()
