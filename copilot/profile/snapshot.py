"""Helpers for parsing structured candidate snapshot text."""

from __future__ import annotations

import re
from typing import Any

GENERIC_PROJECT_KEYWORDS = {
    "负责",
    "项目",
    "系统",
    "场景",
    "模块",
    "策略",
    "架构",
    "优化",
    "前端",
    "后端",
    "技术",
    "实现",
    "设计",
    "面试",
    "模拟",
    "岗位",
    "集成",
    "拆分",
    "负载",
    "性能",
    "动态",
    "问答",
    "功能",
}


def parse_candidate_projects(snapshot: str) -> list[dict[str, Any]]:
    projects: list[dict[str, Any]] = []
    for raw_line in str(snapshot or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("- Project "):
            continue
        _, _, body = line.partition(":")
        parts = [part.strip() for part in body.split("|") if part.strip()]
        if not parts:
            continue

        project = {
            "name": parts[0],
            "summary": "",
            "ownership": "",
            "tech": [],
            "deep_dive_points": [],
            "keywords": [],
        }
        for part in parts[1:]:
            lowered = part.lower()
            if lowered.startswith("ownership:"):
                project["ownership"] = part.split(":", 1)[1].strip()
            elif lowered.startswith("tech:"):
                project["tech"] = _split_csv(part.split(":", 1)[1])
            elif lowered.startswith("deep dive:"):
                project["deep_dive_points"] = _split_csv(part.split(":", 1)[1])
            else:
                if not project["summary"]:
                    project["summary"] = part
        project["keywords"] = _collect_keywords(project)
        projects.append(project)
    return projects


def _split_csv(text: str) -> list[str]:
    result: list[str] = []
    for item in re.split(r"[,，/]", str(text or "")):
        value = item.strip()
        if value and value not in result:
            result.append(value)
    return result


def _collect_keywords(project: dict[str, Any]) -> list[str]:
    result: list[str] = []
    seed_fields = (
        [project.get("name", "")],
        project.get("tech", []),
        project.get("deep_dive_points", []),
    )
    for field in seed_fields:
        values = field if isinstance(field, list) else [field]
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            if text not in result:
                result.append(text)
            for token in re.findall(r"[A-Za-z][A-Za-z0-9+._-]{1,24}|[\u4e00-\u9fff]{2,12}", text):
                if _is_discriminative_keyword(token) and token not in result:
                    result.append(token)
    return result


def _is_discriminative_keyword(token: str) -> bool:
    value = str(token or "").strip()
    if len(value) < 2:
        return False
    if "?" in value:
        return False
    if value in GENERIC_PROJECT_KEYWORDS:
        return False
    if re.fullmatch(r"[\u4e00-\u9fff]{2}", value):
        return False
    return True


__all__ = ["parse_candidate_projects"]
