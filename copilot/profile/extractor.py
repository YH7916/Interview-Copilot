"""Candidate profile extraction from resume-like text, LLM-first with fallback."""

from __future__ import annotations

import json
import re

from copilot.config import get_text_settings
from copilot.llm import call_text, parse_json_response

RESUME_EXTRACT_SYSTEM_PROMPT = (
    "你是一位非常懂技术面试的简历分析助手。"
    "请从候选人的原始简历中提炼出最适合面试官使用的候选人画像。"
    "重点抓住 1 到 2 个最值得深挖的核心项目，不要泛泛罗列所有经历。"
    "如果简历里两个项目已经足够清晰，就围绕这两个项目总结。"
    "输出必须是 JSON。"
)


def build_candidate_profile_summary(text: str, *, max_chars: int = 4000) -> str:
    raw_text = str(text or "").strip()
    normalized = _normalize_source_text(raw_text)
    if not normalized:
        return ""

    llm_summary = _build_candidate_profile_summary_with_llm(raw_text, max_chars=max_chars)
    if llm_summary:
        return llm_summary

    return _build_candidate_profile_summary_with_rules(normalized, max_chars=max_chars)


def _build_candidate_profile_summary_with_llm(text: str, *, max_chars: int) -> str:
    if not _llm_enabled() or not _should_use_llm(text):
        return ""

    prompt = {
        "task": "extract_candidate_snapshot",
        "requirements": [
            "优先总结 1 到 2 个核心项目",
            "每个项目尽量写出：项目名/方向、场景、候选人职责、关键技术、可深挖点",
            "如果教育背景和当前阶段很明确，也一并提炼",
            "不要编造简历里没有的信息",
        ],
        "resume_text": _truncate(text, max_chars=max(1500, max_chars * 2)),
    }
    try:
        response = call_text(
            json.dumps(prompt, ensure_ascii=False, indent=2),
            task="analysis",
            system_prompt=(
                RESUME_EXTRACT_SYSTEM_PROMPT
                + ' 输出格式: {"education":"","stage":"","focus_areas":[],"core_projects":[{"name":"","summary":"","candidate_ownership":"","tech":[],"deep_dive_points":[]}],"skills":[]}'
            ),
            temperature=0.2,
            max_tokens=700,
        )
        payload = parse_json_response(response)
    except Exception:
        return ""

    return _render_llm_snapshot(payload, max_chars=max_chars)


def _render_llm_snapshot(payload: dict, *, max_chars: int) -> str:
    if not isinstance(payload, dict):
        return ""

    lines = ["Candidate Snapshot"]
    education = str(payload.get("education", "") or "").strip()
    stage = str(payload.get("stage", "") or "").strip()
    focus_areas = _as_clean_list(payload.get("focus_areas"))
    skills = _as_clean_list(payload.get("skills"))
    projects = payload.get("core_projects")

    if education:
        lines.append(f"- Education: {education}")
    if stage:
        lines.append(f"- Stage: {stage}")
    if focus_areas:
        lines.append(f"- Focus Areas: {', '.join(focus_areas[:8])}")

    if isinstance(projects, list):
        for index, item in enumerate(projects[:2], 1):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "").strip()
            summary = str(item.get("summary", "") or "").strip()
            ownership = str(item.get("candidate_ownership", "") or "").strip()
            tech = _as_clean_list(item.get("tech"))
            deep_dive_points = _as_clean_list(item.get("deep_dive_points"))

            parts = []
            if name:
                parts.append(name)
            if summary:
                parts.append(summary)
            if ownership:
                parts.append(f"Ownership: {ownership}")
            if tech:
                parts.append(f"Tech: {', '.join(tech[:8])}")
            if deep_dive_points:
                parts.append(f"Deep Dive: {', '.join(deep_dive_points[:5])}")
            if parts:
                lines.append(f"- Project {index}: {' | '.join(parts)}")

    if skills:
        lines.append(f"- Skills: {', '.join(skills[:12])}")

    if len(lines) <= 1:
        return ""
    return _truncate("\n".join(lines), max_chars=max_chars)


def _build_candidate_profile_summary_with_rules(text: str, *, max_chars: int = 4000) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    sections = _split_sections(lines)

    education = _pick_education(sections, lines)
    stage = _pick_stage(lines)
    projects = _pick_projects(sections, lines)
    skills = _pick_skills(sections, lines)
    focus_areas = _pick_focus_areas(projects, skills, lines)

    summary_lines: list[str] = ["Candidate Snapshot"]
    if education:
        summary_lines.append(f"- Education: {education}")
    if stage:
        summary_lines.append(f"- Stage: {stage}")
    if focus_areas:
        summary_lines.append(f"- Focus Areas: {', '.join(focus_areas[:8])}")

    for index, project in enumerate(projects[:3], 1):
        summary_lines.append(f"- Project {index}: {project}")

    if skills:
        summary_lines.append(f"- Skills: {', '.join(skills[:12])}")

    if len(summary_lines) == 1:
        return _truncate(text, max_chars=max_chars)

    summary = "\n".join(summary_lines)
    if len(summary) < 120:
        raw_fallback = _truncate(text, max_chars=max(240, max_chars // 2))
        summary = f"{summary}\n- Raw Notes: {raw_fallback.replace(chr(10), ' | ')}"
    return _truncate(summary, max_chars=max_chars)


def _normalize_source_text(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in str(text or "").replace("\r", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(=+|#+)\s*", "", line)
        line = line.replace("=>", " ")
        line = re.sub(r"#\w+:", " ", line)
        line = re.sub(r"[\{\}\[\]\(\)<>]", " ", line)
        line = line.replace("\\", " ")
        line = re.sub(r"[_*`|]", " ", line)
        line = re.sub(r"\s+", " ", line).strip(" -:")
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"__root__": []}
    current = "__root__"
    for line in lines:
        label = _section_label(line)
        if label:
            current = label
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _section_label(line: str) -> str:
    lowered = line.lower()
    mapping = {
        "education": ("education", "教育", "学校", "学历"),
        "projects": ("project", "projects", "项目", "经历", "experience"),
        "skills": ("skills", "skill", "技术栈", "技能", "tools"),
    }
    for label, tokens in mapping.items():
        if any(token in lowered for token in tokens) and len(line) <= 24:
            return label
    return ""


def _pick_education(sections: dict[str, list[str]], lines: list[str]) -> str:
    candidates = sections.get("education", []) + [
        line
        for line in lines
        if any(token in line.lower() for token in ("大学", "学院", "university", "college", "本科", "硕士", "phd", "博士"))
    ]
    return _compact_sentence(candidates[:3])


def _pick_stage(lines: list[str]) -> str:
    for line in lines:
        stage_match = re.search(
            r"(大一|大二|大三|大四|本科|硕士|研究生|博士|应届|研一|研二|研三|20\d{2}年毕业|预计20\d{2})",
            line,
            re.IGNORECASE,
        )
        if stage_match:
            return line
    return ""


def _pick_projects(sections: dict[str, list[str]], lines: list[str]) -> list[str]:
    project_lines = list(sections.get("projects", []))
    if not project_lines:
        project_lines = [
            line
            for line in lines
            if any(token in line.lower() for token in ("project", "项目", "agent", "rag", "workflow", "memory", "webworker"))
        ]

    projects: list[str] = []
    for line in project_lines:
        if len(line) < 6:
            continue
        if line not in projects:
            projects.append(line)
    return projects[:6]


def _pick_skills(sections: dict[str, list[str]], lines: list[str]) -> list[str]:
    skill_text = " ".join(sections.get("skills", []) or lines[-6:])
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+._-]{1,20}", skill_text)
    keywords = []
    for token in tokens:
        lowered = token.lower()
        if lowered in {
            "python",
            "typescript",
            "javascript",
            "react",
            "node",
            "langchain",
            "autogen",
            "rag",
            "bm25",
            "rerank",
            "chromadb",
            "webworker",
            "docker",
            "fastapi",
            "pytorch",
            "vllm",
            "redis",
            "mysql",
            "postgresql",
            "qwen",
            "openai",
            "codex",
        } and token not in keywords:
            keywords.append(token)
    return keywords


def _pick_focus_areas(projects: list[str], skills: list[str], lines: list[str]) -> list[str]:
    joined = " ".join([*projects, *skills, *lines]).lower()
    mapping = [
        ("multi-agent", "multi-agent"),
        ("agent", "agent"),
        ("rag", "RAG"),
        ("retrieval", "retrieval"),
        ("rerank", "rerank"),
        ("memory", "memory"),
        ("workflow", "workflow"),
        ("webworker", "webworker"),
        ("evaluation", "evaluation"),
        ("deploy", "deployment"),
        ("python", "python"),
    ]
    result = []
    for token, label in mapping:
        if token in joined and label not in result:
            result.append(label)
    return result


def _compact_sentence(lines: list[str]) -> str:
    parts: list[str] = []
    for line in lines:
        value = str(line or "").strip()
        if not value or value in parts:
            continue
        parts.append(value)
    return " | ".join(parts[:3])


def _as_clean_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _llm_enabled() -> bool:
    try:
        return bool(get_text_settings(task="analysis").get("api_key"))
    except Exception:
        return False


def _should_use_llm(text: str) -> bool:
    value = str(text or "").strip()
    if len(value) >= 180:
        return True
    if value.count("\n") >= 4:
        return True
    lowered = value.lower()
    return any(token in lowered for token in ("project", "projects", "项目", "教育", "education", "experience", "skills"))


def _truncate(text: str, *, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 4].rstrip() + "\n..."


__all__ = ["build_candidate_profile_summary", "RESUME_EXTRACT_SYSTEM_PROMPT"]
