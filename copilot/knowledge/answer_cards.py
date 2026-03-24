"""Build concise answer cards on top of the interview question bank."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import Any

from copilot.knowledge.question_bank import OUTPUT_DIR, load_question_bank
from copilot.llm import call_text_async, parse_json_response

ANSWER_CARDS_JSON_PATH = OUTPUT_DIR / "answer_cards.json"
ANSWER_CARDS_MARKDOWN_PATH = OUTPUT_DIR / "answer_cards.md"
RECENT_ANSWER_CARDS_JSON_PATH = OUTPUT_DIR / "recent_answer_cards.json"
RECENT_ANSWER_CARDS_MARKDOWN_PATH = OUTPUT_DIR / "recent_answer_cards.md"

GENERATED_SOURCES = {
    "question_bank.md",
    "recent_question_bank.md",
    "answer_cards.md",
    "recent_answer_cards.md",
}

SEARCH_RESULT_LINE = re.compile(r"^\d+\.\s+(.*)$")
UNTRUSTED_BANNER = "[External content"


class AnswerCardBuilder:
    def __init__(
        self,
        *,
        retriever: Any | None = None,
        search_tool: Any | None = None,
        fetch_tool: Any | None = None,
        llm: Any = call_text_async,
    ):
        self._retriever = retriever
        self._search_tool = search_tool
        self._fetch_tool = fetch_tool
        self._llm = llm

    async def build(
        self,
        bank: dict[str, Any],
        *,
        use_web: bool = False,
        max_cards: int | None = None,
    ) -> dict[str, Any]:
        categories = []
        remaining = max_cards

        for category in bank.get("categories", []):
            if remaining is not None and remaining <= 0:
                break

            items = category.get("clusters") or category.get("questions") or []
            cards = []

            for item in items:
                if remaining is not None and remaining <= 0:
                    break
                cards.append(await self.build_card(category, item, use_web=use_web))
                if remaining is not None:
                    remaining -= 1

            if cards:
                categories.append(
                    {
                        "name": category["name"],
                        "label": category["label"],
                        "stage": category["stage"],
                        "cards": cards,
                    }
                )

        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_bank_generated_at": bank.get("generated_at", ""),
            "categories": categories,
        }

    async def build_card(
        self,
        category: dict[str, Any],
        question: dict[str, Any],
        *,
        use_web: bool = False,
    ) -> dict[str, Any]:
        evidence = await self._collect_evidence(category, question, use_web=use_web)
        payload = await self._draft_card(question, evidence)
        indexes = _normalize_indexes(payload.get("supporting_indexes"), len(evidence))
        status = "needs_evidence" if not evidence else "grounded" if payload else "draft"

        return {
            "question": question["question"],
            "category": question["category"],
            "category_label": question["category_label"],
            "stage": question["stage"],
            "follow_ups": question.get("follow_ups", []),
            "source_count": int(question.get("source_count", 0)),
            "latest_source_at": question.get("latest_source_at", ""),
            "status": status,
            "answer": (payload.get("answer") or _fallback_answer(question)).strip(),
            "pitfalls": _normalize_lines(payload.get("pitfalls")) or _fallback_pitfalls(question),
            "evidence": [evidence[index - 1] for index in indexes] if indexes else evidence[:3],
        }

    async def _collect_evidence(
        self,
        category: dict[str, Any],
        question: dict[str, Any],
        *,
        use_web: bool,
    ) -> list[dict[str, str]]:
        evidence = _report_evidence(question)
        evidence.extend(await self._local_evidence(question["question"]))
        if use_web:
            evidence.extend(await self._web_evidence(category, question))
        return _dedupe_evidence(evidence)[:8]

    async def _local_evidence(self, query: str) -> list[dict[str, str]]:
        if self._retriever is None:
            return []
        try:
            hits = await self._retriever.search(query, top_k_retrieve=6, top_n_rerank=3)
        except Exception:
            return []

        evidence = []
        for hit in hits:
            source = hit.get("metadata", {}).get("source", "")
            if source in GENERATED_SOURCES:
                continue
            evidence.append(
                {
                    "kind": "local_note",
                    "title": source or "local_note",
                    "url": "",
                    "note": _shorten(hit.get("text", ""), 280),
                }
            )
        return evidence

    async def _web_evidence(
        self,
        category: dict[str, Any],
        question: dict[str, Any],
    ) -> list[dict[str, str]]:
        from nanobot.agent.tools.web import WebFetchTool, WebSearchTool

        try:
            search_tool = self._search_tool or WebSearchTool()
            search_raw = await search_tool.execute(_build_search_query(category, question), count=4)
        except Exception:
            return []

        evidence = []
        fetch_tool = self._fetch_tool or WebFetchTool()
        for item in _parse_search_results(search_raw)[:3]:
            try:
                fetched_raw = await fetch_tool.execute(item["url"], maxChars=3500)
                payload = json.loads(fetched_raw)
            except Exception:
                payload = {}

            note = _extract_fetch_text(payload) or item.get("snippet", "")
            evidence.append(
                {
                    "kind": "web_page",
                    "title": item["title"],
                    "url": payload.get("finalUrl") or item["url"],
                    "note": _shorten(note, 320),
                }
            )
        return evidence

    async def _draft_card(self, question: dict[str, Any], evidence: list[dict[str, str]]) -> dict[str, Any]:
        if not any(item.get("kind") != "interview_report" for item in evidence):
            return {}

        prompt = _build_prompt(question, evidence)
        try:
            raw = await self._llm(prompt, task="analysis", max_tokens=900)
            return parse_json_response(raw)
        except Exception:
            return {}


async def rebuild_answer_cards(
    *,
    bank: dict[str, Any] | None = None,
    recent: bool = True,
    use_web: bool = False,
    max_cards: int | None = 20,
    output_dir: Path = OUTPUT_DIR,
    retriever: Any | None = None,
    search_tool: Any | None = None,
    fetch_tool: Any | None = None,
    llm: Any = call_text_async,
) -> dict[str, Any]:
    bank = bank or load_question_bank(recent=recent)
    bundle = await AnswerCardBuilder(
        retriever=retriever,
        search_tool=search_tool,
        fetch_tool=fetch_tool,
        llm=llm,
    ).build(bank, use_web=use_web, max_cards=max_cards)

    json_path, markdown_path = _paths_for(output_dir, recent=recent)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_answer_cards_markdown(bundle), encoding="utf-8")

    return {
        "cards": sum(len(category["cards"]) for category in bundle["categories"]),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "use_web": use_web,
        "recent": recent,
    }


def load_answer_cards(path: Path | None = None, *, recent: bool = True) -> dict[str, Any]:
    json_path, _ = _paths_for(OUTPUT_DIR, recent=recent)
    target = path or json_path
    return json.loads(target.read_text(encoding="utf-8"))


def load_answer_cards_or_empty(path: Path | None = None, *, recent: bool = True) -> dict[str, Any]:
    try:
        return load_answer_cards(path=path, recent=recent)
    except Exception:
        return {"generated_at": "", "categories": []}


def build_answer_card_index(bundle: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for category in bundle.get("categories", []):
        default_category = str(category.get("name", ""))
        for card in category.get("cards", []):
            normalized = _normalize_question_key(card.get("question", ""))
            if not normalized:
                continue
            key = (str(card.get("category", default_category)), normalized)
            index[key] = card
    return index


def find_answer_card(
    question: str,
    *,
    category: str = "",
    bundle: dict[str, Any] | None = None,
    index: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    normalized = _normalize_question_key(question)
    if not normalized:
        return None

    index = index or build_answer_card_index(bundle or {})
    exact = index.get((category, normalized))
    if exact is not None:
        return exact

    best_card = None
    best_score = 0.0
    for (card_category, key), card in index.items():
        if category and card_category and card_category != category:
            continue
        score = SequenceMatcher(None, normalized, key).ratio()
        if normalized in key or key in normalized:
            score += 0.1
        if score > best_score:
            best_score = score
            best_card = card
    return best_card if best_score >= 0.78 else None


def render_answer_cards_markdown(bundle: dict[str, Any]) -> str:
    lines = [
        "---",
        "doc_type: agent_answer_cards",
        "---",
        "",
        "# Agent Interview Answer Cards",
        "",
        f"- generated_at: {bundle['generated_at']}",
        f"- source_bank_generated_at: {bundle.get('source_bank_generated_at', '')}",
        "",
    ]

    for category in bundle.get("categories", []):
        lines.append(f"## {category['label']}")
        lines.append("")
        for item in category.get("cards", []):
            lines.append(f"### {item['question']}")
            lines.append("")
            lines.append(f"- stage: {item['stage']}")
            lines.append(f"- status: {item['status']}")
            lines.append(f"- answer: {item['answer']}")
            if item.get("follow_ups"):
                lines.append(f"- follow-ups: {' / '.join(item['follow_ups'])}")
            lines.append("- pitfalls:")
            for pitfall in item.get("pitfalls", []):
                lines.append(f"  - {pitfall}")
            lines.append("- evidence:")
            for evidence in item.get("evidence", []):
                label = f"[{evidence['kind']}] {evidence['title']}"
                url = f" ({evidence['url']})" if evidence.get("url") else ""
                note = f": {evidence['note']}" if evidence.get("note") else ""
                lines.append(f"  - {label}{url}{note}")
            lines.append("")

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build answer cards from the interview question bank.")
    parser.add_argument("--all-history", action="store_true", help="Use the full bank instead of the recent bank.")
    parser.add_argument("--with-web", action="store_true", help="Supplement local evidence with web search and fetch.")
    parser.add_argument("--max-cards", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="Print result as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = asyncio.run(
        rebuild_answer_cards(
            recent=not args.all_history,
            use_web=args.with_web,
            max_cards=args.max_cards,
        )
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Built {result['cards']} answer cards -> {result['markdown_path']}")
    return 0


def _build_prompt(question: dict[str, Any], evidence: list[dict[str, str]]) -> str:
    evidence_lines = []
    for index, item in enumerate(evidence, 1):
        title = item["title"]
        url = item["url"] or "local"
        note = item["note"]
        evidence_lines.append(f"[{index}] {item['kind']} | {title} | {url}\n{note}")

    follow_ups = question.get("follow_ups") or []
    return f"""
你在整理 AI Agent 面试题簇的答案卡。

要求：
1. `answer` 要适合面试口述，先给核心结论，再补做法、取舍和可追问点，控制在 120 到 220 字。
2. `pitfalls` 给 3 到 5 条，写常见失分点，尤其是答得太空、只讲概念不讲权衡、缺少项目证据这类问题。
3. `supporting_indexes` 只返回最关键的 2 到 4 个证据编号，必须来自给定 evidence。
4. 不要编造来源，不要输出 markdown，只返回 JSON。

返回格式：
{{"answer":"", "pitfalls":[""], "supporting_indexes":[1, 2]}}

题目：{question["question"]}
追问：{" / ".join(follow_ups) if follow_ups else "无"}

evidence:
{chr(10).join(evidence_lines)}
""".strip()


def _build_search_query(category: dict[str, Any], question: dict[str, Any]) -> str:
    label = category["label"]
    base = question["question"]
    if label in base:
        return f"{base} 教程 原理 实现"
    return f"{base} {label} 教程 原理 实现"


def _paths_for(output_dir: Path, *, recent: bool) -> tuple[Path, Path]:
    if recent:
        return output_dir / RECENT_ANSWER_CARDS_JSON_PATH.name, output_dir / RECENT_ANSWER_CARDS_MARKDOWN_PATH.name
    return output_dir / ANSWER_CARDS_JSON_PATH.name, output_dir / ANSWER_CARDS_MARKDOWN_PATH.name


def _report_evidence(question: dict[str, Any]) -> list[dict[str, str]]:
    evidence = []
    for source in question.get("sources", [])[:3]:
        evidence.append(
            {
                "kind": "interview_report",
                "title": source.get("title", ""),
                "url": source.get("source_url", ""),
                "note": "该题出现在这篇面经中，可用于判断题目的真实面试语境。",
            }
        )
    return evidence


def _parse_search_results(text: str) -> list[dict[str, str]]:
    items = []
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    index = 0

    while index < len(lines):
        match = SEARCH_RESULT_LINE.match(lines[index].strip())
        if not match:
            index += 1
            continue

        title = match.group(1).strip()
        url = lines[index + 1].strip() if index + 1 < len(lines) else ""
        snippet_parts = []
        index += 2

        while index < len(lines) and not SEARCH_RESULT_LINE.match(lines[index].strip()):
            snippet_parts.append(lines[index].strip())
            index += 1

        items.append({"title": title, "url": url, "snippet": " ".join(snippet_parts).strip()})

    return [item for item in items if item["url"].startswith("http")]


def _extract_fetch_text(payload: dict[str, Any]) -> str:
    text = str(payload.get("text") or "").strip()
    if not text:
        return ""
    if text.startswith(UNTRUSTED_BANNER):
        text = text.split("\n", 2)[-1]
    return text.strip()


def _dedupe_evidence(items: list[dict[str, str]]) -> list[dict[str, str]]:
    result = []
    seen = set()
    for item in items:
        key = (
            item.get("kind", ""),
            item.get("url", ""),
            item.get("title", ""),
            item.get("note", "")[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _shorten(text: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def _normalize_question_key(text: str) -> str:
    value = str(text or "").strip().lower()
    value = value.replace("llm", "大模型")
    value = value.replace("multi-agent", "多agent")
    value = value.replace("single-agent", "单agent")
    return re.sub(r"[\W_]+", "", value)


def _normalize_indexes(value: Any, max_size: int) -> list[int]:
    result = []
    for item in value or []:
        try:
            index = int(item)
        except Exception:
            continue
        if 1 <= index <= max_size and index not in result:
            result.append(index)
    return result


def _normalize_lines(value: Any) -> list[str]:
    lines = []
    for item in value or []:
        text = str(item).strip()
        if text and text not in lines:
            lines.append(text)
    return lines[:5]


def _fallback_answer(question: dict[str, Any]) -> str:
    follow_ups = question.get("follow_ups") or []
    suffix = f"；同时准备展开 {follow_ups[0]}" if follow_ups else ""
    return f"证据还不够完整，回答时先给出核心定义，再讲实现流程、关键取舍和线上指标{suffix}。"


def _fallback_pitfalls(question: dict[str, Any]) -> list[str]:
    follow_ups = question.get("follow_ups") or []
    pitfalls = [
        "只背概念，不结合自己的项目链路和权衡。",
        "只讲理想方案，不讲失败案例、边界条件和降级策略。",
        "没有可量化指标，无法说明方案为什么有效。",
    ]
    if follow_ups:
        pitfalls.append(f"没有提前准备追问，例如：{follow_ups[0]}。")
    return pitfalls


if __name__ == "__main__":
    raise SystemExit(main())
