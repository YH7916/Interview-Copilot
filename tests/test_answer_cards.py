from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from copilot.knowledge.answer_cards import AnswerCardBuilder, rebuild_answer_cards, render_answer_cards_markdown
from tests.interview_fixtures import (
    make_answer_card_bundle,
    make_bank,
    make_card,
    make_category,
    make_cluster,
    make_source,
)

AGENT_LABEL = "Agent 架构与技能"
AGENT_QUESTION = "单Agent还是多Agent的？"


class FakeRetriever:
    async def search(self, query: str, top_k_retrieve: int = 6, top_n_rerank: int = 3):
        return [
            {
                "text": "RAG 回答要覆盖召回、重排、生成和评测闭环。",
                "metadata": {"source": "rag_notes.md"},
            }
        ]


class EmptyRetriever:
    async def search(self, query: str, top_k_retrieve: int = 6, top_n_rerank: int = 3):
        return []


class FakeSearchTool:
    async def execute(self, query: str, count: int = 4, **kwargs):
        return "\n".join(
            [
                f"Results for: {query}",
                "",
                "1. Agent Tutorial",
                "   https://example.com/agent",
                "   Agent system design and tradeoffs.",
            ]
        )


class FakeFetchTool:
    async def execute(self, url: str, maxChars: int = 3500, **kwargs):
        return json.dumps(
            {
                "url": url,
                "finalUrl": url,
                "text": "# Agent Tutorial\n\nAgent answers should explain planner, tools, memory and fallback.",
            },
            ensure_ascii=False,
        )


async def fake_llm(prompt: str, task: str = "analysis", max_tokens: int = 900):
    assert "supporting_indexes" in prompt
    return json.dumps(
        {
            "answer": "先讲核心目标，再讲链路、权衡和指标。",
            "pitfalls": ["只讲概念，不讲线上权衡。", "没有项目证据。"],
            "supporting_indexes": [2, 3],
        },
        ensure_ascii=False,
    )


def _sample_bank() -> dict:
    return make_bank(
        make_category(
            "agent_architecture",
            AGENT_LABEL,
            stage="project",
            clusters=[
                make_cluster(
                    AGENT_QUESTION,
                    "agent_architecture",
                    AGENT_LABEL,
                    follow_ups=["子Agent任务是什么？"],
                    sources=[make_source(title="字节一面", source_url="https://nowcoder.com/example")],
                )
            ],
            questions=[],
        ),
        generated_at="2026-03-19T10:00:00",
    )


@pytest.mark.asyncio
async def test_answer_card_builder_merges_local_and_web_evidence():
    bundle = await AnswerCardBuilder(
        retriever=FakeRetriever(),
        search_tool=FakeSearchTool(),
        fetch_tool=FakeFetchTool(),
        llm=fake_llm,
    ).build(_sample_bank(), use_web=True, max_cards=1)

    card = bundle["categories"][0]["cards"][0]

    assert card["answer"] == "先讲核心目标，再讲链路、权衡和指标。"
    assert len(card["evidence"]) == 2
    assert {item["kind"] for item in card["evidence"]} == {"local_note", "web_page"}


@pytest.mark.asyncio
async def test_answer_card_builder_falls_back_without_evidence():
    bundle = await AnswerCardBuilder(retriever=EmptyRetriever(), llm=fake_llm).build(_sample_bank(), use_web=False, max_cards=1)

    card = bundle["categories"][0]["cards"][0]

    assert card["status"] == "draft"
    assert card["evidence"][0]["kind"] == "interview_report"


@pytest.mark.asyncio
async def test_rebuild_answer_cards_writes_recent_outputs():
    output_dir = Path("D:/Projects/Interview-Copilot/data/_pytest_answer_cards")
    shutil.rmtree(output_dir, ignore_errors=True)

    result = await rebuild_answer_cards(
        bank=_sample_bank(),
        recent=True,
        use_web=False,
        max_cards=1,
        output_dir=output_dir,
        retriever=FakeRetriever(),
        llm=fake_llm,
    )

    assert result["cards"] == 1
    assert (output_dir / "recent_answer_cards.json").exists()
    assert (output_dir / "recent_answer_cards.md").exists()
    shutil.rmtree(output_dir, ignore_errors=True)


def test_render_answer_cards_markdown_contains_sections():
    markdown = render_answer_cards_markdown(
        make_answer_card_bundle(
            make_category(
                "agent_architecture",
                AGENT_LABEL,
                stage="project",
                cards=[
                    make_card(
                        AGENT_QUESTION,
                        "agent_architecture",
                        AGENT_LABEL,
                        answer="先说结论。",
                        follow_ups=["子Agent任务是什么？"],
                        pitfalls=["只讲概念。"],
                        evidence=[{"kind": "web_page", "title": "Agent Tutorial", "url": "https://example.com", "note": "tradeoffs"}],
                    )
                ],
            ),
            generated_at="2026-03-19T10:00:00",
            source_bank_generated_at="2026-03-19T09:00:00",
        )
    )

    assert "# Agent Interview Answer Cards" in markdown
    assert f"### {AGENT_QUESTION}" in markdown
    assert "[web_page] Agent Tutorial" in markdown
