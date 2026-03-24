from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from copilot.workflows import build_daily_digest, render_daily_digest_markdown, run_agent_workflow

TEST_ROOT = Path(__file__).resolve().parents[1] / "data" / "_test_workspace" / "workflows"


def setup_function() -> None:
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)


def teardown_function() -> None:
    shutil.rmtree(TEST_ROOT, ignore_errors=True)


def test_build_daily_digest_renders_formal_sections() -> None:
    report_dir = TEST_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "byte-agent.md"
    report_path.write_text(
        "\n".join(
            [
                "---",
                'source_updated_at: "2026-03-18T10:00:00"',
                "source_like_count: 15",
                "source_comment_count: 6",
                "source_view_count: 900",
                "---",
                "",
                "# 字节 Agent 一面面经",
            ]
        ),
        encoding="utf-8",
    )

    report_index_path = TEST_ROOT / "report_index.json"
    report_index_path.write_text(
        json.dumps(
            [
                {
                    "title": "字节 Agent 一面面经",
                    "source_url": "https://example.com/byte-agent",
                    "source_path": str(report_path),
                    "captured_at": "2026-03-19T09:00:00",
                    "questions": ["什么是 ReAct？", "RAG 怎么做召回？"],
                },
                {
                    "title": "旧资料",
                    "source_url": "https://example.com/old",
                    "source_path": str(report_path),
                    "captured_at": "2026-03-01T09:00:00",
                    "questions": ["old"],
                },
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    recent_bank_path = TEST_ROOT / "recent_question_bank.json"
    recent_bank_path.write_text(
        json.dumps(
            {
                "categories": [
                    {
                        "name": "agent_architecture",
                        "label": "Agent 架构",
                        "clusters": [
                            {
                                "question": "什么是 ReAct？",
                                "source_count": 3,
                                "latest_source_at": "2026-03-19T09:00:00",
                            }
                        ],
                    },
                    {
                        "name": "rag_retrieval",
                        "label": "RAG 检索",
                        "questions": [
                            {
                                "question": "RAG 怎么做召回？",
                                "source_count": 2,
                                "latest_source_at": "2026-03-19T08:00:00",
                            }
                        ],
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    digest = build_daily_digest(
        days=7,
        report_index_path=report_index_path,
        recent_bank_path=recent_bank_path,
    )
    markdown = render_daily_digest_markdown(digest)

    assert digest["summary"]["fresh_reports"] == 1
    assert digest["summary"]["fresh_questions"] == 2
    assert digest["summary"]["hottest_company"] == "字节"
    assert digest["top_reports"][0]["round"] == "一面"
    assert digest["company_breakdown"][0]["count"] == 1
    assert digest["representative_questions"][0]["question"] == "什么是 ReAct？"
    assert "## Summary" in markdown
    assert "## Top Reports" in markdown
    assert "字节 Agent 一面面经" in markdown


@pytest.mark.asyncio
async def test_run_agent_workflow_executes_real_stage_chain() -> None:
    calls = []

    async def _fake_collect(**kwargs):
        calls.append(("collect", kwargs))
        return {"written": ["a.md"], "dry_run": kwargs["dry_run"]}

    async def _fake_rebuild(**kwargs):
        calls.append(("rebuild", kwargs))
        return {"question_bank_generated_at": "2026-03-19T09:00:00"}

    def _fake_digest(*, days: int) -> dict:
        calls.append(("digest", {"days": days}))
        return {
            "summary": {"fresh_reports": 1},
            "top_reports": [{"title": "字节 Agent 一面面经"}],
            "company_breakdown": [],
            "category_breakdown": [],
            "representative_questions": [],
        }

    result = await run_agent_workflow(
        collect_fn=_fake_collect,
        rebuild_fn=_fake_rebuild,
        digest_fn=_fake_digest,
        updated_within_days=5,
        count_per_query=9,
        max_reports=12,
        fetch_timeout=7,
        with_web=True,
        max_cards=4,
        dry_run=False,
    )

    assert [stage["role"] for stage in result["stages"]] == ["scout", "curator", "reporter"]
    assert result["digest"]["summary"]["fresh_reports"] == 1
    assert calls[0][0] == "collect"
    assert calls[0][1]["count_per_query"] == 9
    assert calls[0][1]["max_reports"] == 12
    assert calls[1][0] == "rebuild"
    assert calls[1][1]["with_web"] is True
    assert calls[1][1]["max_cards"] == 4
    assert calls[2] == ("digest", {"days": 5})


@pytest.mark.asyncio
async def test_run_agent_workflow_skips_curation_in_dry_run() -> None:
    calls = []

    async def _fake_collect(**kwargs):
        calls.append(("collect", kwargs))
        return {"written": []}

    async def _fake_rebuild(**kwargs):
        calls.append(("rebuild", kwargs))
        return {"unexpected": True}

    def _fake_digest(*, days: int) -> dict:
        calls.append(("digest", {"days": days}))
        return {
            "summary": {"fresh_reports": 0},
            "top_reports": [],
            "company_breakdown": [],
            "category_breakdown": [],
            "representative_questions": [],
        }

    result = await run_agent_workflow(
        collect_fn=_fake_collect,
        rebuild_fn=_fake_rebuild,
        digest_fn=_fake_digest,
        dry_run=True,
    )

    assert result["stages"][1]["result"] == {"status": "skipped", "reason": "dry_run"}
    assert [name for name, _ in calls] == ["collect", "digest"]
