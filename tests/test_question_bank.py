from __future__ import annotations

from datetime import datetime, timedelta

import copilot.interview.planner as planner_module
from copilot.interview.planner import InterviewPlanner
from copilot.interview.session import InterviewSession
from copilot.knowledge.question_bank import (
    _clean_question,
    _is_bank_candidate,
    _recent_reports,
    build_question_bank,
    classify_question,
    explode_questions,
)
from tests.interview_fixtures import (
    make_answer_card_bundle,
    make_bank,
    make_card,
    make_category,
    make_cluster,
    make_question_entry,
    make_report,
    make_source,
)

RAG_LABEL = "RAG 与检索链路"
CODING_LABEL = "算法与手撕"


def test_explode_questions_splits_compound_lines():
    line = "项目里 RAG 怎么做的，怎么设计召回排序链路，embedding 召回有什么方案？"
    questions = explode_questions(line)

    assert "项目里 RAG 怎么做的" in questions
    assert "怎么设计召回排序链路" in questions
    assert "embedding 召回有什么方案" in questions


def test_build_question_bank_merges_similar_questions():
    reports = [
        make_report("字节一面", ["单Agent还是多Agent的？", "RAG 怎么做的？"], source_url="https://example.com/1", source_path="a.md", captured_at="2026-03-19T10:00:00"),
        make_report("抖音一面", ["单 Agent 还是多 Agent？", "召回排序链路怎么设计？"], source_url="https://example.com/2", source_path="b.md", captured_at="2026-03-20T10:00:00"),
    ]

    bank = build_question_bank(reports)
    categories = {item["name"]: item for item in bank["categories"]}
    agent_questions = categories["agent_architecture"]["questions"]
    agent_clusters = categories["agent_architecture"]["clusters"]

    assert agent_questions[0]["source_count"] == 2
    assert agent_clusters[0]["source_count"] == 2


def test_rebuild_bank_prefers_real_interview_sources():
    assert _is_bank_candidate({"source_type": "nowcoder_page", "title": "字节一面"})
    assert not _is_bank_candidate({"source_type": "github_page", "title": "100 Agentic AI Interview Questions"})


def test_recent_reports_keeps_recent_slice():
    now = datetime.now()
    reports = [
        {"captured_at": (now - timedelta(days=10)).isoformat(timespec="seconds")},
        {"captured_at": (now - timedelta(days=120)).isoformat(timespec="seconds")},
    ]

    recent = _recent_reports(reports, recent_days=90)

    assert len(recent) == 1


def test_clean_question_removes_project_prefix():
    text = "实习项目1(拷打实现细节): 业务背景，项目谁主要负责的"
    assert _clean_question(text) == "项目谁主要负责的"


def test_classify_question_handles_llm_fundamentals():
    assert classify_question("目前主流的开源模型体系有哪些") == "llm_fundamentals"
    assert classify_question("为什么 BERT 选择 mask 掉 15% 这个比例的词") == "llm_fundamentals"


def test_classify_question_keeps_hand_coding_and_fundamentals_separate():
    assert classify_question("请对比 RLHF、PPO、DPO 算法的技术差异、优缺点及适用场景。") == "llm_fundamentals"
    assert classify_question("手撕：岛屿最大面积") == "coding"


def test_interview_planner_round_one_uses_fixed_flow():
    bank = build_question_bank(
        [
            make_report(
                "字节一面",
                [
                    "自我介绍",
                    "你的项目业务背景和核心指标是什么？",
                    "整体架构和模块链路是怎么设计的？",
                    "RAG 召回链路怎么设计？",
                    "Attention 为什么要用 multi-head？",
                    "岛屿最大面积",
                ],
                source_url="https://example.com/1",
                source_path="a.md",
                captured_at="2026-03-19T10:00:00",
            )
        ]
    )

    plan = InterviewPlanner(bank=bank).plan(
        InterviewSession(user_id="u1", company="字节", focus_topics=["RAG"]),
        max_questions=6,
    )

    assert [item.stage for item in plan] == [
        "opening",
        "project",
        "project",
        "project",
        "foundations",
        "coding",
    ]
    assert [item.category for item in plan] == [
        "opening",
        "project_background",
        "project_architecture",
        "rag_retrieval",
        "llm_fundamentals",
        "coding",
    ]


def test_question_bank_builds_follow_up_clusters():
    bank = build_question_bank(
        [
            make_report(
                "字节一面",
                ["项目里 RAG 怎么做的，怎么设计召回排序链路，embedding 召回有什么方案？"],
                source_url="https://example.com/1",
                source_path="a.md",
                captured_at="2026-03-19T10:00:00",
            )
        ]
    )

    rag_clusters = {item["name"]: item for item in bank["categories"]}["rag_retrieval"]["clusters"]

    assert rag_clusters[0]["question"] == "项目里 RAG 怎么做的"
    assert "怎么设计召回排序链路" in rag_clusters[0]["follow_ups"]
    assert "embedding 召回有什么方案" in rag_clusters[0]["follow_ups"]


def test_interview_planner_round_two_skips_opening():
    bank = build_question_bank(
        [
            make_report(
                "字节二面",
                [
                    "自我介绍",
                    "单Agent还是多Agent的？",
                    "RAG 召回链路怎么设计？",
                    "Attention 怎么实现的？",
                    "Python 为什么有 GIL？",
                ],
                source_url="https://example.com/2",
                source_path="b.md",
                captured_at="2026-03-19T10:00:00",
            )
        ]
    )

    plan = InterviewPlanner(bank=bank).plan(
        InterviewSession(user_id="u2", round_index=2, focus_topics=["Agent", "RAG"]),
        max_questions=4,
    )

    assert plan
    assert all(item.stage != "opening" for item in plan)


def test_interview_planner_supports_coding_first_style():
    bank = build_question_bank(
        [
            make_report(
                "算法面",
                [
                    "自我介绍",
                    "RAG 召回链路怎么设计？",
                    "Attention 怎么实现的？",
                    "岛屿最大面积",
                ],
                source_url="https://example.com/3",
                source_path="c.md",
                captured_at="2026-03-19T10:00:00",
            )
        ]
    )

    plan = InterviewPlanner(bank=bank).plan(
        InterviewSession(user_id="u2", interview_style="coding-first", focus_topics=["RAG"]),
        max_questions=4,
    )

    assert plan[0].category == "coding"
    assert any(item.category == "opening" for item in plan)


def test_interview_planner_auto_uses_no_coding_style_when_bank_has_no_coding():
    bank = build_question_bank(
        [
            make_report(
                "项目面",
                [
                    "自我介绍",
                    "你的项目业务背景和核心指标是什么？",
                    "RAG 召回链路怎么设计？",
                    "你们怎么做评估和长期记忆？",
                    "Attention 为什么要用 multi-head？",
                ],
                source_url="https://example.com/4",
                source_path="d.md",
                captured_at="2026-03-19T10:00:00",
            )
        ]
    )

    plan = InterviewPlanner(bank=bank).plan(
        InterviewSession(user_id="u3", focus_topics=["RAG"]),
        max_questions=5,
    )

    assert all(item.category != "coding" for item in plan)


def test_interview_planner_reclassifies_stale_coding_entries():
    bank = make_bank(
        make_category(
            "coding",
            CODING_LABEL,
            stage="coding",
            questions=[
                make_question_entry(
                    "请对比 RLHF、PPO、DPO 算法的技术差异、优缺点及适用场景。",
                    "coding",
                    CODING_LABEL,
                    stage="coding",
                    sources=[make_source(title="算法面", source_url="a")],
                ),
                make_question_entry(
                    "手撕：岛屿最大面积",
                    "coding",
                    CODING_LABEL,
                    stage="coding",
                    sources=[make_source(title="算法面", source_url="b", source_path="b.md")],
                ),
            ],
        )
    )

    plan = InterviewPlanner(bank=bank).plan(
        InterviewSession(user_id="u4"),
        max_questions=2,
    )

    assert [item.category for item in plan] == ["llm_fundamentals", "coding"]


def test_interview_planner_prefers_recent_questions():
    bank = make_bank(
        make_category(
            "rag_retrieval",
            RAG_LABEL,
            stage="project",
            questions=[
                make_question_entry(
                    "旧的 RAG 问题",
                    "rag_retrieval",
                    RAG_LABEL,
                    latest_source_at="2025-01-01T10:00:00",
                    sources=[make_source(title="旧面经", source_url="a", captured_at="2025-01-01T10:00:00")],
                ),
                make_question_entry(
                    "新的 RAG 问题",
                    "rag_retrieval",
                    RAG_LABEL,
                    source_count=1,
                    latest_source_at="2099-01-01T10:00:00",
                    sources=[make_source(title="新面经", source_url="b", source_path="b.md", captured_at="2099-01-01T10:00:00")],
                ),
            ],
        )
    )

    plan = InterviewPlanner(bank=bank)._pick_questions(
        categories=["rag_retrieval"],
        stage="project",
        limit=1,
        used=set(),
        company="",
        focus_categories={"rag_retrieval"},
    )

    assert plan[0].question == "新的 RAG 问题"


def test_interview_planner_falls_back_to_full_bank(monkeypatch):
    recent_bank = make_bank(source_dir="recent")
    full_bank = make_bank(
        make_category(
            "rag_retrieval",
            RAG_LABEL,
            stage="project",
            questions=[
                make_question_entry(
                    "历史 RAG 问题",
                    "rag_retrieval",
                    RAG_LABEL,
                    source_count=3,
                    latest_source_at="2025-01-01T10:00:00",
                    sources=[make_source(title="历史面经", source_url="a", captured_at="2025-01-01T10:00:00")],
                )
            ],
        ),
        source_dir="full",
    )

    monkeypatch.setattr(
        planner_module,
        "load_question_bank",
        lambda recent=False, path=None: recent_bank if recent else full_bank,
    )

    planner = InterviewPlanner(recent=True)
    plan = planner._pick_questions(
        categories=["rag_retrieval"],
        stage="project",
        limit=1,
        used=set(),
        company="",
        focus_categories={"rag_retrieval"},
    )

    assert plan[0].question == "历史 RAG 问题"


def test_interview_planner_uses_cluster_follow_ups():
    bank = make_bank(
        make_category(
            "rag_retrieval",
            RAG_LABEL,
            stage="project",
            clusters=[
                make_cluster(
                    "RAG 怎么做的",
                    "rag_retrieval",
                    RAG_LABEL,
                    follow_ups=["召回链路怎么设计", "embedding 召回有什么方案"],
                    sources=[make_source(title="字节一面")],
                )
            ],
            questions=[],
        )
    )

    plan = InterviewPlanner(bank=bank).plan(
        InterviewSession(user_id="u3", focus_topics=["RAG"]),
        max_questions=1,
    )

    assert plan[0].question == "RAG 怎么做的"
    assert "召回链路怎么设计" in plan[0].follow_ups


def test_interview_planner_attaches_answer_card(monkeypatch):
    bank = make_bank(
        make_category(
            "rag_retrieval",
            RAG_LABEL,
            stage="project",
            clusters=[
                make_cluster(
                    "RAG 怎么做的",
                    "rag_retrieval",
                    RAG_LABEL,
                    follow_ups=["召回链路怎么设计"],
                    sources=[make_source(title="字节一面")],
                )
            ],
            questions=[],
        )
    )
    cards = make_answer_card_bundle(
        make_category(
            "rag_retrieval",
            RAG_LABEL,
            stage="project",
            cards=[
                make_card(
                    "RAG 怎么做的",
                    "rag_retrieval",
                    RAG_LABEL,
                    answer="先讲召回、重排和生成。",
                    pitfalls=["不要只讲概念。"],
                )
            ],
        )
    )

    monkeypatch.setattr(planner_module, "load_answer_cards_or_empty", lambda recent=False: cards)
    plan = InterviewPlanner(bank=bank).plan(
        InterviewSession(user_id="u3", focus_topics=["RAG"]),
        max_questions=1,
    )

    assert plan[0].answer_status == "grounded"
    assert plan[0].reference_answer == "先讲召回、重排和生成。"
