from __future__ import annotations

from evals.run_eval import (
    compute_recall,
    parse_json_response,
    run_cluster_eval,
    run_ingest_eval,
    run_interview_policy_eval,
    run_review_eval,
)


def test_compute_recall_handles_expected_sources():
    recall = compute_recall({"a.md", "b.md"}, {"b.md", "c.md"})
    assert recall == 0.5
    assert compute_recall({"a.md"}, set()) is None


def test_parse_json_response_strips_code_fences():
    payload = parse_json_response(
        """```json
{"accuracy_score": 5, "faithfulness_score": 4, "reason": "ok"}
```"""
    )
    assert payload["accuracy_score"] == 5
    assert payload["faithfulness_score"] == 4


def test_run_ingest_eval_reports_accuracy():
    report = run_ingest_eval(
        [
            {"title": "字节 AI Agent 一面", "text": "1. 什么是 Agent？", "expected_keep": True},
            {"title": "春招启动", "text": "内推广告", "expected_keep": False},
        ]
    )

    assert report["suite"] == "ingest"
    assert report["summary"]["total"] == 2
    assert report["summary"]["passed"] == 2


def test_run_cluster_eval_reports_category_matches():
    report = run_cluster_eval(
        [
            {"question": "什么是 ReAct？", "expected_category": "agent_architecture"},
            {"question": "Python 为什么会有 GIL？", "expected_category": "python_system"},
        ]
    )

    assert report["suite"] == "cluster"
    assert report["summary"]["passed"] == 2


def test_run_interview_policy_eval_reports_follow_up_matches():
    report = run_interview_policy_eval(
        [
            {
                "name": "needs_follow_up",
                "answer": "先做召回。",
                "depth_score": 2,
                "evidence_score": 1,
                "overall_score": 2.0,
                "expected_follow_up": True,
                "has_follow_ups": True,
            }
        ]
    )

    assert report["suite"] == "interview_policy"
    assert report["summary"]["passed"] == 1


def test_run_review_eval_local_mode_uses_keyword_judge():
    report = run_review_eval(
        [
            {
                "question": "RAG召回链路怎么设计？",
                "expected_answer": "先讲召回和重排。",
                "actual_answer": "我会先做召回，再做重排。",
                "pitfalls": [],
                "expected_keywords": ["召回", "重排"],
                "expected_min_accuracy": 4,
            }
        ],
        judge_mode="local",
    )

    assert report["suite"] == "review"
    assert report["judge_mode"] == "local"
    assert report["summary"]["passed"] == 1
