from copilot.interview.evaluation import build_drill_plan, render_review_summary_with_drill


def test_build_drill_plan_prioritizes_weaker_answers():
    results = [
        {
            "question": "Explain retrieval.",
            "overall_score": 4.2,
            "accuracy_score": 4,
            "clarity_score": 4,
            "depth_score": 4,
            "evidence_score": 4,
            "reason": "Solid answer.",
        },
        {
            "question": "Explain agent memory.",
            "overall_score": 2.4,
            "accuracy_score": 3,
            "clarity_score": 3,
            "depth_score": 2,
            "evidence_score": 1,
            "reason": "Missing concrete implementation detail.",
        },
    ]

    plan = build_drill_plan(results)

    assert plan
    assert plan[0]["question"] == "Explain agent memory."
    assert "implementation detail" in plan[0]["focus"]


def test_render_review_summary_with_drill_section():
    summary = render_review_summary_with_drill(
        [
            {
                "question": "Explain ReAct.",
                "overall_score": 3.0,
                "accuracy_score": 3,
                "clarity_score": 3,
                "depth_score": 2,
                "evidence_score": 2,
                "reason": "Too shallow.",
            }
        ],
        drill_plan=[
            {
                "question": "Explain ReAct.",
                "focus": "go one layer deeper on design trade-offs and edge cases",
                "reason": "Too shallow.",
            }
        ],
    )

    assert "# Interview Review" in summary
    assert "## Next Drill" in summary
    assert "Explain ReAct." in summary


def test_render_interview_review_prefers_runner_summary(monkeypatch):
    monkeypatch.setattr(
        "copilot.app.review_interview_messages",
        lambda messages, persist=False: {
            "count": 1,
            "results": [
                {
                    "question": "Explain ReAct.",
                    "overall_score": 3.0,
                    "accuracy_score": 3,
                    "clarity_score": 3,
                    "depth_score": 2,
                    "evidence_score": 2,
                    "reason": "Too shallow.",
                }
            ],
            "drill_plan": [
                {
                    "question": "Explain ReAct.",
                    "focus": "go one layer deeper on design trade-offs and edge cases",
                    "reason": "Too shallow.",
                }
            ],
            "summary": "# Interview Review\n\n## Next Drill\n\n1. Explain ReAct.\n   - focus: go one layer deeper on design trade-offs and edge cases\n   - reason: Too shallow.\n",
        },
    )

    from copilot.app import render_interview_review

    result = render_interview_review([{"role": "assistant", "content": "Explain ReAct."}])

    assert "## Next Drill" in result
