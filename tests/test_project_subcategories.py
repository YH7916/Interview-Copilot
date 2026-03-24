from copilot.interview.planner import InterviewPlanner
from copilot.interview.session import InterviewSession
from copilot.knowledge.question_bank import build_question_bank, classify_question


def test_classify_question_splits_project_subcategories() -> None:
    assert classify_question("你的项目业务背景和核心指标是什么") == "project_background"
    assert classify_question("项目里负责的边界是什么") == "project_scope"
    assert classify_question("数据清洗和标注是怎么做的") == "project_data"
    assert classify_question("整体架构和模块链路是怎么设计的") == "project_architecture"
    assert classify_question("你们怎么做评估和长期记忆") == "project_evaluation"
    assert classify_question("线上延迟、吞吐和部署成本怎么优化") == "project_deployment"
    assert classify_question("项目中遇到的难点和权衡是什么") == "project_challenges"


def test_interview_planner_prefers_project_subcategories() -> None:
    bank = build_question_bank(
        [
            {
                "title": "Agent 面经",
                "source_url": "https://example.com/1",
                "source_path": "a.md",
                "captured_at": "2026-03-20T10:00:00",
                "questions": [
                "自我介绍",
                "你的项目业务背景和核心指标是什么",
                "项目里负责的边界是什么",
                "数据清洗和标注是怎么做的",
                "整体架构和模块链路是怎么设计的",
                "你们怎么做评估和长期记忆",
                "线上延迟、吞吐和部署成本怎么优化",
                "项目中遇到的难点和权衡是什么",
                "RAG怎么评估",
            ],
        }
    ]
)

    plan = InterviewPlanner(bank=bank).plan(
        InterviewSession(user_id="u1", focus_topics=["architecture", "evaluation", "deployment", "ownership", "data"]),
        max_questions=7,
    )

    categories = {item.category for item in plan}
    assert "project_data" in categories
    assert "project_architecture" in categories
    assert "project_evaluation" in categories
    assert "project_deployment" in categories


def test_build_question_bank_skips_project_placeholder_clusters() -> None:
    bank = build_question_bank(
        [
            {
                "title": "Agent 面经",
                "source_url": "https://example.com/1",
                "source_path": "a.md",
                "captured_at": "2026-03-20T10:00:00",
                "questions": [
                    "项目拷打",
                    "项目深挖",
                    "项目里负责的边界是什么",
                ],
            }
        ]
    )

    categories = {item["name"]: item for item in bank["categories"]}
    scope_questions = categories["project_scope"]["questions"]
    assert any(item["question"] == "项目里负责的边界是什么" for item in scope_questions)
    assert "project" not in categories or not any(
        item["question"] in {"项目拷打", "项目深挖"} for item in categories["project"].get("questions", [])
    )
