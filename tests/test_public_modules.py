from __future__ import annotations


def test_public_nowcoder_module_exports_ingestor() -> None:
    from copilot.sources.nowcoder import NowcoderInterviewIngestor

    assert NowcoderInterviewIngestor is not None


def test_public_knowledge_modules_export_core_types() -> None:
    from copilot.knowledge.answer_cards import AnswerCardBuilder
    from copilot.knowledge.question_bank import CATEGORY_LABELS
    from copilot.knowledge.retrieval import HybridRetriever

    assert AnswerCardBuilder is not None
    assert "opening" in CATEGORY_LABELS
    assert HybridRetriever is not None


def test_public_interview_modules_export_core_types() -> None:
    from copilot.interview.evaluation import SCORE_DIMENSIONS, evaluate_answer
    from copilot.interview.orchestrator import InterviewRunner

    assert "accuracy" in SCORE_DIMENSIONS
    assert evaluate_answer is not None
    assert InterviewRunner is not None


def test_workflows_export_digest_and_agent_layout() -> None:
    from copilot.workflows import (
        build_agent_workflow,
        build_daily_digest,
        render_daily_digest_markdown,
        run_agent_workflow,
    )

    assert callable(build_daily_digest)
    assert callable(render_daily_digest_markdown)
    assert callable(run_agent_workflow)
    assert len(build_agent_workflow()) == 3
