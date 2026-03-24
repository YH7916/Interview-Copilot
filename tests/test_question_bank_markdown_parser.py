from copilot.knowledge.question_bank import _extract_question_lines


def test_extract_question_lines_accepts_bullets():
    text = """
## 题目
- 项目介绍
- RAG 召回链路怎么设计？
- 多 Agent 协作怎么做？

## 原文整理
略
"""

    questions = _extract_question_lines(text)

    assert questions == [
        "项目介绍",
        "RAG 召回链路怎么设计？",
        "多 Agent 协作怎么做？",
    ]


def test_extract_question_lines_skips_placeholder():
    text = """
## 题目
- 未提取到明确题目，建议查看原文整理。
- 真正的问题是什么？
"""

    questions = _extract_question_lines(text)

    assert questions == ["真正的问题是什么？"]
