from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from copilot.sources.nowcoder import (
    DEFAULT_QUERIES,
    FetchedPage,
    NowcoderInterviewIngestor,
    SearchHit,
    analyze_page,
    build_page_from_search_hit,
    build_canonical_id,
    build_hot_post_url,
    build_search_result_url,
    extract_home_recommend_records,
    extract_home_tab_records,
    extract_recommend_cursor_score,
    extract_search_api_records,
    extract_search_hits,
    extract_questions,
    extract_page_text,
    extract_page_metric,
    extract_page_updated_at,
    extract_discuss_hits,
    extract_recent_hits,
    is_hot_post_candidate,
    is_agent_relevant,
    is_recent_page,
    is_recent_timestamp,
    is_preferred_page,
    is_material_path,
    is_nowcoder_candidate,
    is_paid_page,
    normalize_search_query,
    normalize_question_line,
    is_targeted_query,
    page_priority,
    parse_search_record,
    query_target_hits,
    render_markdown,
    score_hit,
    select_queries,
    select_runtime_queries,
)


def test_nowcoder_candidate_accepts_discuss_page():
    hit = SearchHit(
        query="q",
        title="Agent 面经总结",
        url="https://www.nowcoder.com/discuss/123",
        snippet="LLM Agent 面试题整理",
    )
    assert is_nowcoder_candidate(hit, hit.url) is True


def test_nowcoder_candidate_rejects_topic_page():
    hit = SearchHit(
        query="q",
        title="Agent 面经",
        url="https://www.nowcoder.com/creation/subject/abc",
        snippet="Agent 面试题",
    )
    assert is_nowcoder_candidate(hit, hit.url) is False


def test_material_path_only_accepts_discuss_and_detail():
    assert is_material_path("/discuss/123") is True
    assert is_material_path("/feed/main/detail/123") is True
    assert is_material_path("/creation/subject/123") is False


def test_score_prefers_discuss_pages():
    discuss = SearchHit(
        query="q",
        title="Agent 面经总结",
        url="https://www.nowcoder.com/discuss/123",
        snippet="LLM Agent 面试题整理",
    )
    detail = SearchHit(
        query="q",
        title="Agent 面经总结",
        url="https://www.nowcoder.com/feed/main/detail/123",
        snippet="LLM Agent 面试题整理",
    )
    assert score_hit(discuss) > score_hit(detail)


def test_score_prefers_company_round_share_over_summary():
    fresh = SearchHit(
        query="q",
        title="字节 Agent 一面 面经",
        url="https://www.nowcoder.com/discuss/123",
        snippet="真实面试回忆",
    )
    aggregate = SearchHit(
        query="q",
        title="Agent 面经总结",
        url="https://www.nowcoder.com/discuss/124",
        snippet="题库汇总",
    )
    assert score_hit(fresh) > score_hit(aggregate)


def test_score_prefers_hot_posts_when_other_signals_are_similar():
    normal = SearchHit(
        query="q",
        title="Agent 一面 面经",
        url="https://www.nowcoder.com/discuss/123",
        snippet="真实面试回忆",
        hot_value=0,
    )
    hot = SearchHit(
        query="q",
        title="Agent 一面 面经",
        url="https://www.nowcoder.com/discuss/124",
        snippet="真实面试回忆",
        hot_value=8000,
    )
    assert score_hit(hot) > score_hit(normal)


def test_is_hot_post_candidate_accepts_round_or_interview_only_title():
    assert is_hot_post_candidate("字节一面面经")
    assert is_hot_post_candidate("28届实习拷打，一场面试，23个Agent问题")
    assert not is_hot_post_candidate("春招启动，欢迎投递")


def test_build_hot_post_url_uses_discuss_id():
    assert build_hot_post_url({"id": "864153617182355456"}) == "https://www.nowcoder.com/discuss/864153617182355456"


def test_build_search_result_url_uses_content_id():
    assert (
        build_search_result_url({"contentId": "863581806069673984"}, {"title": "CVTE AI Agent开发 一面"})
        == "https://www.nowcoder.com/discuss/863581806069673984"
    )


def test_build_search_result_url_prefers_entity_id_for_discuss_posts():
    assert (
        build_search_result_url(
            {"contentId": "863860210215985152", "entityId": 1619736},
            {"id": "863860210215985152", "entityId": 1619736, "title": "腾讯后台开发一面"},
        )
        == "https://www.nowcoder.com/discuss/1619736"
    )


def test_build_search_result_url_uses_feed_detail_for_moment_data():
    assert (
        build_search_result_url(
            {"momentData": {"id": 2813436}},
            {"id": 2813436, "title": "货拉拉Agent开发日常实习一面（45min）"},
        )
        == "https://www.nowcoder.com/feed/main/detail/2813436"
    )


def test_default_queries_cover_recent_share_variants():
    assert "Agent 一面 面经" in DEFAULT_QUERIES
    assert "AI Agent 面经" in DEFAULT_QUERIES
    assert "春招 Agent 一面" in DEFAULT_QUERIES
    assert "一面 面经" in DEFAULT_QUERIES
    assert "大模型 面经" in DEFAULT_QUERIES
    assert "字节 Agent 面经" in DEFAULT_QUERIES
    assert "function calling 面经" in DEFAULT_QUERIES
    assert len(DEFAULT_QUERIES) >= 40


def test_extract_questions_keeps_numbered_items():
    text = "1. 什么是 agent？\n2. RAG 和 agent 怎么结合？\n结尾总结"
    questions = extract_questions(text, limit=5)
    assert questions[:2] == ["什么是 agent？", "RAG 和 agent 怎么结合？"]


def test_extract_questions_skips_answer_lines():
    text = (
        "1. 为什么要用 RAG\n"
        "核心原因：降低幻觉并提升时效性\n"
        "2. LLM 复读机问题\n"
        "15%是经验上信息破坏和训练信号之间的平衡点\n"
    )
    questions = extract_questions(text, limit=5)
    assert questions == ["为什么要用 RAG", "LLM 复读机问题"]


def test_extract_questions_splits_inline_numbered_items():
    text = "攒人品中，祝大家都能拿到满意的Offer！ 1.项目介绍 2.知识蒸馏，软硬标签怎么提，loss怎么算 3.lora和qlora 4.deepspeed"
    questions = extract_questions(text, limit=5)
    assert questions[:4] == [
        "项目介绍",
        "知识蒸馏，软硬标签怎么提，loss怎么算",
        "lora和qlora",
        "deepspeed",
    ]


def test_normalize_question_line_strips_nested_numbering():
    assert normalize_question_line("1. 2. 3. RAG 怎么评测") == "RAG 怎么评测"


def test_analyze_page_rejects_non_agent_interview():
    page = FetchedPage(
        hit=SearchHit(query="q", title="字节反作弊产品-一面面经", url="https://www.nowcoder.com/discuss/10"),
        canonical_url="https://www.nowcoder.com/discuss/10",
        title="字节反作弊产品-一面面经",
        text="自我介绍\n项目深挖\n风控策略的准确率无法100%，误拦正常用户时如何评估并平衡用户体验",
        updated_at="2099-01-01T00:00:00",
    )
    assert analyze_page(page) is None


def test_is_agent_relevant_rejects_title_only_agent_signal():
    text = "一面面经\n1. 自我介绍\n2. MySQL 索引怎么设计\n3. Redis 持久化"
    assert not is_agent_relevant(text.lower(), title="Agent 一面 面经")


def test_is_preferred_page_rejects_generic_role_post_without_agent_signal():
    page = FetchedPage(
        hit=SearchHit(query="q", title="小红书前端一二面OC", url="https://www.nowcoder.com/discuss/11"),
        canonical_url="https://www.nowcoder.com/discuss/11",
        title="小红书前端一二面OC",
        text="自我介绍\nMySQL 慢查询怎么处理\n索引如何设计",
        updated_at="2099-01-01T00:00:00",
        like_count=99,
        comment_count=20,
        view_count=5000,
    )
    assert not is_preferred_page(page)


def test_is_preferred_page_requires_actual_interview_signal():
    page = FetchedPage(
        hit=SearchHit(query="q", title="阿里Agent爱问", url="https://www.nowcoder.com/discuss/12"),
        canonical_url="https://www.nowcoder.com/discuss/12",
        title="阿里Agent爱问",
        text="Agent 平台更新了新能力，欢迎大家体验 function calling 和 workflow。",
        updated_at="2099-01-01T00:00:00",
    )
    assert not is_preferred_page(page)


def test_is_preferred_page_accepts_title_agent_with_weak_body_signals():
    page = FetchedPage(
        hit=SearchHit(query="q", title="比心 Agent 一面", url="https://www.nowcoder.com/discuss/13"),
        canonical_url="https://www.nowcoder.com/discuss/13",
        title="比心 Agent 一面",
        text="1. prompt 怎么设计\n2. workflow 如何拆分\n3. 反问",
        updated_at="2099-01-01T00:00:00",
    )
    assert is_preferred_page(page)


def test_is_agent_relevant_accepts_post_with_round_only_title_and_agent_body():
    text = "一面面经\nRAG 检索怎么做\nembedding 和 rerank 的作用是什么"
    assert is_agent_relevant(text.lower(), title="一面面经")


def test_is_agent_relevant_accepts_large_model_application_body():
    text = "一面面经\n大模型应用怎么落地\nembedding 和 rerank 如何配合\nworkflow 如何拆分"
    assert is_agent_relevant(text.lower(), title="大模型应用 一面")


def test_extract_page_text_falls_back_to_embedded_rich_text():
    body = """
    <html>
      <head><title>标题_牛客网</title></head>
      <body>
        <script>
          __INITIAL_STATE__={"prefetchData":{"2":{"ssrCommonData":{"contentData":{
            "title":"字节 Agent 一面 面经",
            "richText":"<p>1. 为什么选择做Agent项目？</p><p>2. 如何设计多Agent协作？</p>"
          }}}}};
        </script>
        <div>empty</div>
      </body>
    </html>
    """

    class _Response:
        text = body

    content = extract_page_text(_Response(), max_chars=1000)
    assert "为什么选择做Agent项目" in content
    assert "如何设计多Agent协作" in content


def test_extract_page_updated_at_reads_embedded_timestamp():
    body = """
    <script>
      __INITIAL_STATE__={"prefetchData":{"2":{"ssrCommonData":{"contentData":{
        "title":"字节 Agent 一面 面经",
        "richText":"<p>1. 为什么选择做Agent项目？</p>",
        "showTime":1772269650000
      }}}}};
    </script>
    """
    assert extract_page_updated_at(body) == "2026-02-28T17:07:30"


def test_extract_page_metric_reads_embedded_frequency_data():
    body = """
    <script>
      __INITIAL_STATE__={"prefetchData":{"2":{"ssrCommonData":{"contentData":{
        "title":"字节 Agent 一面 面经",
        "richText":"<p>1. 为什么选择做Agent项目？</p>",
        "frequencyData":{"likeCnt":88,"commentCnt":13,"viewCnt":4096}
      }}}}};
    </script>
    """
    assert extract_page_metric(body, "likeCnt") == 88
    assert extract_page_metric(body, "commentCnt") == 13
    assert extract_page_metric(body, "viewCnt") == 4096


def test_is_targeted_query_accepts_large_model_interview_query():
    assert is_targeted_query("大模型 面经")


def test_query_target_hits_prefers_generic_queries():
    assert query_target_hits("面经", 50) > query_target_hits("AI Agent 一面 面经", 50)


def test_select_runtime_queries_trims_default_query_pool():
    assert len(select_runtime_queries(None, max_reports=200)) <= 24
    assert len(select_runtime_queries(None, max_reports=50)) <= 40


def test_is_recent_page_filters_stale_content():
    recent = FetchedPage(
        hit=SearchHit(query="q", title="Agent 面经", url="https://www.nowcoder.com/discuss/1"),
        canonical_url="https://www.nowcoder.com/discuss/1",
        title="Agent 面经",
        text="1. 什么是 agent？",
        updated_at="2099-01-01T00:00:00",
    )
    stale = FetchedPage(
        hit=SearchHit(query="q", title="Agent 面经", url="https://www.nowcoder.com/discuss/2"),
        canonical_url="https://www.nowcoder.com/discuss/2",
        title="Agent 面经",
        text="1. 什么是 agent？",
        updated_at="2020-01-01T00:00:00",
    )
    assert is_recent_page(recent, within_days=30)
    assert not is_recent_page(stale, within_days=30)


def test_is_recent_page_rejects_missing_timestamp_when_filter_enabled():
    page = FetchedPage(
        hit=SearchHit(query="q", title="Agent 面经", url="https://www.nowcoder.com/discuss/3"),
        canonical_url="https://www.nowcoder.com/discuss/3",
        title="Agent 面经",
        text="1. 什么是 agent？",
    )
    assert not is_recent_page(page, within_days=30)


def test_is_recent_timestamp_rejects_stale_time():
    assert is_recent_timestamp("2099-01-01T00:00:00", within_days=7)
    assert not is_recent_timestamp("2020-01-01T00:00:00", within_days=7)


def test_is_preferred_page_rejects_advertisement_post():
    page = FetchedPage(
        hit=SearchHit(query="q", title="2026春招启动", url="https://www.nowcoder.com/discuss/4"),
        canonical_url="https://www.nowcoder.com/discuss/4",
        title="2026春招启动",
        text="AI Agent 招聘广告",
        updated_at="2099-01-01T00:00:00",
    )
    assert not is_preferred_page(page)


def test_is_preferred_page_rejects_request_for_interview_post():
    page = FetchedPage(
        hit=SearchHit(query="q", title="佬可以出下agent开发面经吗", url="https://www.nowcoder.com/discuss/8"),
        canonical_url="https://www.nowcoder.com/discuss/8",
        title="佬可以出下agent开发面经吗",
        text="求面经，蹲一个真实分享",
        updated_at="2099-01-01T00:00:00",
    )
    assert not is_preferred_page(page)


def test_is_preferred_page_rejects_paid_content():
    page = FetchedPage(
        hit=SearchHit(query="q", title="快手 AI Agent开发 二面", url="https://www.nowcoder.com/discuss/9"),
        canonical_url="https://www.nowcoder.com/discuss/9",
        title="快手 AI Agent开发 二面",
        text="剩余60%内容，订阅专栏后可继续查看，也可单篇购买",
        updated_at="2099-01-01T00:00:00",
    )
    assert not is_preferred_page(page)


def test_is_paid_page_does_not_match_generic_html_shell_text():
    body = """
    <html><body><script>
      __INITIAL_STATE__={"prefetchData":{"2":{"ssrCommonData":{"contentData":{
        "title":"28届实习拷打，一场面试，23个Agent问题",
        "richText":"<p>1. 什么是agent？</p>"
      }}}}}
    </script><div>会员可见能力介绍</div></body></html>
    """
    text = "# 28届实习拷打，一场面试，23个Agent问题\n\n1. 什么是agent？"
    assert not is_paid_page(body, text)


def test_page_priority_prefers_newer_and_more_liked_pages():
    report = analyze_page(
        FetchedPage(
            hit=SearchHit(query="q", title="字节 Agent 一面 面经", url="https://www.nowcoder.com/discuss/5"),
            canonical_url="https://www.nowcoder.com/discuss/5",
            title="字节 Agent 一面 面经",
            text="1. 什么是 agent？",
            updated_at="2099-01-01T00:00:00",
            like_count=10,
            comment_count=1,
            view_count=100,
        )
    )
    assert report is not None

    newer = FetchedPage(
        hit=SearchHit(query="q", title="字节 Agent 一面 面经", url="https://www.nowcoder.com/discuss/6"),
        canonical_url="https://www.nowcoder.com/discuss/6",
        title="字节 Agent 一面 面经",
        text="1. 什么是 agent？\n2. 如何做 RAG？",
        updated_at="2099-01-02T00:00:00",
        like_count=100,
        comment_count=10,
        view_count=1000,
    )
    older = FetchedPage(
        hit=SearchHit(query="q", title="字节 Agent 一面 面经", url="https://www.nowcoder.com/discuss/7"),
        canonical_url="https://www.nowcoder.com/discuss/7",
        title="字节 Agent 一面 面经",
        text="1. 什么是 agent？",
        updated_at="2099-01-01T00:00:00",
        like_count=999,
        comment_count=999,
        view_count=9999,
    )
    report_newer = analyze_page(newer)
    report_older = analyze_page(older)
    assert report_newer is not None
    assert report_older is not None
    assert page_priority(newer, report_newer) > page_priority(older, report_older)


def test_analyze_page_requires_agent_and_interview_keywords():
    page = FetchedPage(
        hit=SearchHit(query="q", title="Agent 面经", url="https://www.nowcoder.com/discuss/123"),
        canonical_url="https://www.nowcoder.com/discuss/123",
        title="Agent 面经",
        text="1. 什么是 agent？\n2. RAG 和 agent 怎么结合？",
    )
    report = analyze_page(page)
    assert report is not None
    assert "什么是 agent" in report.questions[0]


def test_render_markdown_contains_source_and_questions():
    page = FetchedPage(
        hit=SearchHit(query="q", title="Agent 面经", url="https://www.nowcoder.com/discuss/123"),
        canonical_url="https://www.nowcoder.com/discuss/123",
        title="Agent 面经",
        text="1. 什么是 agent？\n2. RAG 和 agent 怎么结合？",
    )
    report = analyze_page(page)
    assert report is not None
    content = render_markdown(page, report)
    assert "https://www.nowcoder.com/discuss/123" in content
    assert "什么是 agent" in content
    assert "- 什么是 agent？" in content


def test_render_markdown_omits_duplicate_heading_from_source_body():
    page = FetchedPage(
        hit=SearchHit(query="q", title="Agent 面经", url="https://www.nowcoder.com/discuss/123"),
        canonical_url="https://www.nowcoder.com/discuss/123",
        title="Agent 面经",
        text="# Agent 面经\n\n1. 什么是 agent？\n2. RAG 怎么结合？",
    )
    report = analyze_page(page)
    assert report is not None
    content = render_markdown(page, report)
    assert "## 原文整理\n1. 什么是 agent？" in content


def test_build_canonical_id_uses_nowcoder_path():
    url = "https://www.nowcoder.com/discuss/123"
    assert build_canonical_id(url) == "discuss-123"


def test_extract_search_hits_reads_nowcoder_embedded_results():
    body = """
    <script>
      __INITIAL_STATE__={"app":{"180":{"records":[
        {"data":{"contentId":"863581806069673984","contentData":{"id":"863581806069673984","title":"CVTE AI Agent开发 一面","content":"1. 多 Agent 怎么拆？"},"frequencyData":{"likeCnt":2,"commentCnt":1,"viewCnt":88}}},
        {"data":{"contentId":"2813436","momentData":{"id":2813436,"title":"货拉拉Agent开发日常实习一面（45min）","content":"1. RAG 如何评测？"},"frequencyData":{"likeCnt":1,"commentCnt":0,"viewCnt":66}}}
      ]}}};
    </script>
    """
    hits = extract_search_hits(body, query="Agent 一面 面经", limit=5)
    assert len(hits) == 2
    assert hits[0].url == "https://www.nowcoder.com/discuss/863581806069673984"
    assert hits[1].url == "https://www.nowcoder.com/feed/main/detail/2813436"
    assert "CVTE AI Agent开发 一面" == hits[0].title


def test_extract_search_api_records_reads_native_nowcoder_payload():
    payload = {
        "success": True,
        "data": {
            "current": 1,
            "size": 20,
            "total": 1,
            "totalPage": 1,
            "records": [
                {
                    "contentType": 74,
                    "momentData": {
                        "id": 2813436,
                        "title": "货拉拉Agent开发日常实习一面（45min）",
                        "content": "1. RAG 如何评测？",
                        "showTime": 1773224897000,
                    },
                    "frequencyData": {"likeCnt": 1, "commentCnt": 0, "viewCnt": 66},
                }
            ],
        },
    }
    records, total_pages = extract_search_api_records(payload)
    hits = [parse_search_record(item, query="Agent 一面 面经") for item in records]
    hits = [item for item in hits if item is not None]
    assert total_pages == 1
    assert len(hits) == 1
    assert hits[0].url == "https://www.nowcoder.com/feed/main/detail/2813436"
    assert hits[0].title == "货拉拉Agent开发日常实习一面（45min）"
    assert hits[0].updated_at == "2026-03-11T18:28:17"
    assert hits[0].view_count == 66


def test_extract_home_recommend_records_reads_records_and_cursor():
    payload = {
        "success": True,
        "data": {
            "records": [
                {
                    "contentId": 2810407,
                    "contentType": 74,
                    "momentData": {
                        "id": 2810407,
                        "title": "AI agent研发实习面经分享",
                        "content": "1. 项目用的什么架构",
                        "showTime": 1773624300000,
                    },
                    "recommendData": {"cursorScore": 1.774000376919e9},
                }
            ],
        },
    }

    records, cursor_score = extract_home_recommend_records(payload)

    assert len(records) == 1
    assert records[0]["contentId"] == 2810407
    assert cursor_score == "1774000376.919"
    assert extract_recommend_cursor_score(records[0]) == "1774000376.919"


def test_extract_home_tab_records_reads_records_and_total_pages():
    payload = {
        "success": True,
        "data": {
            "current": 2,
            "size": 20,
            "total": 6486,
            "totalPage": 325,
            "records": [
                {
                    "contentId": "2814498",
                    "contentType": 74,
                    "momentData": {
                        "id": 2814498,
                        "title": "实习面经 字节大模型算法",
                        "content": "1. 实习拷打",
                        "showTime": 1773926700000,
                    },
                    "frequencyData": {"likeCnt": 1, "commentCnt": 0, "viewCnt": 148},
                }
            ],
        },
    }

    records, total_pages = extract_home_tab_records(payload)

    assert len(records) == 1
    assert records[0]["contentId"] == "2814498"
    assert total_pages == 325


def test_extract_recent_hits_reads_home_feed_payload():
    body = """
    <script>
      __INITIAL_STATE__={"app":{"main":{"feed":[
        {"contentId":2814949,"contentType":74,"title":"比心agent一面","content":"1. prompt 怎么设计\\n2. workflow 如何拆分","showTime":1773972000000,"frequencyData":{"likeCnt":3,"commentCnt":2,"viewCnt":120}},
        {"contentId":2815000,"contentType":74,"title":"普通后端一面","content":"1. MySQL 索引","showTime":1773972000000,"frequencyData":{"likeCnt":1,"commentCnt":0,"viewCnt":50}}
      ]}}};
    </script>
    """
    hits = extract_recent_hits(body)
    assert len(hits) == 2
    assert hits[0].url == "https://www.nowcoder.com/feed/main/detail/2814949"
    assert hits[0].updated_at
    assert hits[0].view_count == 120


def test_extract_discuss_hits_reads_recommend_records():
    body = """
    <script>
      __INITIAL_STATE__={"app":{"172":{"recommends":{"current":1,"total":2000,"records":[
        {"contentId":863581806069673984,"contentType":250,"contentData":{"id":"863581806069673984","title":"CVTE AI Agent开发 一面","content":"1. 多 Agent 怎么拆？","showTime":1773224897000},"frequencyData":{"likeCnt":2,"commentCnt":1,"viewCnt":88}},
        {"contentId":2813436,"contentType":74,"momentData":{"id":2813436,"title":"货拉拉Agent开发日常实习一面（45min）","content":"1. RAG 如何评测？","showTime":1773224897000},"frequencyData":{"likeCnt":1,"commentCnt":0,"viewCnt":66}}
      ]}}}};
    </script>
    """
    hits = extract_discuss_hits(body)
    assert len(hits) == 2
    assert hits[0].url == "https://www.nowcoder.com/discuss/863581806069673984"
    assert hits[1].url == "https://www.nowcoder.com/feed/main/detail/2813436"


def test_build_page_from_feed_search_hit_uses_snippet_and_metrics():
    hit = SearchHit(
        query="q",
        title="字节agent开发一面 实习面经",
        url="https://www.nowcoder.com/feed/main/detail/2810000",
        snippet="1.你了解哪些agent开发的框架？ 2.memory是什么了解吗？ 3.MCP了解吗？",
        updated_at="2099-01-01T00:00:00",
        like_count=8,
        comment_count=2,
        view_count=512,
    )
    page = build_page_from_search_hit(hit)
    assert page is not None
    assert page.title == hit.title
    assert "你了解哪些agent开发的框架" in page.text
    assert page.like_count == 8
    assert page.view_count == 512


def test_select_queries_ignores_generic_bag_of_keywords():
    queries = select_queries(
        ["AI Agent 大模型 智能体 面经 面试题 RAG Function Calling ReAct site:nowcoder.com"]
    )
    assert queries == DEFAULT_QUERIES


def test_targeted_query_requires_company_or_round_signal():
    assert is_targeted_query("字节 Agent 一面 面经")
    assert not is_targeted_query("AI Agent 大模型 智能体 面经 面试题")


def test_normalize_search_query_strips_search_engine_syntax():
    assert normalize_search_query('site:nowcoder.com/discuss "字节 Agent 一面 面经"') == "字节 Agent 一面 面经"


def test_material_path_rejects_discuss_comment_page():
    assert is_material_path("/discuss/comment/22503572") is False


@pytest.mark.asyncio
async def test_ingestor_dry_run_collects_output_paths():
    hit = SearchHit(
        query="q",
        title="Agent 面经",
        url="https://www.nowcoder.com/discuss/123",
        snippet="agent interview",
    )

    ingestor = NowcoderInterviewIngestor(rebuild_index=lambda: None)
    ingestor._fetch_hot_hits = AsyncMock(return_value=[])
    ingestor._fetch_search_hits = AsyncMock(return_value=[hit])
    ingestor._fetch_page = AsyncMock(
        return_value=FetchedPage(
            hit=hit,
            canonical_url=hit.url,
            title="字节 Agent 一面 面经",
            text="1. 什么是 agent？",
            updated_at="2099-01-01T00:00:00",
        )
    )

    result = await ingestor.run(queries=["q"], count_per_query=2, max_reports=1, dry_run=True)

    assert len(result["written"]) == 1
    assert result["written"][0].endswith("discuss-123.md")
