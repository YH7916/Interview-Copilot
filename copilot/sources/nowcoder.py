"""Nowcoder-only interview ingestion."""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlsplit, urlunsplit
from uuid import uuid4

import httpx
from readability import Document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KB_ROOT = PROJECT_ROOT / "data" / "knowledge_base"
OUTPUT_DIR = KB_ROOT / "10-RealQuestions" / "Nowcoder-Agent-Reports"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
SEARCH_API_URL = "https://gw-c.nowcoder.com/api/sparta/pc/search"
HOME_RECOMMEND_API_URL = "https://gw-c.nowcoder.com/api/sparta/home/recommend"
HOME_TAB_CONTENT_API_URL = "https://gw-c.nowcoder.com/api/sparta/home/tab/content"
SEARCH_PAGE_SIZE = 20
HOME_RECOMMEND_PAGE_SIZE = 20
HOME_RECOMMEND_MAX_PAGES = 15
HOME_TAB_CONTENT_PAGE_SIZE = 20
HOME_TAB_CONTENT_MAX_PAGES = 25
HOME_TAB_INTERVIEW_CATEGORY_TYPE = 1
HOME_TAB_INTERVIEW_TAB_ID = 818
RECENT_FEED_MAX_PAGES = 15
DISCOVERY_TOPICS = (
    "Agent",
    "AI Agent",
    "智能体",
    "大模型 Agent",
    "AI应用开发",
    "大模型应用",
    "RAG Agent",
    "RAG",
    "MCP Agent",
    "MCP",
)
DISCOVERY_PATTERNS = (
    "{topic} 一面 面经",
    "{topic} 面经",
    "一面 面经 {topic}",
    "{topic} 开发 一面",
)
DISCOVERY_EXTRAS = (
    "Agent 二面 面经",
    "Agent 三面 面经",
    "Agent 面筋",
    "Agent 凉经",
    "Agent 实习面经",
    "春招 Agent 一面",
    "暑期实习 Agent 一面",
    "攒人品 Agent 一面",
    "一面 面经",
    "二面 面经",
    "三面 面经",
    "大模型 面经",
    "智能体 面经",
    "RAG 面经",
    "MCP 面经",
    "大模型 一面",
    "智能体 一面",
    "AI应用开发 面经",
)
GENERIC_DISCOVERY_QUERIES = (
    "面经",
    "一面",
    "二面",
    "三面",
    "凉经",
    "面筋",
    "攒人品",
    "实习 面经",
    "日常实习 一面",
    "暑期实习 一面",
    "大模型应用 面经",
    "AI应用 面经",
    "工作流 面经",
    "function calling 面经",
    "tool calling 面经",
    "embedding 面经",
    "rerank 面经",
    "prompt 面经",
)
INTERVIEW_KEYWORDS = ("面经", "面试", "面试题", "题库", "复习", "总结", "interview")
AGENT_KEYWORDS = (
    "agent",
    "ai agent",
    "llm",
    "rag",
    "mcp",
    "智能体",
    "大模型",
    "大模型应用",
    "ai应用开发",
    "tool",
    "memory",
    "planner",
)
QUESTION_PREFIXES = ("q:", "问:", "问题", "面试题", "题目", "追问")
TRUNCATED_MARKER = "[内容过长，已截断]"
EMBEDDED_STATE_MARKER = "__INITIAL_STATE__="
COMPANY_KEYWORDS = (
    "字节", "抖音", "腾讯", "阿里", "美团", "快手", "百度", "小红书",
    "京东", "蚂蚁", "滴滴", "华为", "拼多多", "携程", "b站", "米哈游",
)
PRIORITY_COMPANIES = ("字节", "抖音", "腾讯", "阿里", "美团", "快手", "百度", "小红书", "蚂蚁")
DISCOVERY_COMPANY_QUERIES = tuple(f"{company} 一面 面经" for company in COMPANY_KEYWORDS)
DISCOVERY_COMPANY_TOPIC_QUERIES = tuple(
    f"{company} {topic} 面经"
    for company in COMPANY_KEYWORDS
    for topic in ("Agent", "大模型", "智能体", "AI应用", "RAG")
)
ROUND_KEYWORDS = ("一面", "二面", "三面", "四面", "hr面", "终面", "凉经", "面筋", "挂经")
AGGREGATE_KEYWORDS = ("总结", "合集", "汇总", "题库", "八股", "答案", "参考答案", "教程", "大全")
SHARE_KEYWORDS = ("面经", "面筋", "凉经", "挂经", "一面", "二面", "三面", "四面", "hr面", "终面", "offer", "oc")
DISCOVERY_KEYWORDS = SHARE_KEYWORDS + ("面试", "面试题")
REAL_SHARE_HINTS = SHARE_KEYWORDS + ("45min", "60min", "90min", "自我介绍", "反问")
STRONG_AGENT_KEYWORDS = (
    "agent", "智能体", "agentic", "multi-agent", "multi agent", "mcp",
    "rag", "function calling", "tool calling",
)
WEAK_AGENT_KEYWORDS = (
    "llm", "大模型", "prompt", "上下文工程", "skill", "skills", "planner",
    "memory", "embedding", "rerank", "检索", "召回", "重排", "workflow", "工作流",
)
NOISE_KEYWORDS = (
    "内推", "招聘", "招人", "春招启动", "秋招启动", "提前批启动", "校招启动",
    "实习招聘", "求职就业", "训练营", "公开课", "课程", "广告", "投递", "岗位直推",
    "求面经", "蹲面经", "来个面经", "有没有面经", "出下面经", "求分享面经",
    "求拷打", "简历求拷打", "学习路线", "路线规划", "怎么学", "求建议", "可以吗",
)
BLOCKED_ACCESS_KEYWORDS = (
    "付费", "收费", "解锁全文", "付费阅读", "私密", "隐私", "仅楼主可见",
    "仅作者可见", "不可见", "订阅后查看", "会员可见", "订阅专栏后可继续查看",
    "单篇购买", "剩余60%内容",
)
ROLE_NOISE_KEYWORDS = (
    "前端", "后端", "java", "c++", "测试", "客户端", "服务端", "产品", "运营",
    "测开", "嵌入式", "安全", "反作弊", "风控",
)
QUESTION_HINTS = (
    "什么", "为什么", "如何", "怎么", "哪些", "区别", "原理", "流程", "实现",
    "设计", "理解", "介绍", "场景", "作用", "优缺点", "优势", "难点", "策略",
    "机制", "是否", "有没有", "听说过", "讲讲",
)
ANSWER_PREFIXES = (
    "核心原因", "原因", "步骤", "常见方案", "常见解决", "区别本质", "主要来自",
    "这是一种", "选择", "具体的结合方式", "典型格式", "核心思路",
)
NUMBERED_QUESTION_RE = re.compile(r"^(?:\d+\s*[.)、．]\s*|\d+\s+)")
QUESTION_NOISE_RE = re.compile(
    r"^(?:\d{1,2}(?:[./月-]\d{1,2}(?:日)?)?(?:\s*[一二三四五六七八九十]?面)?(?:\s*\d+\s*min)?|"
    r"[一二三四五六七八九十]面\s*\d+\s*min)$",
    re.IGNORECASE,
)


def build_default_queries() -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for query in GENERIC_DISCOVERY_QUERIES:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
    for pattern in DISCOVERY_PATTERNS:
        for topic in DISCOVERY_TOPICS:
            query = pattern.format(topic=topic)
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            queries.append(query)
    for query in DISCOVERY_EXTRAS:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
    for query in DISCOVERY_COMPANY_QUERIES:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
    for query in DISCOVERY_COMPANY_TOPIC_QUERIES:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
    return queries


DEFAULT_QUERIES = build_default_queries()


@dataclass(slots=True)
class SearchHit:
    query: str
    title: str
    url: str
    snippet: str = ""
    hot_value: int = 0
    is_paid: bool = False
    updated_at: str = ""
    like_count: int = 0
    comment_count: int = 0
    view_count: int = 0


@dataclass(slots=True)
class FetchedPage:
    hit: SearchHit
    canonical_url: str
    title: str
    text: str
    updated_at: str = ""
    like_count: int = 0
    comment_count: int = 0
    view_count: int = 0
    is_paid: bool = False


@dataclass(slots=True)
class PageReport:
    title: str
    summary: str
    questions: list[str]


class NowcoderInterviewIngestor:
    """Search Nowcoder pages and save relevant interview notes."""

    def __init__(
        self,
        *,
        output_dir: Path = OUTPUT_DIR,
        rebuild_index: Callable[[], Any] | None = None,
        fetch_timeout_seconds: float = 12.0,
        updated_within_days: int = 30,
    ):
        self.output_dir = output_dir
        self.rebuild_index = rebuild_index or _default_rebuild_index
        self.fetch_timeout_seconds = fetch_timeout_seconds
        self.updated_within_days = updated_within_days
        self.discovery_cuid = str(uuid4())
        self.nowcoder_cookie = clean_text(os.environ.get("NOWCODER_COOKIE"))

    def _build_headers(self, *, referer: str, xhr: bool = False) -> dict[str, str]:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": referer,
        }
        if xhr:
            headers["X-Requested-With"] = "XMLHttpRequest"
        if self.nowcoder_cookie:
            headers["Cookie"] = self.nowcoder_cookie
        return headers

    async def run(
        self,
        *,
        queries: list[str] | None = None,
        count_per_query: int = 6,
        max_reports: int = 8,
        dry_run: bool = False,
        rebuild_index: bool = True,
    ) -> dict[str, Any]:
        chosen_queries = select_runtime_queries(queries, max_reports=max_reports)
        written: list[str] = []
        skipped = {"irrelevant": 0, "fetch_error": 0, "stale": 0}
        candidates: list[tuple[FetchedPage, PageReport]] = []
        semaphore = asyncio.Semaphore(8)
        seen_urls: set[str] = set()

        async def _evaluate(hit: SearchHit):
            async with semaphore:
                return await self._evaluate_hit(hit)

        async def _consume_hits(hits: list[SearchHit]):
            batch: list[SearchHit] = []
            for hit in hits:
                url = canonicalize_url(hit.url)
                if not url or url in seen_urls or not is_nowcoder_candidate(hit, url):
                    continue
                seen_urls.add(url)
                hit.url = url
                batch.append(hit)
            tasks = [asyncio.create_task(_evaluate(hit)) for hit in batch]
            for task in asyncio.as_completed(tasks):
                status, payload = await task
                if status == "ok":
                    candidates.append(payload)
                else:
                    skipped[status] += 1

        tab_hits = await self._fetch_home_tab_hits()
        await _consume_hits(tab_hits)

        if not tab_hits:
            await _consume_hits(await self._fetch_home_recommend_hits())
            await _consume_hits(await self._fetch_recent_hits())
            await _consume_hits(await self._fetch_discuss_hits())
            await _consume_hits(await self._fetch_hot_hits())

        if queries or not tab_hits:
            for query in chosen_queries:
                if len(candidates) >= max_reports:
                    break
                await _consume_hits(await self._fetch_search_hits(query, count_per_query))

        candidates.sort(key=lambda item: page_priority(item[0], item[1]), reverse=True)

        for page, report in candidates[:max_reports]:
            path = self.output_dir / f"{build_canonical_id(page.canonical_url)}.md"
            if not dry_run:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(render_markdown(page, report), encoding="utf-8")
            written.append(str(path))

        if written and rebuild_index and not dry_run:
            _default_rebuild_question_bank()
            self.rebuild_index()

        return {
            "queries": chosen_queries,
            "written": written,
            "skipped": skipped,
            "dry_run": dry_run,
            "using_nowcoder_cookie": bool(self.nowcoder_cookie),
            "primary_source": "home_tab_content" if tab_hits else "public_fallback",
        }

    async def _evaluate_hit(self, hit: SearchHit) -> tuple[str, tuple[FetchedPage, PageReport] | None]:
        page = await self._fetch_page(hit)
        if page is None:
            return "fetch_error", None
        if not is_recent_page(page, within_days=self.updated_within_days):
            return "stale", None

        report = analyze_page(page)
        if report is None or not is_preferred_page(page):
            return "irrelevant", None
        return "ok", (page, report)

    async def _discover(self, queries: list[str], count_per_query: int, *, limit: int) -> list[SearchHit]:
        hits: list[SearchHit] = []
        seen: set[str] = set()

        tab_hits = await self._fetch_home_tab_hits()
        for hit in tab_hits:
            url = canonicalize_url(hit.url)
            if url in seen or not is_nowcoder_candidate(hit, url):
                continue
            seen.add(url)
            hit.url = url
            hits.append(hit)
            if len(hits) >= limit:
                hits.sort(key=score_hit, reverse=True)
                return hits

        if not tab_hits:
            for hit in await self._fetch_home_recommend_hits():
                url = canonicalize_url(hit.url)
                if url in seen or not is_nowcoder_candidate(hit, url):
                    continue
                seen.add(url)
                hit.url = url
                hits.append(hit)
                if len(hits) >= limit:
                    hits.sort(key=score_hit, reverse=True)
                    return hits

            for hit in await self._fetch_recent_hits():
                url = canonicalize_url(hit.url)
                if url in seen or not is_nowcoder_candidate(hit, url):
                    continue
                seen.add(url)
                hit.url = url
                hits.append(hit)
                if len(hits) >= limit:
                    hits.sort(key=score_hit, reverse=True)
                    return hits

            for hit in await self._fetch_discuss_hits():
                url = canonicalize_url(hit.url)
                if url in seen or not is_nowcoder_candidate(hit, url):
                    continue
                seen.add(url)
                hit.url = url
                hits.append(hit)
                if len(hits) >= limit:
                    hits.sort(key=score_hit, reverse=True)
                    return hits

            for hit in await self._fetch_hot_hits():
                url = canonicalize_url(hit.url)
                if url in seen or not is_nowcoder_candidate(hit, url):
                    continue
                seen.add(url)
                hit.url = url
                hits.append(hit)
                if len(hits) >= limit:
                    hits.sort(key=score_hit, reverse=True)
                    return hits

        for query in queries:
            for hit in await self._fetch_search_hits(query, count_per_query):
                url = canonicalize_url(hit.url)
                if url in seen or not is_nowcoder_candidate(hit, url):
                    continue
                seen.add(url)
                hit.url = url
                hits.append(hit)
                if len(hits) >= limit:
                    hits.sort(key=score_hit, reverse=True)
                    return hits

        hits.sort(key=score_hit, reverse=True)
        return hits

    async def _fetch_home_tab_hits(self) -> list[SearchHit]:
        hits: list[SearchHit] = []
        seen: set[str] = set()
        stale_pages = 0

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=self.fetch_timeout_seconds,
                trust_env=False,
                headers=self._build_headers(referer="https://www.nowcoder.com/", xhr=True),
            ) as client:
                total_pages = HOME_TAB_CONTENT_MAX_PAGES
                for page in range(1, HOME_TAB_CONTENT_MAX_PAGES + 1):
                    params = {
                        "pageNo": page,
                        "categoryType": HOME_TAB_INTERVIEW_CATEGORY_TYPE,
                        "tabId": HOME_TAB_INTERVIEW_TAB_ID,
                    }
                    response = await client.get(HOME_TAB_CONTENT_API_URL, params=params)
                    response.raise_for_status()
                    records, total_pages = extract_home_tab_records(response.json())
                    if not records:
                        break

                    page_recent_hits = 0
                    for record in records:
                        hit = parse_search_record(record, query="nowcoder_home_tab_interview")
                        if hit is None:
                            continue
                        url = canonicalize_url(hit.url)
                        if not url or url in seen:
                            continue
                        seen.add(url)
                        hit.url = url
                        hits.append(hit)
                        if hit.updated_at and is_recent_timestamp(hit.updated_at, within_days=self.updated_within_days):
                            page_recent_hits += 1

                    if page_recent_hits == 0:
                        stale_pages += 1
                        if stale_pages >= 2:
                            break
                    else:
                        stale_pages = 0

                    if page >= total_pages:
                        break
        except Exception:
            return hits

        return hits

    async def _fetch_home_recommend_hits(self) -> list[SearchHit]:
        hits: list[SearchHit] = []
        seen: set[str] = set()
        cursor_score: str | None = None
        empty_pages = 0
        no_growth_pages = 0

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=self.fetch_timeout_seconds,
                trust_env=False,
                headers=self._build_headers(referer="https://www.nowcoder.com/", xhr=True),
            ) as client:
                for page in range(1, HOME_RECOMMEND_MAX_PAGES + 1):
                    params: dict[str, Any] = {
                        "page": page,
                        "size": HOME_RECOMMEND_PAGE_SIZE,
                        "cuid": self.discovery_cuid,
                    }
                    if cursor_score:
                        params["cursorScore"] = cursor_score
                    response = await client.get(HOME_RECOMMEND_API_URL, params=params)
                    response.raise_for_status()
                    records, next_cursor_score = extract_home_recommend_records(response.json())
                    if not records:
                        empty_pages += 1
                        if empty_pages >= 2:
                            break
                        continue
                    empty_pages = 0
                    page_new_hits = 0
                    for record in records:
                        hit = parse_search_record(record, query="nowcoder_home_recommend")
                        if hit is None:
                            continue
                        url = canonicalize_url(hit.url)
                        if not url or url in seen:
                            continue
                        seen.add(url)
                        hit.url = url
                        hits.append(hit)
                        page_new_hits += 1
                    if page_new_hits == 0:
                        no_growth_pages += 1
                        if no_growth_pages >= 2:
                            break
                    else:
                        no_growth_pages = 0
                    if len(records) < HOME_RECOMMEND_PAGE_SIZE:
                        break
                    if not next_cursor_score or next_cursor_score == cursor_score:
                        break
                    cursor_score = next_cursor_score
        except Exception:
            return hits

        return hits

    async def _fetch_recent_hits(self) -> list[SearchHit]:
        hits: list[SearchHit] = []
        seen: set[str] = set()
        empty_pages = 0
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=self.fetch_timeout_seconds,
                trust_env=False,
                headers=self._build_headers(referer="https://www.nowcoder.com/"),
            ) as client:
                for page in range(1, RECENT_FEED_MAX_PAGES + 1):
                    url = "https://www.nowcoder.com/?target=main" if page == 1 else f"https://www.nowcoder.com/?target=main&page={page}"
                    response = await client.get(url)
                    response.raise_for_status()
                    page_hits = extract_recent_hits(response.text)
                    if not page_hits:
                        empty_pages += 1
                        if empty_pages >= 2:
                            break
                        continue
                    empty_pages = 0
                    for hit in page_hits:
                        canonical_url = canonicalize_url(hit.url)
                        if not canonical_url or canonical_url in seen:
                            continue
                        seen.add(canonical_url)
                        hit.url = canonical_url
                        hits.append(hit)
        except Exception:
            return hits

        return hits

    async def _fetch_discuss_hits(self) -> list[SearchHit]:
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=self.fetch_timeout_seconds,
                trust_env=False,
                headers=self._build_headers(referer="https://www.nowcoder.com/"),
            ) as client:
                response = await client.get("https://www.nowcoder.com/discuss?order=3&type=2")
                response.raise_for_status()
        except Exception:
            return []

        return extract_discuss_hits(response.text)

    async def _fetch_hot_hits(self) -> list[SearchHit]:
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=self.fetch_timeout_seconds,
                trust_env=False,
                headers=self._build_headers(referer="https://www.nowcoder.com/"),
            ) as client:
                response = await client.get("https://www.nowcoder.com/home/top-posts")
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return []

        hits: list[SearchHit] = []
        for item in payload.get("data") or []:
            title = clean_text(item.get("title"))
            if not title:
                continue
            if not is_hot_post_candidate(title):
                continue
            url = build_hot_post_url(item)
            if not url:
                continue
            hits.append(
                SearchHit(
                    query="nowcoder_hot_posts",
                    title=title,
                    url=url,
                    snippet="nowcoder hot post",
                    hot_value=int(item.get("hotValueFromDolphin") or 0),
                )
            )
        return hits

    async def _fetch_search_hits(self, query: str, limit: int) -> list[SearchHit]:
        normalized_query = normalize_search_query(query)
        if not normalized_query:
            return []

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=self.fetch_timeout_seconds,
                trust_env=False,
                headers=self._build_headers(
                    referer=f"https://www.nowcoder.com/search/all/?query={quote(normalized_query)}"
                ),
            ) as client:
                hits = await self._fetch_search_hits_via_api(client, normalized_query, limit)
                if hits:
                    return hits

                response = await client.get("https://www.nowcoder.com/search/all/", params={"query": normalized_query})
                response.raise_for_status()
        except Exception:
            return []

        return extract_search_hits(response.text, query=normalized_query, limit=limit)

    async def _fetch_search_hits_via_api(
        self,
        client: httpx.AsyncClient,
        query: str,
        limit: int,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        seen: set[str] = set()
        total_pages = 1
        target_hits = query_target_hits(query, limit)
        max_pages = max(2, (target_hits + SEARCH_PAGE_SIZE - 1) // SEARCH_PAGE_SIZE)

        for page in range(1, max_pages + 1):
            response = await client.post(
                SEARCH_API_URL,
                json={"type": "post", "query": query, "page": page},
            )
            response.raise_for_status()
            payload = response.json()
            records, total_pages = extract_search_api_records(payload)
            if not records:
                break

            for record in records:
                hit = parse_search_record(record, query=query)
                if hit is None:
                    continue
                if hit.updated_at and not is_recent_timestamp(hit.updated_at, within_days=self.updated_within_days):
                    continue
                url = canonicalize_url(hit.url)
                if not url or url in seen:
                    continue
                seen.add(url)
                hit.url = url
                hits.append(hit)
                if len(hits) >= target_hits:
                    return hits

            if page >= total_pages:
                break

        return hits

    async def _fetch_page(self, hit: SearchHit) -> FetchedPage | None:
        if "/feed/main/detail/" in hit.url:
            return build_page_from_search_hit(hit)

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=self.fetch_timeout_seconds,
                trust_env=False,
                headers=self._build_headers(referer="https://www.nowcoder.com/"),
            ) as client:
                response = await client.get(hit.url)
                response.raise_for_status()
        except Exception:
            return None

        text = extract_page_text(response, max_chars=24_000)
        if not text:
            return None

        title = extract_page_title(response.text) or hit.title
        return FetchedPage(
            hit=hit,
            canonical_url=canonicalize_url(str(response.url)),
            title=clean_text(title),
            text=text,
            updated_at=extract_page_updated_at(response.text) or hit.updated_at,
            like_count=extract_page_metric(response.text, "likeCnt"),
            comment_count=extract_page_metric(response.text, "commentCnt"),
            view_count=extract_page_metric(response.text, "viewCnt"),
            is_paid=hit.is_paid or is_paid_page(response.text, text),
        )


def build_page_from_search_hit(hit: SearchHit) -> FetchedPage | None:
    text = clean_text_block(f"# {hit.title}\n\n{normalize_content_text(hit.snippet)}")
    if len(text) < 40:
        return None
    return FetchedPage(
        hit=hit,
        canonical_url=canonicalize_url(hit.url),
        title=clean_text(hit.title),
        text=text,
        updated_at=hit.updated_at,
        like_count=hit.like_count,
        comment_count=hit.comment_count,
        view_count=hit.view_count,
        is_paid=hit.is_paid,
    )

def analyze_page(page: FetchedPage) -> PageReport | None:
    haystack = f"{page.title}\n{page.text[:5000]}".lower()
    if page.is_paid:
        return None
    if not is_agent_relevant(haystack, title=page.title):
        return None
    if not contains_any(haystack, INTERVIEW_KEYWORDS + SHARE_KEYWORDS):
        return None

    return PageReport(
        title=page.title or page.hit.title or "Nowcoder Agent Interview",
        summary=first_paragraph(page.text, 180),
        questions=extract_questions(page.text, limit=12),
    )


def render_markdown(page: FetchedPage, report: PageReport) -> str:
    source_body = strip_leading_markdown_title(page.text, page.title)
    content = source_body[:12_000].strip()
    if len(source_body) > 12_000:
        content += f"\n\n{TRUNCATED_MARKER}"

    lines = [
        "---",
        "doc_type: agent_interview_report",
        'source_type: "nowcoder_page"',
        f"source_url: {json.dumps(page.canonical_url, ensure_ascii=False)}",
        f'fetched_at: "{datetime.now().isoformat(timespec="seconds")}"',
        f'source_updated_at: "{page.updated_at}"',
        f"source_like_count: {page.like_count}",
        f"source_comment_count: {page.comment_count}",
        f"source_view_count: {page.view_count}",
        "---",
        "",
        f"# {report.title}",
        "",
        "## 来源",
        f"- 链接: {page.canonical_url}",
        "",
        "## 摘要",
        report.summary,
        "",
        "## 题目",
        *[f"- {question}" for question in report.questions],
        "",
        "## 原文整理",
        content,
        "",
    ]
    return "\n".join(lines)


def extract_search_hits(body: str, *, query: str, limit: int) -> list[SearchHit]:
    payload = parse_embedded_state(body)
    records = find_search_records(payload)
    hits: list[SearchHit] = []

    for record in records:
        hit = parse_search_record(record, query=query)
        if hit is None:
            continue
        hits.append(hit)
        if len(hits) >= limit:
            break

    deduped: list[SearchHit] = []
    seen: set[str] = set()
    for hit in hits:
        url = canonicalize_url(hit.url)
        if not url or url in seen:
            continue
        seen.add(url)
        hit.url = url
        deduped.append(hit)
    return deduped


def extract_search_api_records(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(payload, dict) or not payload.get("success"):
        return [], 0
    data = payload.get("data")
    if not isinstance(data, dict):
        return [], 0
    records = data.get("records")
    if not isinstance(records, list):
        return [], 0
    try:
        total_pages = max(1, int(data.get("totalPage") or 1))
    except Exception:
        total_pages = 1
    return [item for item in records if isinstance(item, dict)], total_pages


def extract_home_recommend_records(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(payload, dict) or not payload.get("success"):
        return [], ""
    data = payload.get("data")
    if not isinstance(data, dict):
        return [], ""
    records = data.get("records")
    if not isinstance(records, list):
        return [], ""

    items = [item for item in records if isinstance(item, dict)]
    next_cursor_score = ""
    for item in reversed(items):
        next_cursor_score = extract_recommend_cursor_score(item)
        if next_cursor_score:
            break
    return items, next_cursor_score


def extract_home_tab_records(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(payload, dict) or not payload.get("success"):
        return [], 0
    data = payload.get("data")
    if not isinstance(data, dict):
        return [], 0
    records = data.get("records")
    if not isinstance(records, list):
        return [], 0
    try:
        total_pages = max(1, int(data.get("totalPage") or 1))
    except Exception:
        total_pages = 1
    return [item for item in records if isinstance(item, dict)], total_pages


def find_search_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        current = value.get("records")
        if isinstance(current, list):
            records.extend(item for item in current if isinstance(item, dict))
        for item in value.values():
            records.extend(find_search_records(item))
    elif isinstance(value, list):
        for item in value:
            records.extend(find_search_records(item))
    return records


def extract_recent_hits(body: str, *, limit: int = 80) -> list[SearchHit]:
    payload = parse_embedded_state(body)
    if not payload:
        return []

    hits: list[SearchHit] = []
    seen: set[str] = set()
    for item in iter_recent_feed_items(payload):
        hit = parse_recent_feed_item(item)
        if hit is None:
            continue
        url = canonicalize_url(hit.url)
        if not url or url in seen:
            continue
        seen.add(url)
        hit.url = url
        hits.append(hit)
        if len(hits) >= limit:
            break
    return hits


def extract_discuss_hits(body: str, *, limit: int = 40) -> list[SearchHit]:
    payload = parse_embedded_state(body)
    if not payload:
        return []

    hits: list[SearchHit] = []
    seen: set[str] = set()
    for record in find_discuss_records(payload):
        hit = parse_search_record(record, query="nowcoder_discuss")
        if hit is None:
            continue
        url = canonicalize_url(hit.url)
        if not url or url in seen:
            continue
        seen.add(url)
        hit.url = url
        hits.append(hit)
        if len(hits) >= limit:
            break
    return hits


def find_discuss_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        recommends = value.get("recommends")
        if isinstance(recommends, dict):
            current = recommends.get("records")
            if isinstance(current, list):
                records.extend(item for item in current if isinstance(item, dict))
        for item in value.values():
            records.extend(find_discuss_records(item))
    elif isinstance(value, list):
        for item in value:
            records.extend(find_discuss_records(item))
    return records


def iter_recent_feed_items(value: Any):
    if isinstance(value, dict):
        if value.get("contentType") == 74 and value.get("contentId"):
            yield value
        for item in value.values():
            yield from iter_recent_feed_items(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_recent_feed_items(item)


def parse_recent_feed_item(item: dict[str, Any]) -> SearchHit | None:
    identifier = clean_text(item.get("contentId") or item.get("id"))
    title = clean_text(item.get("title") or item.get("newTitle") or item.get("desc"))
    snippet = normalize_content_text(item.get("content") or item.get("newContent") or item.get("desc"))
    if not identifier or not title:
        return None
    frequency = item.get("frequencyData")
    if not isinstance(frequency, dict):
        frequency = {}
    return SearchHit(
        query="nowcoder_recent",
        title=title,
        url=f"https://www.nowcoder.com/feed/main/detail/{identifier}",
        snippet=snippet,
        hot_value=safe_int(frequency.get("viewCnt")) + safe_int(frequency.get("likeCnt")) * 100 + safe_int(frequency.get("commentCnt")) * 50,
        is_paid=contains_any(f"{title}\n{snippet}".lower(), BLOCKED_ACCESS_KEYWORDS),
        updated_at=extract_record_updated_at(item),
        like_count=safe_int(frequency.get("likeCnt")),
        comment_count=safe_int(frequency.get("commentCnt")),
        view_count=safe_int(frequency.get("viewCnt")),
    )


def parse_search_record(record: dict[str, Any], *, query: str) -> SearchHit | None:
    data = record.get("data") if isinstance(record.get("data"), dict) else record
    if not isinstance(data, dict):
        return None

    content = data.get("momentData")
    if not isinstance(content, dict):
        content = data.get("contentData")
    if not isinstance(content, dict):
        return None

    url = build_search_result_url(data, content)
    title = clean_text(content.get("title") or record.get("title"))
    snippet = normalize_content_text(content.get("desc") or content.get("content"))
    if not url or not title:
        return None
    if is_paid_search_record(data):
        return None

    frequency = data.get("frequencyData")
    like_count = comment_count = view_count = 0
    if isinstance(frequency, dict):
        try:
            like_count = max(0, int(frequency.get("likeCnt", 0)))
            comment_count = max(0, int(frequency.get("commentCnt", 0)))
            view_count = max(0, int(frequency.get("viewCnt", 0)))
        except Exception:
            like_count = comment_count = view_count = 0

    return SearchHit(
        query=query,
        title=title,
        url=url,
        snippet=snippet,
        hot_value=view_count + like_count * 100 + comment_count * 50,
        is_paid=False,
        updated_at=extract_record_updated_at(content),
        like_count=like_count,
        comment_count=comment_count,
        view_count=view_count,
    )


def build_search_result_url(data: dict[str, Any], content: dict[str, Any]) -> str:
    if isinstance(data.get("momentData"), dict):
        identifier = clean_text(content.get("id") or data.get("contentId"))
        if not identifier:
            return ""
        return f"https://www.nowcoder.com/feed/main/detail/{identifier}"
    identifier = clean_text(
        content.get("entityId")
        or data.get("entityId")
        or data.get("contentId")
        or content.get("id")
    )
    if not identifier:
        return ""
    return f"https://www.nowcoder.com/discuss/{identifier}"


def extract_recommend_cursor_score(record: dict[str, Any]) -> str:
    recommend_data = record.get("recommendData")
    if not isinstance(recommend_data, dict):
        return ""
    value = recommend_data.get("cursorScore")
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return str(value)
    return clean_text(value)


def extract_record_updated_at(content: dict[str, Any]) -> str:
    for key in ("showTime", "editTime", "updateTime", "createTime", "createdAt"):
        timestamp = parse_timestamp_ms(content.get(key))
        if timestamp:
            return timestamp
    return ""


def is_paid_search_record(data: dict[str, Any]) -> bool:
    blog = data.get("blogZhuanlan")
    if isinstance(blog, dict):
        try:
            if float(blog.get("articlePrice") or 0) > 0:
                return True
        except Exception:
            return True
    content = data.get("contentData")
    if not isinstance(content, dict):
        content = data.get("momentData")
    if isinstance(content, dict):
        haystack = f"{clean_text(content.get('title'))}\n{clean_text(content.get('content'))}".lower()
        if contains_any(haystack, BLOCKED_ACCESS_KEYWORDS):
            return True
    return False


def is_paid_page(body: str, text: str) -> bool:
    if contains_any(text.lower(), BLOCKED_ACCESS_KEYWORDS):
        return True
    payload = parse_embedded_state(body)
    return payload_contains_paid_marker(payload)


def payload_contains_paid_marker(value: Any) -> bool:
    if isinstance(value, dict):
        blog = value.get("blogZhuanlan")
        if isinstance(blog, dict):
            try:
                if float(blog.get("articlePrice") or 0) > 0:
                    return True
            except Exception:
                return True
        if value.get("beMyOnly") is True:
            return True
        for item in value.values():
            if payload_contains_paid_marker(item):
                return True
    elif isinstance(value, list):
        for item in value:
            if payload_contains_paid_marker(item):
                return True
    return False


def is_nowcoder_candidate(hit: SearchHit, url: str) -> bool:
    parsed = urlsplit(url)
    if parsed.netloc.lower() not in {"nowcoder.com", "www.nowcoder.com"}:
        return False
    if not is_material_path(parsed.path):
        return False

    haystack = f"{hit.title}\n{hit.snippet}\n{url}".lower()
    if hit.is_paid:
        return False
    if any(keyword in haystack for keyword in NOISE_KEYWORDS):
        return False
    return contains_any(haystack, DISCOVERY_KEYWORDS) or any(
        keyword in haystack for keyword in COMPANY_KEYWORDS + ROUND_KEYWORDS
    )


def score_hit(hit: SearchHit) -> int:
    haystack = f"{hit.title}\n{hit.snippet}\n{hit.url}".lower()
    path = urlsplit(hit.url).path.lower()
    agent_score = _agent_score(haystack, hit.title)
    score = 0

    if "/discuss/" in path:
        score += 6
    if "/feed/main/detail/" in path:
        score += 5
    if contains_any(haystack, INTERVIEW_KEYWORDS):
        score += 4
    if contains_any(haystack, AGENT_KEYWORDS):
        score += 6
    if contains_any(haystack, REAL_SHARE_HINTS):
        score += 4
    if any(keyword in hit.title for keyword in COMPANY_KEYWORDS):
        score += 5
    if any(keyword in hit.title for keyword in ROUND_KEYWORDS):
        score += 5
    if any(keyword in haystack for keyword in ("真实", "回忆", "流程", "挂了", "offer")):
        score += 2
    if any(keyword in haystack for keyword in AGGREGATE_KEYWORDS):
        score -= 8
    if any(keyword in haystack for keyword in BLOCKED_ACCESS_KEYWORDS + NOISE_KEYWORDS):
        score -= 12
    if any(keyword in hit.title for keyword in ROLE_NOISE_KEYWORDS) and agent_score < 5:
        score -= 8
    score += min(hit.hot_value // 1000, 6)
    score += agent_score

    return score


def is_hot_post_candidate(title: str) -> bool:
    lowered = title.lower()
    if any(keyword in lowered for keyword in NOISE_KEYWORDS + BLOCKED_ACCESS_KEYWORDS):
        return False
    if any(keyword in lowered for keyword in DISCOVERY_KEYWORDS):
        return True
    return any(keyword in lowered for keyword in STRONG_AGENT_KEYWORDS + WEAK_AGENT_KEYWORDS)


def build_hot_post_url(item: dict[str, Any]) -> str:
    identifier = clean_text(item.get("id"))
    if not identifier:
        return ""
    return f"https://www.nowcoder.com/discuss/{identifier}"


def select_queries(queries: list[str] | None) -> list[str]:
    cleaned = [normalize_search_query(item) for item in queries or [] if item and item.strip()]
    targeted = [item for item in cleaned if is_targeted_query(item)]
    return targeted or list(DEFAULT_QUERIES)


def select_runtime_queries(queries: list[str] | None, *, max_reports: int) -> list[str]:
    selected = select_queries(queries)
    if queries:
        return selected
    ranked = sorted(selected, key=query_runtime_priority, reverse=True)
    limit = 24 if max_reports >= 100 else 40
    return ranked[:limit]


def is_targeted_query(query: str) -> bool:
    lowered = query.lower()
    has_company = any(keyword in query for keyword in COMPANY_KEYWORDS)
    has_round = any(keyword in query for keyword in ROUND_KEYWORDS)
    has_share_signal = contains_any(lowered, SHARE_KEYWORDS + INTERVIEW_KEYWORDS)
    has_agent = _agent_score(lowered) >= 1 or contains_any(lowered, AGENT_KEYWORDS)
    query_terms = [part for part in query.split() if part]
    max_terms = 6 if (has_company or has_round) else 4
    return has_agent and (has_company or has_round or has_share_signal) and len(query_terms) <= max_terms


def normalize_search_query(query: str) -> str:
    value = clean_text(query)
    value = re.sub(r"\bsite:[^\s]+", " ", value, flags=re.IGNORECASE)
    value = value.replace("/discuss", " ").replace("/feed/main/detail", " ")
    value = value.replace('"', " ").replace("'", " ")
    return clean_text(value)


def extract_page_text(response: httpx.Response, *, max_chars: int) -> str:
    doc = Document(response.text)
    content = html_to_text(doc.summary())
    title = clean_text(doc.title())
    embedded_title, embedded_content = extract_embedded_content(response.text)
    if len(content) < 200 and embedded_content:
        content = embedded_content
    if embedded_title:
        title = embedded_title
    text = f"# {title}\n\n{content}" if title else content
    return clean_text_block(text[:max_chars])


def extract_page_title(body: str) -> str:
    title, _ = extract_embedded_content(body)
    return title or clean_text(Document(body).title())


def extract_page_updated_at(body: str) -> str:
    payload = parse_embedded_state(body)
    if not payload:
        return ""
    content = find_rich_text_payload(payload)
    if not content:
        return ""
    for key in ("showTime", "editTime", "updateTime", "createTime", "createdAt"):
        value = content.get(key)
        timestamp = parse_timestamp_ms(value)
        if timestamp:
            return timestamp
    return ""


def extract_page_metric(body: str, key: str) -> int:
    payload = parse_embedded_state(body)
    if not payload:
        return 0
    content = find_rich_text_payload(payload)
    if not content:
        return 0
    frequency = content.get("frequencyData")
    if not isinstance(frequency, dict):
        return 0
    try:
        return max(0, int(frequency.get(key, 0)))
    except Exception:
        return 0


def extract_embedded_content(body: str) -> tuple[str, str]:
    payload = parse_embedded_state(body)
    if not payload:
        return "", ""

    content = find_rich_text_payload(payload)
    if not content:
        return "", ""

    title = clean_text(content.get("title"))
    text = html_to_text(content.get("richText", ""))
    return title, text


def parse_embedded_state(body: str) -> dict[str, Any]:
    marker_index = body.find(EMBEDDED_STATE_MARKER)
    if marker_index < 0:
        return {}

    start = marker_index + len(EMBEDDED_STATE_MARKER)
    try:
        payload, _ = json.JSONDecoder().raw_decode(body[start:])
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def find_rich_text_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        rich_text = value.get("richText")
        if isinstance(rich_text, str) and rich_text.strip():
            return value
        for item in value.values():
            found = find_rich_text_payload(item)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = find_rich_text_payload(item)
            if found:
                return found
    return {}


def html_to_text(value: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", "", value, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|section|article|h\d|li|ul|ol|blockquote)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return clean_text_block(text)


def normalize_content_text(value: Any) -> str:
    text = str(value or "")
    if "<" in text and ">" in text:
        text = html_to_text(text)
    else:
        text = html.unescape(text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = split_inline_numbered_lines(text)
        text = clean_text_block(text)
    return text


def split_inline_numbered_lines(text: str) -> str:
    return re.sub(
        r"(?<!\n)(?<!\d)\s+((?:\d{1,2}|[一二三四五六七八九十]+)[.、．])\s*(?=[A-Za-z\u4e00-\u9fff])",
        r"\n\1 ",
        text,
    )


def strip_leading_markdown_title(text: str, title: str) -> str:
    heading = f"# {clean_text(title)}"
    body = text.strip()
    if body.startswith(heading):
        return body[len(heading):].lstrip()
    return body


def extract_questions(text: str, *, limit: int) -> list[str]:
    questions: list[str] = []
    normalized_text = split_inline_numbered_lines(text)
    for line in normalized_text.splitlines():
        raw = clean_text(line)
        stripped = normalize_question_line(raw)
        if len(stripped) < 4:
            continue
        if looks_like_question(raw):
            questions.append(stripped)
        if len(questions) >= limit:
            break
    return dedupe(questions) or ["未提取到明确题目，建议查看原文整理。"]


def parse_timestamp_ms(value: Any) -> str:
    try:
        raw = int(value)
    except Exception:
        return ""
    if raw <= 0:
        return ""
    if raw > 10**12:
        raw //= 1000
    try:
        return datetime.fromtimestamp(raw).isoformat(timespec="seconds")
    except Exception:
        return ""


def is_recent_page(page: FetchedPage, *, within_days: int) -> bool:
    if within_days <= 0:
        return True
    if not page.updated_at:
        return False
    return is_recent_timestamp(page.updated_at, within_days=within_days)


def is_recent_timestamp(value: str, *, within_days: int) -> bool:
    if within_days <= 0:
        return True
    try:
        updated_at = datetime.fromisoformat(value)
    except ValueError:
        return False
    return updated_at >= datetime.now() - timedelta(days=within_days)


def is_preferred_page(page: FetchedPage) -> bool:
    haystack = f"{page.title}\n{page.hit.title}\n{page.hit.snippet}\n{page.text[:4000]}".lower()
    has_share_signal = any(keyword in haystack for keyword in SHARE_KEYWORDS)
    has_interview_signal = any(keyword in haystack for keyword in INTERVIEW_KEYWORDS + SHARE_KEYWORDS)
    has_company_or_round = any(keyword in haystack for keyword in COMPANY_KEYWORDS + ROUND_KEYWORDS)
    title_has_agent = contains_any(page.title.lower(), STRONG_AGENT_KEYWORDS + WEAK_AGENT_KEYWORDS)
    agent_score = _agent_score(page.text[:4000], page.title)
    strong_count, weak_count = _agent_counts(page.text[:4000])
    title_lower = page.title.lower()
    if any(keyword in haystack for keyword in NOISE_KEYWORDS):
        return False
    if page.is_paid or page.hit.is_paid:
        return False
    if any(keyword in haystack for keyword in BLOCKED_ACCESS_KEYWORDS):
        return False
    if any(keyword in haystack for keyword in AGGREGATE_KEYWORDS) and not has_company_or_round:
        return False
    if any(keyword in title_lower for keyword in ROLE_NOISE_KEYWORDS) and strong_count < 2 and weak_count < 2:
        return False
    if any(keyword in page.title for keyword in ROLE_NOISE_KEYWORDS) and agent_score < 5:
        return False
    if not has_interview_signal or not (has_share_signal or has_company_or_round):
        return False
    if agent_score >= 4:
        return True
    return title_has_agent and (strong_count >= 1 or weak_count >= 2)


def page_priority(page: FetchedPage, report: PageReport) -> tuple[float, int, int, int, int]:
    return (
        parse_iso_timestamp(page.updated_at),
        page.like_count,
        page.comment_count,
        page.view_count,
        len(report.questions),
    )


def parse_iso_timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return 0.0


def query_target_hits(query: str, limit: int) -> int:
    lowered = query.lower()
    query_terms = len([part for part in query.split() if part])
    is_generic = not contains_any(lowered, AGENT_KEYWORDS) and not any(keyword in query for keyword in COMPANY_KEYWORDS)
    if is_generic:
        multiplier = 6 if query_terms <= 2 else 4
    elif query_terms <= 2:
        multiplier = 5
    else:
        multiplier = 3
    return min(max(limit * multiplier, SEARCH_PAGE_SIZE * 5), SEARCH_PAGE_SIZE * 30)


def query_runtime_priority(query: str) -> tuple[int, int, int]:
    lowered = query.lower()
    has_company = any(keyword in query for keyword in COMPANY_KEYWORDS)
    priority_company = any(keyword in query for keyword in PRIORITY_COMPANIES)
    has_agent = contains_any(lowered, AGENT_KEYWORDS)
    generic_share = query in GENERIC_DISCOVERY_QUERIES
    return (
        3 if generic_share else 0,
        2 if has_agent else 0,
        1 if has_company and priority_company else 0,
    )


def safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except Exception:
        return 0


def looks_like_question(line: str) -> bool:
    normalized = normalize_question_line(line)
    lowered = normalized.lower()
    if not normalized or normalized.endswith((":", "：")):
        return False
    if QUESTION_NOISE_RE.match(normalized):
        return False
    if any(normalized.startswith(prefix) for prefix in ANSWER_PREFIXES):
        return False
    if line.endswith("?") or line.endswith("？"):
        return True
    if lowered.startswith(QUESTION_PREFIXES):
        return True
    if any(hint in lowered for hint in QUESTION_HINTS):
        return True
    if NUMBERED_QUESTION_RE.match(line):
        return len(normalized) <= 30 and not any(marker in normalized for marker in ("如下", "例如", "包括"))
    return False


def normalize_question_line(line: str) -> str:
    value = clean_text(line)
    while True:
        updated = NUMBERED_QUESTION_RE.sub("", value).strip()
        if updated == value:
            break
        value = updated
    return value.strip(" -:：")


def is_agent_relevant(text: str, *, title: str = "") -> bool:
    title_lower = title.lower()
    body_strong_count, body_weak_count = _agent_counts(text)
    if any(keyword in title_lower for keyword in STRONG_AGENT_KEYWORDS) and (
        body_strong_count >= 1 or body_weak_count >= 2
    ):
        return True
    if body_strong_count >= 2:
        return True
    if body_strong_count >= 1 and body_weak_count >= 1:
        return True
    if any(keyword in title_lower for keyword in WEAK_AGENT_KEYWORDS) and body_weak_count >= 2:
        return True
    if body_weak_count >= 3:
        return True
    return False


def keyword_hit_count(text: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def _agent_counts(text: str, title: str = "") -> tuple[int, int]:
    lowered = f"{title}\n{text}".lower()
    return keyword_hit_count(lowered, STRONG_AGENT_KEYWORDS), keyword_hit_count(lowered, WEAK_AGENT_KEYWORDS)


def _agent_score(text: str, title: str = "") -> int:
    strong_count, weak_count = _agent_counts(text, title)
    score = strong_count * 3 + weak_count
    if any(keyword in title.lower() for keyword in STRONG_AGENT_KEYWORDS):
        score += 2
    return score


def first_paragraph(text: str, max_chars: int) -> str:
    for block in text.split("\n\n"):
        cleaned = clean_text(block)
        if len(cleaned) >= 30:
            return cleaned[:max_chars]
    return clean_text(text)[:max_chars]


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_text_block(value: str) -> str:
    text = value.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def is_material_path(path: str) -> bool:
    path = path.lower()
    if "/discuss/comment/" in path:
        return False
    if "/discuss/" in path or "/feed/main/detail/" in path:
        return not any(marker in path for marker in ("/creation/subject/", "/exam/", "/topics/", "/subject/"))
    return False


def canonicalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    return urlunsplit((parsed.scheme, parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))


def build_canonical_id(url: str) -> str:
    path = urlsplit(url).path.strip("/").replace("/", "-").lower()
    return path or "nowcoder-page"


def _default_rebuild_index():
    from copilot.knowledge.index import rebuild_chroma_collection

    return rebuild_chroma_collection()


def _default_rebuild_question_bank():
    from copilot.knowledge.pipeline import rebuild_question_bank

    return rebuild_question_bank()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search Nowcoder agent interview materials and ingest them.")
    parser.add_argument("--query", action="append", dest="queries", help="Custom search query.")
    parser.add_argument("--count-per-query", type=int, default=6, help="Search results per query.")
    parser.add_argument("--max-reports", type=int, default=8, help="Maximum reports to write.")
    parser.add_argument("--dry-run", action="store_true", help="Discover and analyze without writing files.")
    parser.add_argument("--skip-rebuild", action="store_true", help="Skip vector index rebuild.")
    parser.add_argument("--fetch-timeout", type=float, default=12.0, help="Per-page fetch timeout in seconds.")
    parser.add_argument("--updated-within-days", type=int, default=30, help="Only keep materials updated within this many days.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = asyncio.run(
        NowcoderInterviewIngestor(
            fetch_timeout_seconds=args.fetch_timeout,
            updated_within_days=args.updated_within_days,
        ).run(
            queries=args.queries,
            count_per_query=args.count_per_query,
            max_reports=args.max_reports,
            dry_run=args.dry_run,
            rebuild_index=not args.skip_rebuild,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
