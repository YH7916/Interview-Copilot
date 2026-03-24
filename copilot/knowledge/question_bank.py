"""Aggregate raw interview reports into a structured question bank."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "data" / "knowledge_base" / "10-RealQuestions" / "Nowcoder-Agent-Reports"
OUTPUT_DIR = ROOT / "data" / "knowledge_base" / "10-RealQuestions"
REPORT_INDEX_PATH = OUTPUT_DIR / "report_index.json"
JSON_PATH = OUTPUT_DIR / "question_bank.json"
MARKDOWN_PATH = OUTPUT_DIR / "question_bank.md"
RECENT_JSON_PATH = OUTPUT_DIR / "recent_question_bank.json"
RECENT_MARKDOWN_PATH = OUTPUT_DIR / "recent_question_bank.md"
DEFAULT_RECENT_DAYS = 90

CATEGORY_LABELS = {
    "opening": "开场与经历",
    "project_background": "项目背景与目标",
    "project_scope": "职责边界与业务影响",
    "project_data": "数据与样本工程",
    "project_architecture": "项目架构与链路",
    "project_evaluation": "评估、记忆与反馈",
    "project_deployment": "部署、性能与成本",
    "project_challenges": "难点、故障与权衡",
    "project": "项目深挖",
    "prompt_context": "提示词与上下文工程",
    "agent_architecture": "Agent 架构与技能",
    "rag_retrieval": "RAG 与检索链路",
    "code_agent": "Code Agent 与工程实现",
    "llm_fundamentals": "LLM 基础原理",
    "python_system": "Python 与系统基础",
    "coding": "算法与手撕",
}

CATEGORY_RULES = [
    ("opening", ("自我介绍", "实习经历", "项目介绍", "经历介绍")),
    ("prompt_context", ("提示词", "prompt", "上下文工程", "todo list", "system prompt", "模板", "context engineering")),
    ("agent_architecture", ("agent", "react", "多agent", "单agent", "multi-agent", "skill", "tool", "planner", "memory", "mcp")),
    ("rag_retrieval", ("rag", "召回", "改写", "query rewrite", "embedding", "rerank", "粗排", "精排", "检索", "bm25", "向量")),
    ("code_agent", ("ast", "lsp", "单测", "覆盖率", "mock", "插桩", "代码解析", "测试生成", "分支覆盖率")),
    ("llm_fundamentals", ("llm", "attention", "transformer", "qkv", "decoder", "prefix lm", "causal lm", "token", "embedding", "lora", "sft", "微调", "bert", "llama", "qwen", "glm", "开源模型", "mask", "multi-head", "self attention")),
    ("python_system", ("python", "gil", "多线程", "锁", "rlock", "信号量", "cpp", "编译", "链接")),
    ("coding", ("手撕", "算法", "岛屿", "链表", "二叉树", "动态规划", "acm")),
]

CODING_STRONG_KEYWORDS = (
    "手撕",
    "写代码",
    "代码题",
    "算法题",
    "编程题",
    "现场 coding",
    "coding",
    "leetcode",
    "lc ",
    "机试",
    "笔试题",
    "acm",
)

CODING_PROBLEM_KEYWORDS = (
    "岛屿",
    "链表",
    "二叉树",
    "二叉搜索树",
    "动态规划",
    "dp",
    "dfs",
    "bfs",
    "回溯",
    "滑动窗口",
    "单调栈",
    "队列",
    "栈",
    "堆",
    "并查集",
    "拓扑排序",
    "最短路径",
    "快排",
    "归并",
    "接雨水",
    "括号匹配",
    "lru",
)

CODING_EXCLUDE_KEYWORDS = (
    "rlhf",
    "ppo",
    "dpo",
    "grpo",
    "attention",
    "transformer",
    "lora",
    "qlora",
    "rag",
    "agent",
    "memory",
    "prompt",
    "embedding",
)

LLM_FUNDAMENTALS_HINTS = (
    "rlhf",
    "ppo",
    "dpo",
    "grpo",
    "attention",
    "transformer",
    "lora",
    "qlora",
    "deepspeed",
    "vllm",
    "flash attention",
    "蒸馏",
    "量化",
    "kl",
)

PROJECT_BACKGROUND_KEYWORDS = (
    "背景", "场景", "业务", "用户", "需求", "目标", "价值", "收益",
    "指标", "kpi", "roi", "为什么要做", "为什么做", "项目介绍", "业务价值",
)

PROJECT_SCOPE_KEYWORDS = (
    "负责", "边界", "分工", "角色", "贡献", "影响力", "最有影响力",
    "代表自己的项目", "负责的部分", "负责哪些", "具体工作内容",
)

PROJECT_DATA_KEYWORDS = (
    "数据", "样本", "语料", "标注", "打标", "清洗", "去重", "过滤",
    "切分", "chunk", "数据集", "负样本", "采样", "构造数据", "自动化生成数据",
)

PROJECT_ARCHITECTURE_KEYWORDS = (
    "架构", "链路", "模块", "组件", "流程", "整体", "系统设计", "异步",
    "调度", "编排", "路由", "pipeline", "router", "orchestr", "workflow",
)

PROJECT_EVALUATION_KEYWORDS = (
    "评估", "评测", "a/b", "ab", "judge", "feedback", "反馈", "命中率",
    "召回率", "准确率", "f1", "precision", "recall", "memory", "记忆",
    "长期记忆", "短期记忆", "弱点", "复盘",
)

PROJECT_DEPLOYMENT_KEYWORDS = (
    "部署", "上线", "推理", "时延", "延迟", "吞吐", "并发", "qps",
    "成本", "优化", "缓存", "显存", "量化", "服务化", "稳定性",
    "vllm", "onnx", "flash attention", "benchmark",
)

PROJECT_CHALLENGE_KEYWORDS = (
    "问题", "难点", "困难", "故障", "异常", "中断", "tradeoff", "权衡",
    "失败", "瓶颈", "风险", "坑", "如何解决", "解决办法", "怎么避免",
    "怎么保证", "怎么处理", "纠偏",
)

PROJECT_SIGNAL_KEYWORDS = (
    "项目", "业务", "场景", "上线", "落地", "效果", "指标", "系统",
    "模块", "链路", "方案", "设计", "优化", "部署",
)

STAGE_BY_CATEGORY = {
    "opening": "opening",
    "project_background": "project",
    "project_scope": "project",
    "project_data": "project",
    "project_architecture": "project",
    "project_evaluation": "project",
    "project_deployment": "project",
    "project_challenges": "project",
    "project": "project",
    "prompt_context": "project",
    "agent_architecture": "project",
    "rag_retrieval": "project",
    "code_agent": "project",
    "llm_fundamentals": "foundations",
    "python_system": "foundations",
    "coding": "coding",
}

QUESTION_CUES = (
    "怎么",
    "如何",
    "为什么",
    "什么",
    "哪些",
    "哪个",
    "区别",
    "原理",
    "流程",
    "实现",
    "设计",
    "有没有",
    "是否",
    "讲",
    "说",
    "介绍",
    "理解",
    "知道",
    "了解",
    "谁",
)

CONTEXT_PREFIX = re.compile(
    r"^\s*(?:\d+[.)、]\s*)?(?:(?:[一二三四五六七八九十]+面|问项目|问实习|问八股|八股|实习项目\d*|代码|手撕|追问)\s*[:：]\s*)"
)
PROJECT_PREFIX = re.compile(
    r"^\s*(?:实习项目\d*|项目|项目里|项目中|拷打实现细节)\s*(?:\([^)]*\)|（[^）]*）)?\s*[:：]\s*"
)
BUSINESS_PREFIX = re.compile(r"^\s*(?:业务背景|部门业务背景)\s*[，,:：]\s*")
QUESTION_LINE = re.compile(r"^\d+\.\s*(.*)$")
BULLET_LINE = re.compile(r"^[-*+]\s+(.*)$")


@dataclass(slots=True)
class QuestionSource:
    title: str
    source_url: str
    source_path: str
    captured_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "source_url": self.source_url,
            "source_path": self.source_path,
            "captured_at": self.captured_at,
        }


@dataclass(slots=True)
class BankQuestion:
    question: str
    category: str
    stage: str
    aliases: list[str] = field(default_factory=list)
    sources: list[QuestionSource] = field(default_factory=list)

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def latest_source_at(self) -> str:
        values = [item.captured_at for item in self.sources if item.captured_at]
        return max(values) if values else ""

    def add_occurrence(self, alias: str, source: QuestionSource) -> None:
        if alias not in self.aliases:
            self.aliases.append(alias)
        if not any(item.source_url == source.source_url for item in self.sources):
            self.sources.append(source)
        self.question = min([self.question, alias], key=lambda value: (len(value), value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "category": self.category,
            "category_label": CATEGORY_LABELS[self.category],
            "stage": self.stage,
            "aliases": self.aliases,
            "source_count": self.source_count,
            "latest_source_at": self.latest_source_at,
            "sources": [item.to_dict() for item in self.sources],
        }


@dataclass(slots=True)
class BankCluster:
    head: str
    category: str
    stage: str
    aliases: list[str] = field(default_factory=list)
    follow_up_counts: dict[str, int] = field(default_factory=dict)
    sources: list[QuestionSource] = field(default_factory=list)

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def latest_source_at(self) -> str:
        values = [item.captured_at for item in self.sources if item.captured_at]
        return max(values) if values else ""

    def add_occurrence(self, head: str, follow_ups: list[str], source: QuestionSource) -> None:
        if head not in self.aliases:
            self.aliases.append(head)
        if not any(item.source_url == source.source_url for item in self.sources):
            self.sources.append(source)
        self.head = min([self.head, head], key=lambda value: (len(value), value))
        for item in follow_ups:
            if item and item != self.head:
                self.follow_up_counts[item] = self.follow_up_counts.get(item, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        ranked_follow_ups = sorted(
            self.follow_up_counts.items(),
            key=lambda item: (-item[1], len(item[0]), item[0]),
        )
        return {
            "question": self.head,
            "category": self.category,
            "category_label": CATEGORY_LABELS[self.category],
            "stage": self.stage,
            "aliases": self.aliases,
            "follow_ups": [text for text, _ in ranked_follow_ups[:4]],
            "source_count": self.source_count,
            "latest_source_at": self.latest_source_at,
            "sources": [item.to_dict() for item in self.sources],
        }


def rebuild_question_bank(
    source_dir: Path = SOURCE_DIR,
    output_dir: Path = OUTPUT_DIR,
    recent_days: int = DEFAULT_RECENT_DAYS,
) -> dict[str, Any]:
    reports = _collect_reports(source_dir, load_report_index())
    reports = [report for report in reports if report["questions"] and _is_bank_candidate(report)]
    bank = build_question_bank(reports)
    recent_bank = build_question_bank(_recent_reports(reports, recent_days))

    output_dir.mkdir(parents=True, exist_ok=True)
    REPORT_INDEX_PATH.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    JSON_PATH.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")
    MARKDOWN_PATH.write_text(render_question_bank_markdown(bank), encoding="utf-8")
    RECENT_JSON_PATH.write_text(json.dumps(recent_bank, ensure_ascii=False, indent=2), encoding="utf-8")
    RECENT_MARKDOWN_PATH.write_text(render_question_bank_markdown(recent_bank), encoding="utf-8")

    return {
        "reports": len(reports),
        "questions": sum(len(category["questions"]) for category in bank["categories"]),
        "recent_questions": sum(len(category["questions"]) for category in recent_bank["categories"]),
        "report_index_path": str(REPORT_INDEX_PATH),
        "json_path": str(JSON_PATH),
        "markdown_path": str(MARKDOWN_PATH),
        "recent_json_path": str(RECENT_JSON_PATH),
        "recent_markdown_path": str(RECENT_MARKDOWN_PATH),
    }


def load_question_bank(path: Path | None = None, *, recent: bool = False) -> dict[str, Any]:
    target = path or (RECENT_JSON_PATH if recent else JSON_PATH)
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {
            "generated_at": "",
            "source_dir": str(target.parent),
            "categories": [],
        }


def load_report_index(path: Path = REPORT_INDEX_PATH) -> list[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def build_question_bank(reports: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[BankQuestion]] = {name: [] for name in CATEGORY_LABELS}
    cluster_buckets: dict[str, list[BankCluster]] = {name: [] for name in CATEGORY_LABELS}

    for report in reports:
        source = QuestionSource(
            title=report["title"],
            source_url=report["source_url"],
            source_path=report["source_path"],
            captured_at=report["captured_at"],
        )
        for raw in report["questions"]:
            exploded = explode_questions(raw)
            if exploded:
                cluster_category = _classify_group(exploded)
                cluster = _find_existing_cluster(cluster_buckets[cluster_category], exploded[0])
                if cluster is None:
                    cluster = BankCluster(
                        head=exploded[0],
                        category=cluster_category,
                        stage=STAGE_BY_CATEGORY[cluster_category],
                        aliases=[exploded[0]],
                        sources=[source],
                    )
                    cluster_buckets[cluster_category].append(cluster)
                cluster.add_occurrence(exploded[0], exploded[1:], source)

            for question in exploded:
                category = classify_question(question)
                item = _find_existing(buckets[category], question)
                if item is None:
                    item = BankQuestion(
                        question=question,
                        category=category,
                        stage=STAGE_BY_CATEGORY[category],
                        aliases=[question],
                        sources=[source],
                    )
                    buckets[category].append(item)
                else:
                    item.add_occurrence(question, source)

    categories = []
    for name, questions in buckets.items():
        clusters = cluster_buckets[name]
        if not questions and not clusters:
            continue
        questions.sort(key=lambda item: (item.latest_source_at, item.source_count, item.question), reverse=True)
        clusters.sort(key=lambda item: (item.latest_source_at, item.source_count, item.head), reverse=True)
        categories.append(
            {
                "name": name,
                "label": CATEGORY_LABELS[name],
                "stage": STAGE_BY_CATEGORY[name],
                "clusters": [item.to_dict() for item in clusters],
                "questions": [item.to_dict() for item in questions],
            }
        )

    categories.sort(key=lambda item: list(CATEGORY_LABELS).index(item["name"]))
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(SOURCE_DIR),
        "categories": categories,
    }


def render_question_bank_markdown(bank: dict[str, Any]) -> str:
    lines = [
        "---",
        "doc_type: agent_question_bank",
        "---",
        "",
        "# Agent Interview Question Bank",
        "",
        f"- generated_at: {bank['generated_at']}",
        "",
    ]

    for category in bank["categories"]:
        lines.append(f"## {category['label']}")
        lines.append("")
        for index, question in enumerate(category.get("clusters") or category["questions"], 1):
            lines.append(f"{index}. {question['question']}")
            lines.append(f"   - stage: {question['stage']}")
            lines.append(f"   - frequency: {question['source_count']}")
            if question.get("latest_source_at"):
                lines.append(f"   - latest_seen: {question['latest_source_at']}")
            follow_ups = question.get("follow_ups", [])
            if follow_ups:
                lines.append(f"   - follow-ups: {' / '.join(follow_ups[:3])}")
            aliases = [alias for alias in question["aliases"] if alias != question["question"]][:2]
            if aliases:
                lines.append(f"   - variants: {' / '.join(aliases)}")
            sources = [item["title"] for item in question["sources"][:2]]
            if sources:
                lines.append(f"   - sources: {' / '.join(sources)}")
        lines.append("")

    return "\n".join(lines)


def _collect_reports(source_dir: Path, cached: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_path = {item["source_path"]: item for item in cached if item.get("source_path")}
    reports = []

    for path in sorted(source_dir.glob("*.md")):
        stat = path.stat()
        cache = by_path.get(str(path))
        if cache and cache.get("file_mtime") == stat.st_mtime and cache.get("file_size") == stat.st_size:
            reports.append(cache)
            continue
        reports.append(_parse_report(path))

    return reports


def _recent_reports(reports: list[dict[str, Any]], recent_days: int) -> list[dict[str, Any]]:
    cutoff = datetime.now() - timedelta(days=recent_days)
    result = []
    for report in reports:
        try:
            captured_at = datetime.fromisoformat(report.get("captured_at", ""))
        except ValueError:
            continue
        if captured_at >= cutoff:
            result.append(report)
    return result


def explode_questions(text: str) -> list[str]:
    base = _clean_question(text)
    if not base:
        return []
    if _looks_like_project_placeholder(base):
        return []

    candidates = [base]
    for separator in ("？", "?", "；", ";", "。"):
        split_items: list[str] = []
        for item in candidates:
            split_items.extend(piece for piece in item.split(separator) if piece.strip())
        candidates = split_items or candidates

    if len(base) > 24:
        split_items = []
        for item in candidates:
            split_items.extend(piece for piece in item.split("，") if piece.strip())
        candidates = split_items or candidates

    questions = []
    for item in candidates:
        cleaned = _clean_question(item)
        if cleaned and looks_like_question(cleaned):
            questions.append(cleaned)

    if questions:
        return _dedupe(questions)
    return [] if _looks_like_project_placeholder(base) else [base]


def classify_question(question: str) -> str:
    lowered = question.lower()
    if _looks_like_coding_question(question, lowered):
        return "coding"
    if _looks_like_llm_fundamentals_question(lowered):
        return "llm_fundamentals"
    for name, keywords in CATEGORY_RULES:
        if name == "coding":
            continue
        if any(keyword in lowered for keyword in keywords):
            return name
    return _classify_project_question(question, lowered)


def _looks_like_coding_question(question: str, lowered: str) -> bool:
    if any(keyword in lowered for keyword in CODING_STRONG_KEYWORDS):
        return True
    if any(keyword in lowered for keyword in CODING_EXCLUDE_KEYWORDS):
        return False
    return any(keyword in question for keyword in CODING_PROBLEM_KEYWORDS)


def _looks_like_llm_fundamentals_question(lowered: str) -> bool:
    return any(keyword in lowered for keyword in LLM_FUNDAMENTALS_HINTS)


def _classify_group(questions: list[str]) -> str:
    ranked = [classify_question(item) for item in questions]
    for category in ranked:
        if category not in {"project", "project_background"}:
            return category
    return ranked[0] if ranked else "project"


def _classify_project_question(question: str, lowered: str) -> str:
    if any(keyword in lowered for keyword in PROJECT_SCOPE_KEYWORDS):
        return "project_scope"
    if any(keyword in lowered for keyword in PROJECT_DATA_KEYWORDS):
        return "project_data"
    if any(keyword in lowered for keyword in PROJECT_EVALUATION_KEYWORDS):
        return "project_evaluation"
    if any(keyword in lowered for keyword in PROJECT_DEPLOYMENT_KEYWORDS):
        return "project_deployment"
    if any(keyword in lowered for keyword in PROJECT_CHALLENGE_KEYWORDS):
        return "project_challenges"
    if any(keyword in lowered for keyword in PROJECT_ARCHITECTURE_KEYWORDS):
        return "project_architecture"
    if any(keyword in lowered for keyword in PROJECT_BACKGROUND_KEYWORDS):
        return "project_background"
    if any(keyword in question for keyword in PROJECT_SIGNAL_KEYWORDS):
        return "project"
    return "project"


def looks_like_question(text: str) -> bool:
    if text in {"自我介绍", "项目介绍", "介绍一下你自己"}:
        return True
    if _looks_like_project_placeholder(text):
        return False
    return len(text) >= 6 and any(cue in text.lower() for cue in QUESTION_CUES)


def _looks_like_project_placeholder(text: str) -> bool:
    normalized = re.sub(r"\s+", "", str(text or "")).lower()
    return normalized in {
        "项目拷打",
        "深挖项目",
        "项目相关",
        "项目深挖",
        "项目提问",
        "项目+技术拷打",
        "实习拷打",
        "论文拷打",
    }


def _parse_report(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), path.stem)
    source_url = _match_frontmatter(text, "source_url")
    stat = path.stat()
    return {
        "title": title,
        "source_url": source_url,
        "source_path": str(path),
        "source_type": _match_frontmatter(text, "source_type"),
        "captured_at": _match_frontmatter(text, "fetched_at") or datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "file_mtime": stat.st_mtime,
        "file_size": stat.st_size,
        "content_hash": sha256(text.encode("utf-8")).hexdigest(),
        "questions": _extract_question_lines(text),
    }


def _is_bank_candidate(report: dict[str, Any]) -> bool:
    if report.get("source_type") == "nowcoder_page":
        return True
    title = report.get("title", "")
    return any(keyword in title for keyword in ("面经", "一面", "二面", "三面"))


def _extract_question_lines(text: str) -> list[str]:
    lines = text.splitlines()
    questions: list[str] = []
    in_questions = False

    for line in lines:
        if line.startswith("## "):
            name = line[3:].strip()
            if name in {"高频题目", "题目"}:
                in_questions = True
                continue
            if in_questions:
                break
        if not in_questions:
            continue

        stripped = line.strip()
        match = QUESTION_LINE.match(stripped) or BULLET_LINE.match(stripped)
        if not match:
            continue
        question = match.group(1).strip()
        if not question or question.startswith(("未提取到明确题目", "鏈彁鍙栧埌鏄庣‘棰樼洰")):
            continue
        questions.append(question)

    return questions


def _match_frontmatter(text: str, key: str) -> str:
    match = re.search(rf"^{key}:\s+\"?(.*?)\"?$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _clean_question(text: str) -> str:
    value = str(text or "").strip()
    value = QUESTION_LINE.sub(r"\1", value)
    while True:
        updated = CONTEXT_PREFIX.sub("", value).strip()
        if updated == value:
            break
        value = updated
    while True:
        updated = PROJECT_PREFIX.sub("", value).strip()
        if updated == value:
            break
        value = updated
    while True:
        updated = BUSINESS_PREFIX.sub("", value).strip()
        if updated == value:
            break
        value = updated
    value = re.sub(r"\s+", " ", value)
    return value.strip(" -:：，,")


def _normalize_question(text: str) -> str:
    value = text.lower()
    replacements = {
        "llm": "大模型",
        "multi-agent": "多agent",
        "single-agent": "单agent",
        "prompt engineering": "提示词工程",
        "promptengineering": "提示词工程",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = re.sub(r"[\W_]+", "", value)
    return value


def _find_existing(items: list[BankQuestion], question: str) -> BankQuestion | None:
    normalized = _normalize_question(question)
    for item in items:
        current = _normalize_question(item.question)
        if current == normalized:
            return item
        if current in normalized or normalized in current:
            return item
        if SequenceMatcher(None, current, normalized).ratio() >= 0.86:
            return item
    return None


def _find_existing_cluster(items: list[BankCluster], head: str) -> BankCluster | None:
    normalized = _normalize_question(head)
    for item in items:
        current = _normalize_question(item.head)
        if current == normalized:
            return item
        if current in normalized or normalized in current:
            return item
        if SequenceMatcher(None, current, normalized).ratio() >= 0.82:
            return item
    return None


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _normalize_question(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
