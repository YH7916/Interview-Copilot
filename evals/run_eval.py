"""Focused evaluation suites for the current Interview Harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from copilot.config import get_text_settings
from copilot.interview.evaluation import evaluate_answer as evaluate_against_reference
from copilot.interview.policy import should_follow_up
from copilot.knowledge.question_bank import classify_question
from copilot.llm import call_text, parse_json_response
from copilot.sources.nowcoder import FetchedPage, SearchHit, analyze_page, is_preferred_page

DATASETS_DIR = BASE_DIR / "evals" / "datasets"
DEFAULT_DATASETS = {
    "ingest": DATASETS_DIR / "ingest_eval.json",
    "cluster": DATASETS_DIR / "cluster_eval.json",
    "interview_policy": DATASETS_DIR / "interview_policy_eval.json",
    "review": DATASETS_DIR / "review_eval.json",
}

JudgeFn = Callable[[str], str]


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def call_text_model(prompt: str, *, task: str) -> str:
    return call_text(prompt, task=task)


def compute_recall(retrieved_sources: set[str], expected_sources: set[str]) -> float | None:
    if not expected_sources:
        return None
    return len(retrieved_sources & expected_sources) / len(expected_sources)


def run_ingest_eval(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for sample in dataset:
        page = _make_eval_page(sample)
        report = analyze_page(page)
        predicted_keep = report is not None and is_preferred_page(page)
        records.append(
            {
                "title": sample["title"],
                "expected_keep": bool(sample["expected_keep"]),
                "predicted_keep": predicted_keep,
                "matched": predicted_keep is bool(sample["expected_keep"]),
            }
        )
    return _finalize_suite("ingest", records)


def run_cluster_eval(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for sample in dataset:
        predicted = classify_question(sample["question"])
        records.append(
            {
                "question": sample["question"],
                "expected_category": sample["expected_category"],
                "predicted_category": predicted,
                "matched": predicted == sample["expected_category"],
            }
        )
    return _finalize_suite("cluster", records)


def run_interview_policy_eval(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for sample in dataset:
        predicted = should_follow_up(
            has_follow_ups=bool(sample.get("has_follow_ups", True)),
            answer_text=sample["answer"],
            depth_score=int(sample["depth_score"]),
            evidence_score=int(sample["evidence_score"]),
            overall_score=float(sample["overall_score"]),
        )
        records.append(
            {
                "name": sample["name"],
                "expected_follow_up": bool(sample["expected_follow_up"]),
                "predicted_follow_up": predicted,
                "matched": predicted is bool(sample["expected_follow_up"]),
            }
        )
    return _finalize_suite("interview_policy", records)


def run_review_eval(
    dataset: list[dict[str, Any]],
    *,
    judge_mode: str = "auto",
) -> dict[str, Any]:
    records = []
    judge_fn, mode_used = _build_review_judge(judge_mode, dataset)
    for sample in dataset:
        result = evaluate_against_reference(
            question=sample["question"],
            expected_answer=sample["expected_answer"],
            actual_answer=sample["actual_answer"],
            pitfalls=list(sample.get("pitfalls", [])),
            judge_fn=judge_fn,
        )
        expected_min = int(sample.get("expected_min_accuracy", 1))
        records.append(
            {
                "question": sample["question"],
                "expected_min_accuracy": expected_min,
                "predicted_accuracy": int(result["accuracy_score"]),
                "matched": int(result["accuracy_score"]) >= expected_min,
                "reason": result["reason"],
            }
        )
    suite = _finalize_suite("review", records)
    suite["judge_mode"] = mode_used
    return suite


def run_suite(name: str, *, judge_mode: str = "auto") -> dict[str, Any]:
    dataset = load_dataset(DEFAULT_DATASETS[name])
    if name == "ingest":
        return run_ingest_eval(dataset)
    if name == "cluster":
        return run_cluster_eval(dataset)
    if name == "interview_policy":
        return run_interview_policy_eval(dataset)
    if name == "review":
        return run_review_eval(dataset, judge_mode=judge_mode)
    raise ValueError(f"Unknown eval suite: {name}")


def run_all_suites(*, judge_mode: str = "auto") -> dict[str, Any]:
    suites = [run_suite(name, judge_mode=judge_mode) for name in DEFAULT_DATASETS]
    total = sum(item["summary"]["total"] for item in suites)
    passed = sum(item["summary"]["passed"] for item in suites)
    return {
        "suites": suites,
        "summary": {
            "total": total,
            "passed": passed,
            "accuracy": round(passed / total, 3) if total else 0.0,
        },
    }


def print_report(report: dict[str, Any]) -> None:
    if "suites" in report:
        print("Interview Harness Eval Report")
        for suite in report["suites"]:
            _print_suite(suite)
        print("=" * 40)
        print(
            f"overall: {report['summary']['passed']}/{report['summary']['total']} "
            f"({report['summary']['accuracy']:.1%})"
        )
        return
    _print_suite(report)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run focused evaluation suites for Interview Copilot.")
    parser.add_argument(
        "--suite",
        choices=["all", *DEFAULT_DATASETS.keys()],
        default="all",
        help="Evaluation suite to run.",
    )
    parser.add_argument(
        "--judge-mode",
        choices=["auto", "local", "model"],
        default="auto",
        help="Judge mode for review_eval.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of human-readable text.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = (
        run_all_suites(judge_mode=args.judge_mode)
        if args.suite == "all"
        else run_suite(args.suite, judge_mode=args.judge_mode)
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
    return 0


def _make_eval_page(sample: dict[str, Any]) -> FetchedPage:
    url = sample.get("url", "https://www.nowcoder.com/discuss/eval")
    return FetchedPage(
        hit=SearchHit(query="eval", title=sample["title"], url=url, snippet=sample.get("snippet", "")),
        canonical_url=url,
        title=sample["title"],
        text=sample["text"],
        updated_at=sample.get("updated_at", "2099-01-01T00:00:00"),
        like_count=int(sample.get("like_count", 0)),
        comment_count=int(sample.get("comment_count", 0)),
        view_count=int(sample.get("view_count", 0)),
    )


def _build_review_judge(judge_mode: str, dataset: list[dict[str, Any]]) -> tuple[JudgeFn, str]:
    if judge_mode == "local":
        return _local_review_judge(dataset), "local"
    if judge_mode == "model":
        return lambda prompt: call_text_model(prompt, task="judge"), "model"

    settings = get_text_settings("judge")
    if settings.get("api_key"):
        return lambda prompt: call_text_model(prompt, task="judge"), "model"
    return _local_review_judge(dataset), "local"


def _local_review_judge(dataset: list[dict[str, Any]]) -> JudgeFn:
    lookup = {sample["question"]: sample for sample in dataset}

    def _judge(prompt: str) -> str:
        question = _extract_between(prompt, "- 问题：", "- 参考答案：")
        answer = _extract_after(prompt, "- 候选人回答：")
        sample = lookup.get(question, {})
        keywords = [str(item).lower() for item in sample.get("expected_keywords", [])]
        lowered = answer.lower()
        hit_count = sum(1 for keyword in keywords if keyword and keyword in lowered)
        ratio = hit_count / max(len(keywords), 1)
        if ratio >= 0.8:
            score = 5
        elif ratio >= 0.6:
            score = 4
        elif ratio >= 0.4:
            score = 3
        elif ratio >= 0.2:
            score = 2
        else:
            score = 1
        payload = {
            "accuracy_score": score,
            "clarity_score": max(1, min(5, score)),
            "depth_score": max(1, min(5, score)),
            "evidence_score": max(1, min(5, score - (0 if len(answer) >= 40 else 1))),
            "structure_score": max(1, min(5, score)),
            "reason": f"local judge matched {hit_count}/{max(len(keywords), 1)} expected keywords",
        }
        return json.dumps(payload, ensure_ascii=False)

    return _judge


def _extract_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    end = text.find(end_marker, start + len(start_marker))
    if start < 0 or end < 0:
        return ""
    return text[start + len(start_marker):end].strip()


def _extract_after(text: str, marker: str) -> str:
    start = text.find(marker)
    return text[start + len(marker):].strip() if start >= 0 else ""


def _finalize_suite(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    passed = sum(1 for record in records if record["matched"])
    return {
        "suite": name,
        "summary": {
            "total": total,
            "passed": passed,
            "accuracy": round(passed / total, 3) if total else 0.0,
        },
        "records": records,
    }


def _print_suite(suite: dict[str, Any]) -> None:
    summary = suite["summary"]
    extra = f" | judge={suite['judge_mode']}" if "judge_mode" in suite else ""
    print("=" * 40)
    print(f"{suite['suite']}{extra}")
    print(f"passed: {summary['passed']}/{summary['total']} ({summary['accuracy']:.1%})")
    for record in suite["records"][:5]:
        label = next((record.get(key) for key in ("title", "question", "name") if record.get(key)), "sample")
        print(f"- {label}: {'ok' if record['matched'] else 'fail'}")


if __name__ == "__main__":
    raise SystemExit(main())
