"""Run the daily copilot workflow end to end."""

from __future__ import annotations

import argparse
import asyncio
import json

from copilot.app import run_agent_workflow


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the daily interview-copilot workflow.")
    parser.add_argument("--days", type=int, default=7, help="Only keep materials updated within this many days.")
    parser.add_argument("--count-per-query", type=int, default=20)
    parser.add_argument("--max-reports", type=int, default=50)
    parser.add_argument("--fetch-timeout", type=float, default=12.0)
    parser.add_argument("--with-web", action="store_true")
    parser.add_argument("--max-cards", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = await run_agent_workflow(
        updated_within_days=args.days,
        count_per_query=args.count_per_query,
        max_reports=args.max_reports,
        fetch_timeout=args.fetch_timeout,
        with_web=args.with_web,
        max_cards=args.max_cards,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
