"""Simple entry point for recent Nowcoder ingest."""

from __future__ import annotations

import asyncio
import json

from copilot.app import collect_nowcoder_interviews


async def main() -> int:
    result = await collect_nowcoder_interviews(
        count_per_query=30,
        max_reports=200,
        dry_run=False,
        fetch_timeout=12.0,
        updated_within_days=7,
        rebuild_index=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
