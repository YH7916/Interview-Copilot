"""Simple entry point for the local daily digest."""

from __future__ import annotations

import argparse

from copilot.app import render_daily_digest


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a daily digest from local interview reports.")
    parser.add_argument("--days", type=int, default=1, help="Look back this many days.")
    args = parser.parse_args()
    print(render_daily_digest(days=args.days))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
