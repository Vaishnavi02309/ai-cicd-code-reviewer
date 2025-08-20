"""
CLI for Git Insights.
Example:
  python -m project.cli --limit 100 --out insights.json
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from .git_insights import collect_summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate a small JSON report from recent git history.")
    p.add_argument("--limit", type=int, default=50, help="How many recent commits to scan.")
    p.add_argument("--out", type=str, default="", help="Optional output file (JSON).")
    args = p.parse_args(argv)

    try:
        summary = collect_summary(limit=args.limit)
        payload = summary.to_json()
        if args.out:
            Path(args.out).write_text(payload, encoding="utf-8")
            print(f"Wrote {args.out}")
        else:
            print(payload)
        return 0
    except Exception as exc:
        print(f"[git-insights] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
