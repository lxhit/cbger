#!/usr/bin/env python3
"""Build history-supported Bronze-R profiles from MicroLens CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from cbger.profiles import build_bootstrap_profiles
from cbger.profiles_v2 import audit_profiles_v2, build_profiles_v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--titles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-output", type=Path, default=None)
    parser.add_argument("--users", type=int, default=100_000)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--top-tags", type=int, default=20)
    parser.add_argument("--max-interests", type=int, default=12)
    args = parser.parse_args()

    bootstrap = args.bootstrap_output or args.output.with_name("bootstrap_profiles.jsonl")
    build_bootstrap_profiles(
        args.pairs, args.titles, bootstrap, args.users, args.min_history,
        args.max_history, args.top_tags,
    )
    build_profiles_v2(bootstrap, args.titles, args.output, args.max_interests)
    report = audit_profiles_v2(args.output, args.output.with_suffix(".audit.json"))
    print(report)


if __name__ == "__main__":
    main()
