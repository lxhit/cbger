from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .io import write_jsonl

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "and", "are", "bad", "beautiful", "for", "from", "has", "have", "into",
    "not", "that", "the", "this", "three", "two", "video", "with", "you", "your",
    "day", "new", "recommendation",
}


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token.lower() not in STOPWORDS
    ]


def build_bootstrap_profiles(
    pairs_path: Path,
    titles_path: Path,
    out_path: Path,
    users: int,
    min_history: int,
    max_history: int,
    top_tags: int,
) -> int:
    titles: dict[str, str] = {}
    with titles_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            item = row.get("item") or row.get("videoID")
            title = row.get("title") or ""
            if item:
                titles[item] = title

    histories: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with pairs_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            user = row.get("user") or row.get("userID")
            item = row.get("item") or row.get("videoID")
            timestamp = row.get("timestamp")
            if user and item and timestamp:
                histories[user].append((int(timestamp), item))

    eligible = [
        (user, sorted(events))
        for user, events in histories.items()
        if len(events) >= min_history
    ]
    eligible.sort(key=lambda pair: (-len(pair[1]), pair[0]))
    records: list[dict] = []
    for user, events in eligible[:users]:
        history_ids = [item for _, item in events[-max_history:]]
        tag_counts: Counter[str] = Counter()
        support: dict[str, list[str]] = defaultdict(list)
        for item in history_ids:
            for token in set(_tokens(titles.get(item, ""))):
                tag_counts[token] += 1
                support[token].append(item)
        tags = [
            tag
            for tag, count in tag_counts.most_common(top_tags)
            if count >= 2
        ]
        if not tags:
            continue
        records.append(
            {
                "user_id": user,
                "history_ids": history_ids,
                "interests": [
                    {
                        "tags": tags,
                        "support_video_ids": sorted(
                            {item for tag in tags for item in support[tag]}
                        ),
                    }
                ],
                "provenance": {
                    "label_type": "Bronze",
                    "method": "title_token_frequency_bootstrap",
                },
            }
        )
    return write_jsonl(out_path, records)
