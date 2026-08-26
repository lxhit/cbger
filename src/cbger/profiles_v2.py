from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .io import read_jsonl, write_jsonl

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
GENERIC = set(ENGLISH_STOP_WORDS) | {
    "animation",
    "bad",
    "beautiful",
    "big",
    "day",
    "episode",
    "film",
    "funny",
    "good",
    "life",
    "like",
    "man",
    "movie",
    "new",
    "people",
    "recommendation",
    "today",
    "video",
    "world",
}


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token.lower() not in GENERIC
    ]


def _phrases(text: str) -> list[str]:
    tokens = _tokens(text)
    return [" ".join(tokens[index : index + 2]) for index in range(len(tokens) - 1)]


def _title_map(path: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            item = row.get("item") or row.get("videoID")
            if item:
                titles[item] = row.get("title") or ""
    return titles


def build_profiles_v2(
    bronze_profiles: Path,
    titles_path: Path,
    out_path: Path,
    max_interests: int = 12,
) -> int:
    titles = _title_map(titles_path)
    terms_by_item = {
        item: set(_tokens(title) + _phrases(title)) for item, title in titles.items()
    }
    title_documents = list(terms_by_item.values())
    document_frequency: Counter[str] = Counter(
        term for document in title_documents for term in document
    )
    total_documents = max(1, len(title_documents))
    source_profiles = list(read_jsonl(bronze_profiles))
    profile_frequency: Counter[str] = Counter()
    for profile in source_profiles:
        terms = {
            term
            for item in profile["history_ids"]
            for term in terms_by_item.get(str(item), set())
        }
        profile_frequency.update(terms)
    popular_terms = {
        term
        for term, count in profile_frequency.items()
        if len(source_profiles) >= 100 and count / len(source_profiles) > 0.10
    }

    records: list[dict] = []
    for profile in source_profiles:
        history = [str(item) for item in profile["history_ids"]]
        support: dict[str, list[str]] = defaultdict(list)
        recent_support: dict[str, list[str]] = defaultdict(list)
        old_support: dict[str, list[str]] = defaultdict(list)
        recent_start = max(0, len(history) - max(5, math.ceil(len(history) / 3)))
        for index, item in enumerate(history):
            terms = {
                term
                for term in terms_by_item.get(item, set())
                if term not in popular_terms
                and not any(part in popular_terms for part in term.split())
            }
            for term in terms:
                support[term].append(item)
                if index >= recent_start:
                    recent_support[term].append(item)
                else:
                    old_support[term].append(item)

        candidates: list[tuple[float, str, str, list[str]]] = []
        for term, items in support.items():
            count = len(items)
            idf = math.log((total_documents + 1) / (document_frequency[term] + 1)) + 1
            phrase_bonus = 1.25 if " " in term else 1.0
            if count >= 3:
                candidates.append((count * idf * phrase_bonus, "long_term", term, items))
            if len(recent_support[term]) >= 2:
                kind = "emerging" if not old_support[term] else "short_term"
                score = len(recent_support[term]) * idf * phrase_bonus * 1.15
                candidates.append((score, kind, term, recent_support[term]))

        chosen: list[dict] = []
        seen_terms: set[str] = set()
        for score, kind, term, items in sorted(candidates, reverse=True):
            if term in seen_terms:
                continue
            seen_terms.add(term)
            confidence = min(0.99, 0.45 + 0.08 * len(items) + 0.03 * min(score, 8))
            chosen.append(
                {
                    "name": term,
                    "type": kind,
                    "tags": term.split(),
                    "phrase": term if " " in term else None,
                    "confidence": round(confidence, 4),
                    "support_video_ids": sorted(set(items)),
                    "support_count": len(set(items)),
                }
            )
            if len(chosen) >= max_interests:
                break
        if not chosen:
            continue
        records.append(
            {
                "profile_id": f"microlens_{profile['user_id']}_v2",
                "profile_version": "2.0",
                "user_id": str(profile["user_id"]),
                "history_ids": history,
                "interests": chosen,
                "provenance": {
                    "label_type": "Bronze-R",
                    "method": "idf_recency_support_grounded",
                    "source_profile": profile.get("provenance"),
                    "titles_snapshot": titles_path.name,
                    "max_cross_user_frequency": 0.10,
                    "removed_popular_terms": len(popular_terms),
                },
            }
        )
    return write_jsonl(out_path, records)


def audit_profiles_v2(path: Path, out: Path | None = None) -> dict:
    profiles = list(read_jsonl(path))
    type_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    support_errors: list[str] = []
    for profile in profiles:
        history = set(profile["history_ids"])
        for interest in profile["interests"]:
            type_counts[interest["type"]] += 1
            tag_counts.update(interest["tags"])
            support_ids = set(interest["support_video_ids"])
            if not support_ids or not support_ids <= history:
                support_errors.append(f"{profile['profile_id']}:{interest['name']}")
    report = {
        "valid": not support_errors,
        "profiles": len(profiles),
        "users": len({profile["user_id"] for profile in profiles}),
        "interests": sum(len(profile["interests"]) for profile in profiles),
        "interest_types": dict(type_counts),
        "top_tags": tag_counts.most_common(30),
        "support_errors": support_errors[:100],
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
