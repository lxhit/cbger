from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import torch

from .io import read_jsonl, write_jsonl

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")


def _tokens(value: object) -> set[str]:
    if isinstance(value, list):
        return {token.lower() for item in value for token in TOKEN_RE.findall(str(item))}
    return {token.lower() for token in TOKEN_RE.findall(str(value or ""))}


def _v05_id(sample_id: str) -> str:
    return sample_id.replace("pbger_v0_4_", "pbger_v0_5_", 1)


def _normalise(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(1e-8)


def _profile_vector(profile: dict) -> torch.Tensor:
    slots = profile["slots"]
    features = torch.tensor([slot["feature"] for slot in slots], dtype=torch.float32)
    confidence = torch.tensor([slot["confidence"] for slot in slots], dtype=torch.float32)
    confidence = confidence / confidence.sum().clamp_min(1e-8)
    return _normalise((features * confidence.unsqueeze(-1)).sum(dim=0))


def _relocate_evidence(sample: dict, target_index: int) -> dict:
    result = copy.deepcopy(sample)
    ordered = sorted(result["timeline"], key=lambda row: row["target_interval"][0])
    durations = {
        entry["segment_id"]: float(entry["target_interval"][1])
        - float(entry["target_interval"][0])
        for entry in ordered
    }
    evidence_index = next(i for i, entry in enumerate(ordered) if entry["role"] == "evidence")
    evidence = ordered.pop(evidence_index)
    ordered.insert(target_index, evidence)
    cursor = 0.0
    for entry in ordered:
        duration = durations[entry["segment_id"]]
        entry["target_interval"] = [round(cursor, 6), round(cursor + duration, 6)]
        cursor += duration
    result["timeline"] = ordered
    result["evidence_segments"] = [
        entry["target_interval"][:] for entry in ordered if entry["role"] == "evidence"
    ]
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_v05(
    dataset: Path,
    profiles_path: Path,
    profile_features_path: Path,
    item_features_path: Path,
    segment_paths: list[Path],
    out: Path,
    position_out: Path,
    profile_out: Path,
    report_path: Path,
    freeze_path: Path,
    profile_margin: float = 2.0,
) -> dict:
    """Create v0.5 and deterministic position/profile counterfactual diagnostics."""
    rows = list(read_jsonl(dataset))
    profiles_raw = {str(row["user_id"]): row for row in read_jsonl(profiles_path)}
    profile_features = {
        str(row["user_id"]): row for row in read_jsonl(profile_features_path)
    }
    item_features = {
        row["segment_id"]: _normalise(torch.tensor(row["feature"], dtype=torch.float32))
        for row in read_jsonl(item_features_path)
    }
    segment_semantics = {
        row["segment_id"]: _tokens(row.get("semantic_tags", []))
        | _tokens(row.get("attribution", ""))
        for path in segment_paths
        for row in read_jsonl(path)
    }
    user_interest_tokens = {
        user_id: {
            token
            for interest in profile.get("interests", [])
            for token in (
                _tokens(interest.get("tags", []))
                | _tokens(interest.get("name", ""))
                | _tokens(interest.get("phrase", ""))
            )
        }
        for user_id, profile in profiles_raw.items()
    }

    base: list[dict] = []
    for row in rows:
        converted = copy.deepcopy(row)
        converted["sample_id"] = _v05_id(row["sample_id"])
        if converted.get("counterfactual_of"):
            converted["counterfactual_of"] = _v05_id(converted["counterfactual_of"])
        converted["provenance"] = {
            **converted.get("provenance", {}),
            "dataset_version": "0.5",
            "derived_from": row["sample_id"],
        }
        base.append(converted)
    write_jsonl(out, base)

    positives = [row for row in base if row["relevance"] == 1 and row["split"] == "test"]
    positions: list[dict] = []
    position_names = ("beginning", "middle", "end")
    for positive in positives:
        count = len(positive["timeline"])
        indexes = (0, count // 2, count - 1)
        group = f"position::{positive['sample_id']}"
        for name, target_index in zip(position_names, indexes):
            variant = _relocate_evidence(positive, target_index)
            variant["sample_id"] = f"{positive['sample_id']}__position_{name}"
            variant["diagnostic_group_id"] = group
            variant["provenance"] = {
                **variant["provenance"],
                "diagnostic": "position_swap",
                "base_sample_id": positive["sample_id"],
                "target_position": name,
                "label_type": "Gold-P",
            }
            positions.append(variant)
    write_jsonl(position_out, positions)

    user_ids = sorted(set(profile_features) & set(profiles_raw))
    profile_swaps: list[dict] = []
    counts: Counter[str] = Counter()
    for positive in positives:
        original_user = str(positive["user_id"])
        if original_user not in user_interest_tokens:
            counts["missing_original_profile"] += 1
            continue
        candidate_ids = [entry["segment_id"] for entry in positive["timeline"]]
        if any(segment_id not in item_features for segment_id in candidate_ids):
            counts["missing_item_feature"] += 1
            continue
        evidence_index = next(
            i for i, entry in enumerate(positive["timeline"]) if entry["role"] == "evidence"
        )
        candidate_tokens = [
            segment_semantics.get(segment_id, set()) | _tokens(entry.get("attribution", ""))
            for segment_id, entry in zip(candidate_ids, positive["timeline"])
        ]
        original_tokens = user_interest_tokens[original_user]
        original_scores = [len(original_tokens & tokens) for tokens in candidate_tokens]
        selected: tuple[float, str, int, int, int] | None = None
        for swapped_user in user_ids:
            if swapped_user == original_user:
                continue
            interest_tokens = user_interest_tokens.get(swapped_user, set())
            if not interest_tokens:
                continue
            scores = [len(interest_tokens & tokens) for tokens in candidate_tokens]
            distractor_indexes = [i for i in range(len(scores)) if i != evidence_index]
            new_evidence_index = max(distractor_indexes, key=lambda i: scores[i])
            new_margin = scores[new_evidence_index] - scores[evidence_index]
            original_margin = original_scores[evidence_index] - original_scores[new_evidence_index]
            if new_margin < int(profile_margin) or original_margin < 1:
                continue
            quality = 2.0 * new_margin + original_margin
            candidate = (quality, swapped_user, new_evidence_index, new_margin, original_margin)
            if selected is None or candidate > selected:
                selected = candidate
        if selected is None:
            counts["no_profile_swap"] += 1
            continue
        _, swapped_user, new_evidence_index, new_margin, original_margin = selected
        variant = copy.deepcopy(positive)
        variant["sample_id"] = f"{positive['sample_id']}__profile_{swapped_user}"
        variant["user_id"] = swapped_user
        variant["history_ids"] = profiles_raw[swapped_user]["history_ids"]
        for index, entry in enumerate(variant["timeline"]):
            if index == evidence_index:
                entry["role"] = "profile_counterfactual_distractor"
            elif index == new_evidence_index:
                entry["role"] = "evidence"
        variant["evidence_segments"] = [
            variant["timeline"][new_evidence_index]["target_interval"][:]
        ]
        variant["diagnostic_group_id"] = f"profile::{positive['sample_id']}"
        variant["provenance"] = {
            **variant["provenance"],
            "diagnostic": "profile_swap",
            "base_sample_id": positive["sample_id"],
            "original_user_id": original_user,
            "swapped_user_id": swapped_user,
            "original_evidence_segment_id": candidate_ids[evidence_index],
            "swapped_evidence_segment_id": candidate_ids[new_evidence_index],
            "profile_swap_margin": new_margin,
            "original_preference_margin": original_margin,
            "label_type": "Bronze-CF",
            "construction": "behavior_tag_segment_semantic_minimal_intervention",
        }
        profile_swaps.append(variant)
        counts["profile_swap"] += 1
    write_jsonl(profile_out, profile_swaps)

    report = {
        "version": "0.5",
        "base_records": len(base),
        "base_pairs": len(base) // 2,
        "test_positives": len(positives),
        "position_swap_records": len(positions),
        "position_groups": len(positions) // 3,
        "profile_swap_records": len(profile_swaps),
        "profile_swap_coverage": len(profile_swaps) / max(1, len(positives)),
        "profile_margin": profile_margin,
        "profile_swap_teacher": "behavior tags plus independent segment semantics; no CLIP",
        "counts": dict(counts),
        "limitations": [
            "ProfileSwap is Bronze-CF weak supervision from lexical-semantic gates.",
            "PositionSwap is a virtual-timeline intervention, not a re-encoded physical video.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    jsonl_paths = (out, position_out, profile_out)
    freeze = {
        "version": "0.5",
        "artifacts": {
            str(path): {"sha256": _sha256(path), "records": sum(1 for _ in read_jsonl(path))}
            for path in jsonl_paths
        },
        "policy": "v0.5 diagnostics are frozen; threshold changes require a new version",
    }
    freeze["artifacts"][str(report_path)] = {"sha256": _sha256(report_path), "records": 1}
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    return report
