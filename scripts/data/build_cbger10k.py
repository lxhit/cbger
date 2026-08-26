#!/usr/bin/env python3
"""Build PBGER-v0.7 without using CLIP or the evaluated PBGER backbone.

The builder matches MicroLens behavior-grounded interest profiles to structured
Qwen3-VL event annotations with an independent BGE-M3 text encoder.  It then
creates paired factual/counterfactual virtual timelines.
"""
from __future__ import annotations

import argparse, hashlib, json, math, random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ann_text(a):
    events = "; ".join(x.get("event", "") for x in a.get("event_sequence", []))
    return (
        "Topics: " + ", ".join(a.get("topics", [])) + ". Objects: "
        + ", ".join(a.get("objects", [])) + ". Actions: "
        + ", ".join(a.get("actions", [])) + ". Ordered events: " + events
        + ". Outcome: " + str(a.get("outcome", ""))
    )


def stable_bucket(value):
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 100


def user_split(user):
    b = stable_bucket(str(user))
    return "train" if b < 70 else ("validation" if b < 80 else "test")


def profile_text(p):
    interests = sorted(p.get("interests", []), key=lambda x: (-float(x.get("confidence", 0)), x.get("name", "")))
    chunks = []
    for x in interests[:12]:
        name = str(x.get("phrase") or x.get("name") or "").strip()
        if not name:
            continue
        support = max(1, int(x.get("support_count", 1)))
        confidence = float(x.get("confidence", .5))
        kind = str(x.get("type", "stable"))
        repeat = min(3, max(1, round(confidence * math.log2(support + 1) * 2)))
        chunks.extend([f"{kind} interest: {name}"] * repeat)
    return ". ".join(chunks), [str(x.get("name", "")) for x in interests[:12]]


def timeline_entry(seg, role, target_start):
    duration = max(.25, float(seg["end"]) - float(seg["start"]))
    return {
        "attribution": seg.get("attribution", ""), "role": role,
        "segment_id": seg["segment_id"], "source_id": seg["source_id"],
        "source_interval": [float(seg["start"]), float(seg["end"])],
        "target_interval": [round(target_start, 3), round(target_start + duration, 3)],
    }, target_start + duration


def build_pair(split, pair_no, user, profile, evidence_i, replacement_i, distractor_idx,
               segments, texts, scores, seed):
    rng = random.Random(seed + pair_no + stable_bucket(user))
    selected = list(dict.fromkeys(distractor_idx))[:8]
    pool = [i for i in range(len(segments)) if i not in selected and i not in {evidence_i, replacement_i}]
    while len(selected) < 8 and pool:
        selected.append(pool.pop(rng.randrange(len(pool))))
    insert_at = rng.randrange(len(selected) + 1)

    def make_timeline(focal_i, focal_role):
        order = selected[:]
        order.insert(insert_at, focal_i)
        out, t, evidence_interval = [], 0.0, None
        for i in order:
            role = focal_role if i == focal_i else "hard_distractor"
            entry, t = timeline_entry(segments[i], role, t)
            if role == "evidence": evidence_interval = entry["target_interval"]
            out.append(entry)
        return out, ([evidence_interval] if evidence_interval else [])

    factual_id = f"pbger_v0_7_{split}_{pair_no*2:06d}"
    cf_id = f"pbger_v0_7_{split}_{pair_no*2+1:06d}"
    factual_tl, evidence = make_timeline(evidence_i, "evidence")
    cf_tl, _ = make_timeline(replacement_i, "hard_distractor")
    common = {
        "split": split, "track": "behavior_grounded_sparse_evidence",
        "user_id": user, "history_ids": profile.get("history_ids", []),
        "difficulty": "behavior_mismatch_content_neighbor",
    }
    prov = {
        "dataset_version": "PBGER-v0.7", "construction": "qwen3vl_bge_behavior_virtual_timeline",
        "visual_teacher": "Qwen3-VL-8B-Instruct BF16 4FPS", "matching_encoder": "BAAI/bge-m3",
        "evaluated_backbone_used_for_labels": False, "profile_label_type": profile.get("provenance", {}).get("label_type", "Bronze-R"),
        "interest_tags": profile_text(profile)[1], "positive_score": round(float(scores[evidence_i]), 6),
        "replacement_score": round(float(scores[replacement_i]), 6),
        "behavior_gap": round(float(scores[evidence_i] - scores[replacement_i]), 6), "seed": seed,
    }
    factual = {**common, "sample_id": factual_id, "counterfactual_of": None, "relevance": 1,
               "timeline": factual_tl, "evidence_segments": evidence,
               "provenance": {**prov, "label_type": "Silver-QB"}}
    cf = {**common, "sample_id": cf_id, "counterfactual_of": factual_id, "relevance": 0,
          "timeline": cf_tl, "evidence_segments": [],
          "provenance": {**prov, "label_type": "Silver-CF", "operation": "replace_behavior_evidence_with_content_neighbor",
                         "removed_segment_id": segments[evidence_i]["segment_id"],
                         "replacement_segment_id": segments[replacement_i]["segment_id"]}}
    return factual, cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing data/interim inputs.",
    )
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--pairs", default="3500,500,1000")
    ap.add_argument("--min-positive", type=float, default=.30)
    ap.add_argument("--min-gap", type=float, default=.06)
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = args.output_dir or root / "data/interim/cbger10k_build"
    out_dir.mkdir(parents=True, exist_ok=True)
    annotations = {}
    ann_path = root / "data/interim/v0_7/qwen3_vl_segment_annotations_4fps_bf16.jsonl"
    for r in read_jsonl(ann_path):
        if r.get("ok") and r.get("annotation") and float(r["annotation"].get("uncertainty", 1)) <= .5:
            annotations[str(r["segment_id"])] = r["annotation"]
    profiles = list(read_jsonl(root / "data/interim/user_profiles_v2.jsonl"))
    model = SentenceTransformer(args.model, device="cuda")
    target_pairs = dict(zip(("train", "validation", "test"), map(int, args.pairs.split(","))))
    report = {"version": "0.7", "seed": args.seed, "annotations_valid": len(annotations), "splits": {},
              "leakage_control": {"clip_used_in_construction": False, "pbger_backbone_used_in_construction": False}}
    all_source_sets = {}
    for split in ("train", "validation", "test"):
        seg_rows = [r for r in read_jsonl(root / f"data/interim/v0_6/segments_{split}.jsonl") if r["segment_id"] in annotations]
        seg_texts = [ann_text(annotations[r["segment_id"]]) for r in seg_rows]
        seg_emb = model.encode(seg_texts, batch_size=args.batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=True).astype("float32")
        split_profiles = [p for p in profiles if user_split(str(p["user_id"])) == split and profile_text(p)[0]]
        rng = random.Random(args.seed + {"train": 0, "validation": 1, "test": 2}[split])
        rng.shuffle(split_profiles)
        rows, used_evidence, failures = [], Counter(), Counter()
        used_by_user = defaultdict(set)
        rounds = max(1, math.ceil(target_pairs[split] / max(1, len(split_profiles))))
        profile_queue = [p for round_no in range(rounds) for p in split_profiles]
        for p in profile_queue:
            if len(rows) >= target_pairs[split] * 2: break
            ptext, _ = profile_text(p)
            pvec = model.encode([ptext], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32")
            us = seg_emb @ pvec
            ranked = np.argsort(-us)
            uid = str(p["user_id"])
            evidence_i = next((int(i) for i in ranked[:500]
                               if used_evidence[int(i)] < 3 and int(i) not in used_by_user[uid]), None)
            if evidence_i is None or us[evidence_i] < args.min_positive:
                failures["low_positive"] += 1; continue
            content = seg_emb @ seg_emb[evidence_i]
            neighbors = np.argsort(-content)
            replacement_i = next((int(i) for i in neighbors[1:300]
                                  if us[evidence_i] - us[int(i)] >= args.min_gap and content[int(i)] >= .45), None)
            if replacement_i is None:
                failures["no_behavior_mismatched_neighbor"] += 1; continue
            distractors = [int(i) for i in neighbors[1:500] if int(i) not in {evidence_i, replacement_i}
                           and us[int(i)] <= us[evidence_i] - args.min_gap / 2]
            factual, cf = build_pair(split, len(rows)//2, str(p["user_id"]), p, evidence_i,
                                     replacement_i, distractors, seg_rows, seg_texts, us, args.seed)
            rows.extend([factual, cf]); used_evidence[evidence_i] += 1
            used_by_user[uid].add(evidence_i)
        out_path = out_dir / f"pbger_v0_7_{split}.jsonl"
        write_jsonl(out_path, rows)
        np.savez_compressed(out_dir / f"bge_m3_{split}_segment_embeddings.npz",
                            segment_ids=np.array([r["segment_id"] for r in seg_rows]), embeddings=seg_emb)
        sources = {r["source_id"] for r in seg_rows}; all_source_sets[split] = sources
        report["splits"][split] = {"records": len(rows), "pairs": len(rows)//2, "segments_available": len(seg_rows),
                                    "users_available": len(split_profiles),
                                    "unique_users_used": len({r["user_id"] for r in rows}),
                                    "pairs_per_user_max": max(Counter(r["user_id"] for r in rows[::2]).values(), default=0),
                                    "failures": dict(failures), "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest()}
    report["source_overlap"] = {"train_validation": len(all_source_sets["train"] & all_source_sets["validation"]),
                                "train_test": len(all_source_sets["train"] & all_source_sets["test"]),
                                "validation_test": len(all_source_sets["validation"] & all_source_sets["test"])}
    with open(out_dir / "build_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
