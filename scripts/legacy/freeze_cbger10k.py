#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from cbger.io import read_jsonl, sha256_file, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v07-dataset", type=Path, required=True)
    ap.add_argument("--v07-profiles", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    source = list(read_jsonl(a.v07_dataset))
    id_map = {r["sample_id"]: r["sample_id"].replace("pbger_v0_7_", "pbger_v0_8_") for r in source}
    rows = []
    for old in source:
        row = dict(old); row["sample_id"] = id_map[old["sample_id"]]
        row["counterfactual_of"] = id_map[old["counterfactual_of"]] if old.get("counterfactual_of") else None
        provenance = dict(old.get("provenance", {})); provenance.pop("interest_tags", None)
        provenance.update({"dataset_version":"PBGER-v0.8","derived_from":"PBGER-v0.7 Qwen behavior",
            "task_axes":["where","intervention","whether"],"profile_swap_in_scope":False,
            "profile_representation":"single_global_behavior_profile"})
        row["provenance"] = provenance; rows.append(row)
    users = {str(r["user_id"]) for r in rows}; profiles = []
    for p in read_jsonl(a.v07_profiles):
        uid = str(p["user_id"])
        if uid not in users: continue
        slots = [s for s in p["slots"] if s.get("source") == "concatenated_profile_clip_text"]
        if len(slots) != 1: raise RuntimeError(f"user {uid}: expected one global slot, got {len(slots)}")
        profiles.append({"user_id":uid,"profile_version":"PBGER-v0.8-global","slots":[{
            "type":"global_behavior","type_id":0,"text":"global behavior profile","confidence":1.0,
            "feature":slots[0]["feature"],"source":"history_supported_global_clip_text"}]})
    if {p["user_id"] for p in profiles} != users: raise RuntimeError("incomplete v0.8 profiles")
    dataset=a.output_dir/"pbger_v0_8.jsonl"; profile_path=a.output_dir/"pbger_v0_8_global_profiles.jsonl"
    write_jsonl(dataset, rows); write_jsonl(profile_path, profiles)
    counts=Counter((r["split"],r["relevance"]) for r in rows)
    report={"version":"PBGER-v0.8","records":len(rows),"pairs":sum(r["relevance"]==1 for r in rows),
        "users":len(users),"task_axes":["where","intervention","whether"],"profile_swap_removed":True,
        "profile_schema":"one history-supported global behavior vector per user",
        "split_relevance_counts":{f"{k[0]}:{k[1]}":v for k,v in counts.items()},
        "artifacts":{str(dataset):{"sha256":sha256_file(dataset),"records":len(rows)},
            str(profile_path):{"sha256":sha256_file(profile_path),"records":len(profiles)}}}
    (a.output_dir/"pbger_v0_8_build_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
