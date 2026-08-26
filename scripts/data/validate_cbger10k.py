#!/usr/bin/env python3
import argparse
from pathlib import Path
from cbger.io import read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", type=Path, default=Path("data/cbger10k/cbger10k.jsonl"))
ap.add_argument(
    "--profiles", type=Path,
    default=Path("data/cbger10k/global_profiles_clip_vit_b32.jsonl"),
)
args = ap.parse_args()
d = list(read_jsonl(args.dataset))
p = list(read_jsonl(args.profiles))
positive = {r["sample_id"] for r in d if r["relevance"] == 1}
counterfactual = [r for r in d if r["relevance"] == 0]
assert all(r["counterfactual_of"] in positive for r in counterfactual)
assert all(sum(x["role"] == "evidence" for x in r["timeline"]) == 1
           for r in d if r["relevance"] == 1)
assert all(len(r["slots"]) == 1
           and r["slots"][0]["type"] == "global_behavior"
           and len(r["slots"][0]["feature"]) == 512 for r in p)
assert len(d) == 10_000 and len(positive) == 5_000 and len(p) == 3_026
assert all(len(row["timeline"]) == 9 for row in d)
print({"dataset": "CBGER-10K", "records": len(d), "pairs": len(positive),
       "profiles": len(p), "validation": "PASS"})
