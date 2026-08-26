#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from cbger.io import read_jsonl, write_jsonl

ap=argparse.ArgumentParser(); ap.add_argument("--dataset",type=Path,required=True)
ap.add_argument("--profiles",type=Path,required=True); ap.add_argument("--features",type=Path,required=True)
ap.add_argument("--output",type=Path,required=True); ap.add_argument("--name",required=True); a=ap.parse_args()
profiles={str(x["user_id"]):x for x in read_jsonl(a.profiles)}
features={x["segment_id"]:torch.tensor(x["feature"]) for x in read_jsonl(a.features)}
out=[]
for sample in read_jsonl(a.dataset):
    slots=profiles[str(sample["user_id"])]["slots"]
    confidence=torch.tensor([s["confidence"] for s in slots]); confidence=confidence/confidence.sum()
    user=F.normalize((torch.tensor([s["feature"] for s in slots])*confidence[:,None]).sum(0),dim=0)
    items=F.normalize(torch.stack([features[x["segment_id"]] for x in sample["timeline"]]),dim=-1)
    score=items@user; ranked=[{**x,"score":float(score[i])} for i,x in enumerate(sample["timeline"])]
    ranked.sort(key=lambda x:(-x["score"],x["segment_id"]))
    out.append({"sample_id":sample["sample_id"],"split":sample["split"],"user_id":sample["user_id"],
        "relevance":sample["relevance"],"counterfactual_of":sample.get("counterfactual_of"),
        "ranked_candidates":ranked,"sample_score":float(torch.logsumexp(score,0)),"representation":a.name})
a.output.parent.mkdir(parents=True,exist_ok=True); write_jsonl(a.output,out); print({"records":len(out),"output":str(a.output)})
