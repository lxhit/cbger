#!/usr/bin/env python3
"""Evaluate CBGER-10K Where, Whether and Intervention axes."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from cbger.io import read_jsonl


def scores(row):
    return {candidate["segment_id"]: float(candidate["score"])
            for candidate in row["ranked_candidates"]}


def ndcg(relevance, cutoff):
    values = relevance[:cutoff]
    dcg = sum(value / math.log2(rank + 2) for rank, value in enumerate(values))
    ideal = sorted(relevance, reverse=True)[:cutoff]
    idcg = sum(value / math.log2(rank + 2) for rank, value in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def mean(values):
    return sum(values) / max(1, len(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--aggregation", choices=("stored", "mean", "max"), default="mean",
        help="Video-level Whether score. The paper uses arithmetic mean.",
    )
    a = ap.parse_args()
    samples = {r["sample_id"]:r for r in read_jsonl(a.dataset) if r["split"] == a.split}
    pred = {r["sample_id"]:r for r in read_jsonl(a.predictions) if r["split"] == a.split}
    positives = {sid:r for sid,r in samples.items() if r["relevance"] == 1}
    counterfactuals = {r["counterfactual_of"]:r for r in samples.values()
                       if r.get("counterfactual_of")}
    if set(samples) - set(pred):
        raise RuntimeError(f"missing {len(set(samples)-set(pred))} predictions")
    rr=[]; n1=[]; n3=[]; n5=[]; na=[]; pair=[]; ties=[]; intervention=[]; margins=[]
    for sid, positive in positives.items():
        p=pred[sid]; evidence={r["segment_id"] for r in positive["timeline"] if r["role"]=="evidence"}
        rel=[int(r["segment_id"] in evidence) for r in p["ranked_candidates"]]
        ranks=[i+1 for i,v in enumerate(rel) if v]
        rr.append(1/min(ranks) if ranks else 0.0)
        n1.append(ndcg(rel,1)); n3.append(ndcg(rel,3)); n5.append(ndcg(rel,5)); na.append(ndcg(rel,len(rel)))
        cf=counterfactuals[sid]; cp=pred[cf["sample_id"]]
        if a.aggregation == "mean":
            p_sample = mean(scores(p).values())
            cp_sample = mean(scores(cp).values())
        elif a.aggregation == "max":
            p_sample = max(scores(p).values())
            cp_sample = max(scores(cp).values())
        else:
            p_sample = float(p["sample_score"])
            cp_sample = float(cp["sample_score"])
        margin=p_sample-cp_sample
        pair.append(margin>0); ties.append(margin==0)
        ps,cs=scores(p),scores(cp)
        local=ps[cf["provenance"]["removed_segment_id"]]-cs[cf["provenance"]["replacement_segment_id"]]
        intervention.append(local>0); margins.append(local)
    report={"split":a.split,"positive_records":len(positives),"paired_records":len(pair),
        "mrr":mean(rr),"ndcg_at_1":mean(n1),"ndcg_at_3":mean(n3),"ndcg_at_5":mean(n5),"ndcg_all":mean(na),
        "pair_accuracy":mean(pair),"pair_tie_rate":mean(ties),
        "whether_aggregation": a.aggregation,
        "intervention_consistency":mean(intervention),"mean_intervention_margin":mean(margins),
        "task_axes":{"where":["mrr","ndcg"],"intervention":["intervention_consistency"],
                     "whether":["pair_accuracy"]},"profile_swap_in_scope":False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
