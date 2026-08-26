#!/usr/bin/env python3
"""Aggregate three-seed CBGER-10K reports."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path


METRICS=("mrr","ndcg_at_1","ndcg_at_3","ndcg_at_5","ndcg_all",
         "pair_accuracy","intervention_consistency","mean_intervention_margin")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--reports",type=Path,nargs="+",required=True)
    ap.add_argument("--method",required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    rows=[json.loads(p.read_text(encoding="utf-8")) for p in a.reports]
    summary={"method":a.method,"seeds":len(rows),"profile_swap_in_scope":False,"metrics":{}}
    for key in METRICS:
        values=[float(r[key]) for r in rows]
        summary["metrics"][key]={"mean":statistics.fmean(values),
            "std":statistics.stdev(values) if len(values)>1 else 0.0,"values":values}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
