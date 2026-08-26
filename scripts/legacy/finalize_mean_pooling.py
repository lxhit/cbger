#!/usr/bin/env python3
"""Finalize PBGER-v0.8 with parameter-free mean evidence pooling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cbger.io import read_jsonl


R = Path(__file__).resolve().parents[2]
SEEDS = (20260831, 20260832, 20260833)
RUNS = {
    "CLIP": {
        "lite": "outputs/v08/pbger_lite/lite/raw/seed_{seed}.jsonl",
        "cf": "outputs/v08/pbger_lite/lite_cf/raw/seed_{seed}.jsonl",
        "lite_summary": "outputs/v08/pbger_lite/lite/summary.json",
        "cf_summary": "outputs/v08/pbger_lite/lite_cf/summary.json",
    },
    "SmolVLM2": {
        "lite": "outputs/v08/cross_ladder/smol_lite/raw/seed_{seed}.jsonl",
        "cf": "outputs/v08/cross_backbone/smolvlm2/raw/seed_{seed}.jsonl",
        "lite_summary": "outputs/v08/cross_ladder/smol_lite/summary.json",
        "cf_summary": "outputs/v08/cross_ladder/smol_cf/summary.json",
    },
    "DINOv2+BGE": {
        "lite": "outputs/v08/cross_ladder/dino_lite/raw/seed_{seed}.jsonl",
        "cf": "outputs/v08/cross_backbone/dinov2_bge/raw/seed_{seed}.jsonl",
        "lite_summary": "outputs/v08/cross_ladder/dino_lite/summary.json",
        "cf_summary": "outputs/v08/cross_ladder/dino_cf/summary.json",
    },
}


def load(path):
    return list(read_jsonl(R / path))


def pair_correctness(rows, scoring):
    by = {x["sample_id"]: x for x in rows if x["split"] == "test"}
    neg = {x["counterfactual_of"]: x for x in by.values() if x.get("counterfactual_of")}
    pos = sorted((x for x in by.values() if int(x["relevance"]) == 1), key=lambda x: x["sample_id"])
    return np.asarray([float(scoring(x) > scoring(neg[x["sample_id"]])) for x in pos])


def raw_score(x):
    return float(x["sample_score"])


def mean_score(x):
    return float(np.mean([float(y["score"]) for y in x["ranked_candidates"]]))


def summarize(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "values": [float(x) for x in values]}


def metric(summary, name):
    return summary["metrics"][name]


def paired_bootstrap(arrays_a, arrays_b, rng, draws=10000):
    observed = float(np.mean([b.mean() - a.mean() for a, b in zip(arrays_a, arrays_b)]))
    distribution = []
    for _ in range(draws):
        per_seed = []
        for a, b in zip(arrays_a, arrays_b):
            idx = rng.integers(0, len(a), len(a))
            per_seed.append(float((b[idx] - a[idx]).mean()))
        distribution.append(float(np.mean(per_seed)))
    distribution = np.asarray(distribution)
    p = min(1.0, 2 * min(float((distribution <= 0).mean()), float((distribution >= 0).mean())))
    return {"delta": observed, "ci95": [float(x) for x in np.quantile(distribution, [.025, .975])], "p": p}


def main():
    report = {"dataset": "PBGER-v0.8", "seeds": list(SEEDS), "pooling": "arithmetic mean over all segment scores", "backbones": {}}
    rng = np.random.default_rng(20260831)
    for backbone, cfg in RUNS.items():
        lite_rows = [load(cfg["lite"].format(seed=s)) for s in SEEDS]
        cf_rows = [load(cfg["cf"].format(seed=s)) for s in SEEDS]
        lite_summary = json.loads((R / cfg["lite_summary"]).read_text())
        cf_summary = json.loads((R / cfg["cf_summary"]).read_text())
        correct = {
            "lite_raw": [pair_correctness(x, raw_score) for x in lite_rows],
            "lite_mean": [pair_correctness(x, mean_score) for x in lite_rows],
            "cf_raw": [pair_correctness(x, raw_score) for x in cf_rows],
            "cf_mean": [pair_correctness(x, mean_score) for x in cf_rows],
        }
        pair_metrics = {k: summarize([a.mean() for a in v]) for k, v in correct.items()}
        tests = {
            "mean_effect_without_cf": paired_bootstrap(correct["lite_raw"], correct["lite_mean"], rng),
            "mean_effect_with_cf": paired_bootstrap(correct["cf_raw"], correct["cf_mean"], rng),
            "cf_effect_without_mean": paired_bootstrap(correct["lite_raw"], correct["cf_raw"], rng),
            "cf_effect_with_mean": paired_bootstrap(correct["lite_mean"], correct["cf_mean"], rng),
        }
        # Difference-in-differences for the complete 2x2 design.
        observed = tests["mean_effect_with_cf"]["delta"] - tests["mean_effect_without_cf"]["delta"]
        dist = []
        for _ in range(10000):
            per_seed = []
            for lr, lm, cr, cm in zip(correct["lite_raw"], correct["lite_mean"], correct["cf_raw"], correct["cf_mean"]):
                idx = rng.integers(0, len(lr), len(lr))
                per_seed.append(float(((cm[idx] - cr[idx]) - (lm[idx] - lr[idx])).mean()))
            dist.append(float(np.mean(per_seed)))
        dist = np.asarray(dist)
        tests["cf_x_mean_interaction"] = {
            "delta": float(observed),
            "ci95": [float(x) for x in np.quantile(dist, [.025, .975])],
            "p": min(1.0, 2 * min(float((dist <= 0).mean()), float((dist >= 0).mean()))),
        }
        report["backbones"][backbone] = {
            "pair_accuracy": pair_metrics,
            "mrr_lite": metric(lite_summary, "mrr"),
            "intervention_lite": metric(lite_summary, "intervention_consistency"),
            "mrr_cf": metric(cf_summary, "mrr"),
            "intervention_cf": metric(cf_summary, "intervention_consistency"),
            "paired_bootstrap": tests,
        }

    out = R / "outputs/v08/mean_pooling_final"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# PBGER-v0.8 parameter-free Mean-pooling final", "",
             "All results use three seeds. Mean pooling changes only the Whether score; MRR and Intervention retain the frozen segment-ranking results.", ""]
    clip = report["backbones"]["CLIP"]
    lines += ["## CLIP 2x2 ablation", "",
              "| Structured CF | Mean pooling | Model | MRR | PairAcc | Intervention |",
              "|---:|---:|---|---:|---:|---:|"]
    cells = [(0, 0, "PBGER-Lite", "lite_raw", "lite"),
             (0, 1, "PBGER-Lite + Mean", "lite_mean", "lite"),
             (1, 0, "PBGER-Lite + Structured CF", "cf_raw", "cf"),
             (1, 1, "PBGER-final", "cf_mean", "cf")]
    for cf, mean, name, pk, branch in cells:
        p = clip["pair_accuracy"][pk]; m = clip[f"mrr_{branch}"]; i = clip[f"intervention_{branch}"]
        lines.append(f"| {cf} | {mean} | {name} | {m['mean']:.4f} ± {m['std']:.4f} | {p['mean']:.4f} ± {p['std']:.4f} | {i['mean']:.4f} ± {i['std']:.4f} |")
    lines += ["", "## Cross-backbone Mean pooling", "",
              "| Backbone | Lite raw PairAcc | Lite + Mean | Structured CF raw | PBGER-final (CF + Mean) |",
              "|---|---:|---:|---:|---:|"]
    for b, x in report["backbones"].items():
        p=x["pair_accuracy"]
        lines.append(f"| {b} | {p['lite_raw']['mean']:.4f} ± {p['lite_raw']['std']:.4f} | {p['lite_mean']['mean']:.4f} ± {p['lite_mean']['std']:.4f} | {p['cf_raw']['mean']:.4f} ± {p['cf_raw']['std']:.4f} | {p['cf_mean']['mean']:.4f} ± {p['cf_mean']['std']:.4f} |")
    lines += ["", "## Paired bootstrap significance", "",
              "Positive delta favors the second condition named by each contrast.", "",
              "| Backbone | Contrast | Delta | 95% CI | p |",
              "|---|---|---:|---:|---:|"]
    labels = {
        "mean_effect_without_cf": "Mean vs raw, without CF",
        "mean_effect_with_cf": "Mean vs raw, with CF",
        "cf_effect_without_mean": "CF vs Lite, raw Whether",
        "cf_effect_with_mean": "CF vs Lite, Mean Whether",
        "cf_x_mean_interaction": "CF x Mean interaction",
    }
    for b, x in report["backbones"].items():
        for key, label in labels.items():
            t=x["paired_bootstrap"][key]
            lines.append(f"| {b} | {label} | {t['delta']:+.4f} | [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] | {t['p']:.4f} |")
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
