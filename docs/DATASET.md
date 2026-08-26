# CBGER-10K data card

## Scope

CBGER-10K is a controlled diagnostic benchmark for personalized video evidence retrieval. It evaluates localization (**Where**), factual/counterfactual evidence ordering (**Whether**) and focal-slot response under replacement (**Intervention**). It does not claim that a constructed segment causally changes real user engagement.

## Sources

- **MicroLens:** timestamped user-item histories and video titles used to derive behavior-supported profiles.
- **FineVideo:** source videos, temporal intervals and provenance used to form the segment pool.
- **Qwen3-VL-8B-Instruct:** independent structured visual-semantic annotation at 4 FPS, BF16 and deterministic decoding.
- **BGE-M3:** constructor-side profile/segment matching. The downstream model uses frozen CLIP ViT-B/32 features, reducing direct constructor-solver coupling.

## Size and splits

- 10,000 records = 5,000 factual/counterfactual pairs
- 3,026 users
- 7,000/1,000/2,000 train/validation/test records
- 9 segments per timeline
- stable 70/10/20 user split
- FineVideo sources partitioned before matching; user and source identities are disjoint across splits

## Main record schema

`data/cbger10k/cbger10k.jsonl` contains one JSON object per line.

| Field | Type | Meaning |
|---|---|---|
| `sample_id` | string | Stable record ID. The frozen release retains legacy `pbger_v0_8_*` IDs so original checkpoints and predictions remain compatible. |
| `split` | string | `train`, `validation`, or `test`. |
| `user_id` | string | Anonymized MicroLens user identifier. |
| `history_ids` | list[string] | Chronological behavior-history item IDs. |
| `relevance` | integer | `1` factual evidence-present; `0` matched counterfactual. |
| `counterfactual_of` | string/null | Factual sample ID for a counterfactual record. |
| `timeline` | list[object] | Nine ordered segment records. |
| `evidence_segments` | list[[start,end]] | Target-time evidence interval for factual records; empty for counterfactuals. |
| `difficulty` | string | Construction difficulty descriptor. |
| `provenance` | object | Models, thresholds, scores, seed, labels and edited segment IDs. |

Each timeline entry contains `segment_id`, `source_id`, `source_interval`, `target_interval`, `role`, and `attribution`.

## Profiles and features

- `global_profiles_clip_vit_b32.jsonl`: one 512-D history-supported global behavior-profile feature per user.
- `clip_vit_b32_segment_features.jsonl`: frozen 512-D CLIP ViT-B/32 feature for every referenced segment.

The profile file is a solver-side representation. Constructor-side matching used Qwen3-VL structured text and BGE-M3, not these CLIP features.

## Automated construction

### 1. Behavior profile

Normalize MicroLens titles, remove low-information tokens, form unigrams/bigrams and record exact supporting history IDs. For phrase `a` and user `u`:

```text
idf(a) = log((|U| + 1) / (df_U(a) + 1))
c_u(a) = |H_u(a)| * idf(a) * rho(a)
rho(a) = 1.25 for bigrams, otherwise 1
```

The result is an operational summary of observed history, not psychological ground truth.

```bash
PYTHONPATH=src python scripts/data/build_behavior_profiles.py \
  --pairs data/raw/microlens/interactions.csv \
  --titles data/raw/microlens/titles.csv \
  --output data/interim/user_profiles_v2.jsonl
```

### 2. Segment annotation

Eligible FineVideo intervals are 1-12 seconds. Run:

```bash
PYTHONPATH=src:scripts/data python scripts/data/annotate_qwen3vl.py \
  --root "$PWD" --fps 4 --precision bf16
```

The generator returns deterministic JSON with objects, actions, ordered events, outcome, state change, topics and uncertainty. Invalid JSON, decode failures and uncertainty above 0.5 are rejected.

### 3. Matching and mining

```bash
PYTHONPATH=src python scripts/data/build_cbger10k.py \
  --root "$PWD" --pairs 3500,500,1000 \
  --min-positive 0.30 --min-gap 0.06
```

For each user, BGE-M3 matches the serialized profile to structured segment text. A factual evidence candidate must have score at least 0.30. The replacement is a semantic neighbor with content similarity at least 0.45 and behavior score at least 0.06 lower. Eight nearby behavior-weaker segments are retained as hard distractors.

### 4. Pair construction

The factual and counterfactual share:

- user and complete history;
- temporal slot of the focal segment;
- eight distractor segment IDs and order;
- all non-focal target intervals.

Only the factual focal evidence is replaced. This supports paired evaluation without claiming causal engagement effects.

## Evaluation labels

- **Where:** rank the designated factual evidence segment using MRR and NDCG.
- **Whether:** require `q(u,V+) > q(u,V-)`; report Pair Accuracy.
- **Intervention:** require the focal factual segment score to exceed the replacement score at the same slot; report Intervention Consistency.

## Integrity

The original frozen files have these paper-stage SHA-256 values:

```text
f0a292c83a2e49fc1c120a1d432608ce4ed70a7e7a93ad72fc999b97479322cf  cbger10k.jsonl
7446fb145fd6766c9dbd48a114f3d72ee55017255364bdeb8fb1816068c17ff6  global_profiles_clip_vit_b32.jsonl
```

Run `sha256sum -c checksums.sha256` from the repository root to validate all release artifacts.

## Limitations

Labels are automatically constructed and may inherit Qwen3-VL/BGE-M3 biases. A behavior-weaker replacement is not guaranteed to be universally irrelevant. Histories and candidate videos come from different public resources, so the benchmark measures history-supported evidence rather than observed exposure or causal preference.
