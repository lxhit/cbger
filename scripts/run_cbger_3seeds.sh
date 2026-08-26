#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA="$ROOT/data/cbger10k/cbger10k.jsonl"
PROFILES="$ROOT/data/cbger10k/global_profiles_clip_vit_b32.jsonl"
FEATURES="$ROOT/data/features/clip_vit_b32_segment_features.jsonl"
OUT="$ROOT/outputs/cbger"
mkdir -p "$OUT"/{predictions,checkpoints,metrics}

for SEED in 20260831 20260832 20260833; do
  PYTHONPATH="$ROOT/src:$ROOT/scripts/train" "$PYTHON" "$ROOT/scripts/train/train_cbger.py" \
    --dataset "$DATA" --profiles "$PROFILES" --features "$FEATURES" \
    --output "$OUT/predictions/seed_${SEED}.jsonl" \
    --checkpoint "$OUT/checkpoints/seed_${SEED}.pt" \
    --seed "$SEED" --warmup-epochs 2 --counterfactual-epochs 2 --robust-epochs 2 \
    --hidden-dim 256 --learning-rate 2e-4 --distill-weight 2 \
    --counterfactual-distill-scale .25 --robust-distill-scale 0 \
    --pair-weight 2 --necessity-weight .5 --sufficiency-weight .5 \
    --relocation-weight 1.5 --personalization-weight 0 --sparse-weight 0 \
    --no-routing --preload-features

  PYTHONPATH="$ROOT/src" "$PYTHON" "$ROOT/scripts/eval/evaluate_cbger.py" \
    --dataset "$DATA" --predictions "$OUT/predictions/seed_${SEED}.jsonl" \
    --aggregation mean --output "$OUT/metrics/seed_${SEED}.json"
done

PYTHONPATH="$ROOT/src" "$PYTHON" "$ROOT/scripts/eval/summarize_seeds.py" \
  --reports "$OUT/metrics/seed_20260831.json" "$OUT/metrics/seed_20260832.json" \
  "$OUT/metrics/seed_20260833.json" --method CBGER --output "$OUT/summary.json"

cat "$OUT/summary.json"

