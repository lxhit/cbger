#!/usr/bin/env bash
set -euo pipefail
echo "The released checkpoints preserve the exact original state dictionaries."
echo "Use scripts/run_cbger_3seeds.sh to regenerate predictions and metrics."
echo "Use scripts/eval/evaluate_cbger.py to evaluate generated prediction files."
