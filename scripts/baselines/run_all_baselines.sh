#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python}"
DATA=$ROOT/data/cbger10k/cbger10k.jsonl
PROFILES=$ROOT/data/cbger10k/global_profiles_clip_vit_b32.jsonl
FEATURES=$ROOT/data/features/clip_vit_b32_segment_features.jsonl
QD_DATA=$ROOT/data/baselines/qd_detr_official/cbger10k
OUT=$ROOT/outputs/baselines
CK=$ROOT/outputs/checkpoints/baselines
SEEDS=(20260831 20260832 20260833)
mkdir -p "$OUT" "$CK"
cd "$ROOT"

evaluate_three() {
  local method=$1 directory=$2
  PYTHONPATH=src "$PY" scripts/eval/summarize_seeds.py --reports \
    "$directory/eval/seed_20260831.json" "$directory/eval/seed_20260832.json" \
    "$directory/eval/seed_20260833.json" --method "$method" --output "$directory/summary.json"
}

run_prnet() {
  local dir=$OUT/prnet; mkdir -p "$dir"/{raw,eval,logs} "$CK/prnet"
  for seed in "${SEEDS[@]}"; do
    if [[ ! -s "$dir/raw/seed_${seed}.jsonl" ]]; then
      PYTHONPATH=src:scripts/train CUDA_VISIBLE_DEVICES=0 "$PY" scripts/baselines/train_prnet.py \
        --model prnet_paper_faithful --dataset "$DATA" --profiles "$PROFILES" --features "$FEATURES" \
        --output "$dir/raw/seed_${seed}.jsonl" --checkpoint "$CK/prnet/seed_${seed}.pt" \
        --hidden-dim 256 --epochs 20 --early-stop-patience 5 --learning-rate 3e-4 \
        --pair-weight 2 --paper-loss-weight 1 --seed "$seed" --preload-features \
        > "$dir/logs/seed_${seed}.log" 2>&1
    fi
    PYTHONPATH=src "$PY" scripts/eval/evaluate_cbger.py --aggregation mean --dataset "$DATA" \
      --predictions "$dir/raw/seed_${seed}.jsonl" --output "$dir/eval/seed_${seed}.json"
  done
  evaluate_three "PR-Net paper-faithful" "$dir"
}

run_qddetr() {
  local repo=$ROOT/third_party/qd_detr dir=$OUT/qddetr
  mkdir -p "$dir"/{runs,markers,logs,raw,eval}
  for seed in "${SEEDS[@]}"; do
    local marker=$dir/markers/seed_${seed}.complete
    if [[ ! -s "$marker" ]]; then
      cd "$repo"
      PYTHONPATH="$repo" CUDA_VISIBLE_DEVICES=0 "$PY" qd_detr/train.py \
        --seed "$seed" --dset_name hl --ctx_mode video_tef \
        --train_path "$QD_DATA/pbger_train.jsonl" --eval_path "$QD_DATA/pbger_validation.jsonl" \
        --eval_split_name val --v_feat_dirs "$QD_DATA/video_features" --v_feat_dim 512 \
        --t_feat_dir "$QD_DATA/query_features" --t_feat_dim 512 --clip_length 1 --max_v_l 16 \
        --max_q_l 32 --bsz 32 --eval_bsz 100 --num_workers 4 --results_root "$dir/runs" \
        --exp_id "pbger_v08_seed_${seed}" --n_epoch 200 --max_es_cnt 30 \
        > "$dir/logs/seed_${seed}.log" 2>&1
      local run
      run=$(find "$dir/runs" -maxdepth 1 -type d -name "hl-video_tef-pbger_v08_seed_${seed}-*" \
        -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
      [[ -n "$run" && -s "$run/model_best.ckpt" ]]; printf '%s\n' "$run" > "$marker"
      cd "$ROOT"
    fi
    local run; run=$(cat "$marker")
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 "$PY" scripts/baselines/export_qddetr.py \
      --checkpoint "$run/model_best.ckpt" --dataset "$DATA" --profiles "$PROFILES" \
      --features "$FEATURES" --output "$dir/raw/seed_${seed}.jsonl" --batch-size 128 --seed "$seed"
    PYTHONPATH=src "$PY" scripts/eval/evaluate_cbger.py --aggregation mean --dataset "$DATA" \
      --predictions "$dir/raw/seed_${seed}.jsonl" --output "$dir/eval/seed_${seed}.json"
  done
  evaluate_three "QD-DETR official" "$dir"
}

run_modern() {
  local kind=$1 method=$2 repo train_cmd
  local dir=$OUT/$kind
  mkdir -p "$dir"/{runs,markers,logs,raw,eval}
  case "$kind" in
    tr_detr) repo=$ROOT/third_party/tr_detr ;;
    flashvtg) repo=$ROOT/third_party/flashvtg ;;
    mqvtg) repo=$ROOT/third_party/mqvtg ;;
  esac
  for seed in "${SEEDS[@]}"; do
    local marker=$dir/markers/seed_${seed}.complete
    if [[ ! -s "$marker" ]]; then
      cd "$repo"
      if [[ "$kind" == flashvtg ]]; then
        PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 "$PY" FlashVTG/train.py data/MR.py \
          --seed "$seed" --dset_name hl --ctx_mode video_tef --train_path "$QD_DATA/pbger_train.jsonl" \
          --eval_path "$QD_DATA/pbger_validation.jsonl" --eval_split_name val --v_feat_dirs "$QD_DATA/video_features" \
          --v_feat_dim 512 --t_feat_dir "$QD_DATA/query_features" --t_feat_dim 512 --clip_length 1 \
          --max_v_l 16 --max_q_l 32 --bsz 64 --eval_bsz 1 --num_workers 4 --results_root "$dir/runs" \
          --exp_id "pbger_v08_seed_${seed}" --n_epoch 150 --max_es_cnt 30 --eval_epoch 1 --use_neg \
          --enc_layers 3 --t2v_layers 6 --dummy_layers 2 --num_dummies 40 --kernel_size 5 \
          --num_conv_layers 1 --num_mlp_layers 5 --lw_reg 1 --lw_cls 5 --lw_sal 0.1 \
          --lw_saliency 0.8 --label_loss_coef 4 --use_SRM > "$dir/logs/seed_${seed}.log" 2>&1
      else
        local extra=()
        if [[ "$kind" == mqvtg ]]; then
          extra=(--mq_loss_coef 1.0 --mq_commitment 0.25 --mq_codebook_size 1024 \
                 --mq_codebook_init "$ROOT/data/baselines/mqvtg_paper_repro/pbger_v0_7/codebook_1024x256.npy")
        fi
        PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 "$PY" tr_detr/train.py \
          --seed "$seed" --dset_name hl --ctx_mode video_tef --train_path "$QD_DATA/pbger_train.jsonl" \
          --eval_path "$QD_DATA/pbger_validation.jsonl" --eval_split_name val --v_feat_dirs "$QD_DATA/video_features" \
          --v_feat_dim 512 --t_feat_dir "$QD_DATA/query_features" --t_feat_dim 512 --clip_length 1 \
          --max_v_l 16 --max_q_l 32 --bsz 32 --eval_bsz 100 --num_workers 4 --results_root "$dir/runs" \
          --exp_id "pbger_v08_seed_${seed}" --n_epoch 200 --max_es_cnt 30 --VTC_loss_coef 0.3 \
          --CTC_loss_coef 0.5 "${extra[@]}" > "$dir/logs/seed_${seed}.log" 2>&1
      fi
      local run
      run=$(find "$dir/runs" -maxdepth 1 -type d -name "hl-video_tef-pbger_v08_seed_${seed}-*" \
        -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
      [[ -n "$run" && -s "$run/model_best.ckpt" ]]; printf '%s\n' "$run" > "$marker"
      cd "$ROOT"
    fi
    local run; run=$(cat "$marker")
    CUDA_VISIBLE_DEVICES=0 "$PY" scripts/baselines/export_modern_vtg.py --kind "$kind" \
      --checkpoint "$run/model_best.ckpt" --dataset "$DATA" --profiles "$PROFILES" --features "$FEATURES" \
      --output "$dir/raw/seed_${seed}.jsonl" --seed "$seed" --split test
    PYTHONPATH=src "$PY" scripts/eval/evaluate_cbger.py --aggregation mean --dataset "$DATA" \
      --predictions "$dir/raw/seed_${seed}.jsonl" --output "$dir/eval/seed_${seed}.json"
  done
  evaluate_three "$method" "$dir"
}

case "${1:-all}" in
  prnet) run_prnet ;;
  qddetr) run_qddetr ;;
  tr_detr) run_modern tr_detr "TR-DETR official" ;;
  flashvtg) run_modern flashvtg "FlashVTG official" ;;
  mqvtg) run_modern mqvtg "MQVTG paper-faithful" ;;
  all) run_prnet; run_qddetr; run_modern tr_detr "TR-DETR official"; \
       run_modern flashvtg "FlashVTG official"; run_modern mqvtg "MQVTG paper-faithful" ;;
  *) echo "usage: $0 [prnet|qddetr|tr_detr|flashvtg|mqvtg|all]" >&2; exit 2 ;;
esac
