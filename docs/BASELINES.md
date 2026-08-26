# Baseline reproduction

All methods use the same CBGER-10K splits and frozen CLIP ViT-B/32 visual/text representations. Train with seeds `20260831`, `20260832`, and `20260833`; select hyperparameters/checkpoints on validation only.

## Frozen CLIP similarity

```bash
PYTHONPATH=src python scripts/baselines/frozen_clip.py \
  --dataset data/cbger10k/cbger10k.jsonl \
  --profiles data/cbger10k/global_profiles_clip_vit_b32.jsonl \
  --features data/features/clip_vit_b32_segment_features.jsonl \
  --output outputs/baselines/clip.jsonl --name clip_similarity
```

## PR-Net adaptation

The paper-faithful personalized-highlight adapter is contained in `scripts/baselines/train_prnet.py` and its supporting classes in the original experiment snapshot. Run with hidden dimension 256, 20 epochs, patience 5, learning rate `3e-4`, pair weight 2 and paper loss weight 1.

## Official temporal-grounding repositories

Clone under `third_party/`:

```bash
git clone https://github.com/wjun0830/QD-DETR.git third_party/qd_detr
git clone https://github.com/mingyao1120/TR-DETR.git third_party/tr_detr
git -C third_party/tr_detr checkout fca43a9c64f10eb6a055365081e5ff2abdbcafda
git clone https://github.com/Zhuo-Cao/FlashVTG.git third_party/flashvtg
git -C third_party/flashvtg checkout 25b95916feed900038fa762985c196ee14f16c59
```

MQVTG was reproduced as the paper-specific TR-DETR extension used in the original project; its codebook initialization and patch must be documented or released separately before claiming one-command reproduction.

The consolidated launcher is `scripts/baselines/run_all_baselines.sh`. The official VTG repositories must be cloned and CBGER-10K must first be converted into their query/video feature layout under `data/baselines/qd_detr_official/cbger10k`.

Machine-readable three-seed metrics are under `results/baselines/`.
