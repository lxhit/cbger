# Script map

Supported release entry points:

- `run_cbger_3seeds.sh`: reproduce the CBGER three-seed experiment.
- `data/annotate_qwen3vl.py`: structured segment annotation.
- `data/build_cbger10k.py`: BGE-M3 matching and pair construction.
- `data/validate_cbger10k.py`: frozen benchmark validation.
- `train/train_cbger.py`: exact result-generating CBGER trainer.
- `eval/evaluate_cbger.py`: Where/Whether/Intervention metrics; mean pooling is the paper default.
- `baselines/run_all_baselines.sh`: consolidated baseline launcher after third-party setup.

`legacy/` preserves paper-stage transformation/report scripts with historical PBGER paths and names. They are included for provenance, not as public entry points.

