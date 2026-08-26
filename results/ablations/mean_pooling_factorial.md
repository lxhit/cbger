# PBGER-v0.8 parameter-free Mean-pooling final

All results use three seeds. Mean pooling changes only the Whether score; MRR and Intervention retain the frozen segment-ranking results.

## CLIP 2x2 ablation

| Structured CF | Mean pooling | Model | MRR | PairAcc | Intervention |
|---:|---:|---|---:|---:|---:|
| 0 | 0 | PBGER-Lite | 0.4274 ± 0.0100 | 0.6230 ± 0.0046 | 0.6830 ± 0.0079 |
| 0 | 1 | PBGER-Lite + Mean | 0.4274 ± 0.0100 | 0.6810 ± 0.0085 | 0.6830 ± 0.0079 |
| 1 | 0 | PBGER-Lite + Structured CF | 0.4432 ± 0.0122 | 0.6297 ± 0.0085 | 0.6987 ± 0.0035 |
| 1 | 1 | PBGER-final | 0.4432 ± 0.0122 | 0.6977 ± 0.0070 | 0.6987 ± 0.0035 |

## Cross-backbone Mean pooling

| Backbone | Lite raw PairAcc | Lite + Mean | Structured CF raw | PBGER-final (CF + Mean) |
|---|---:|---:|---:|---:|
| CLIP | 0.6230 ± 0.0046 | 0.6810 ± 0.0085 | 0.6297 ± 0.0085 | 0.6977 ± 0.0070 |
| SmolVLM2 | 0.6543 ± 0.0115 | 0.6240 ± 0.0053 | 0.6600 ± 0.0056 | 0.6240 ± 0.0010 |
| DINOv2+BGE | 0.6447 ± 0.0075 | 0.5863 ± 0.0071 | 0.6323 ± 0.0142 | 0.5907 ± 0.0100 |

## Paired bootstrap significance

Positive delta favors the second condition named by each contrast.

| Backbone | Contrast | Delta | 95% CI | p |
|---|---|---:|---:|---:|
| CLIP | Mean vs raw, without CF | +0.0580 | [+0.0387, +0.0773] | 0.0000 |
| CLIP | Mean vs raw, with CF | +0.0680 | [+0.0483, +0.0873] | 0.0000 |
| CLIP | CF vs Lite, raw Whether | +0.0067 | [-0.0110, +0.0243] | 0.4636 |
| CLIP | CF vs Lite, Mean Whether | +0.0167 | [+0.0057, +0.0277] | 0.0024 |
| CLIP | CF x Mean interaction | +0.0100 | [-0.0100, +0.0303] | 0.3382 |
| SmolVLM2 | Mean vs raw, without CF | -0.0303 | [-0.0487, -0.0120] | 0.0010 |
| SmolVLM2 | Mean vs raw, with CF | -0.0360 | [-0.0543, -0.0180] | 0.0002 |
| SmolVLM2 | CF vs Lite, raw Whether | +0.0057 | [-0.0070, +0.0180] | 0.3766 |
| SmolVLM2 | CF vs Lite, Mean Whether | +0.0000 | [-0.0090, +0.0087] | 1.0000 |
| SmolVLM2 | CF x Mean interaction | -0.0057 | [-0.0207, +0.0093] | 0.4688 |
| DINOv2+BGE | Mean vs raw, without CF | -0.0583 | [-0.0793, -0.0377] | 0.0000 |
| DINOv2+BGE | Mean vs raw, with CF | -0.0417 | [-0.0627, -0.0207] | 0.0004 |
| DINOv2+BGE | CF vs Lite, raw Whether | -0.0123 | [-0.0293, +0.0043] | 0.1508 |
| DINOv2+BGE | CF vs Lite, Mean Whether | +0.0043 | [-0.0107, +0.0193] | 0.5730 |
| DINOv2+BGE | CF x Mean interaction | +0.0167 | [-0.0050, +0.0393] | 0.1444 |
