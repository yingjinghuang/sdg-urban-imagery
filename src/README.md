# `src/` — pipeline stages

The pipeline runs end-to-end in this order:

| # | Stage | What it does | Inputs | Outputs |
|---|---|---|---|---|
| 1 | `preprocess/` | Build per-country label tables; join street-view image paths to neighborhood IDs; compute mean/std for normalization. | Raw country-specific indicator tables; neighborhood shapefiles. | `processed/{Country}/{City}/labels.pkl`, `paths.pkl`, `labels_norm.pkl`. |
| 2 | `datasets/` | Construct spatial-contrastive training pairs (same-neighborhood pairs for the spatial-contrastive objective). | `processed/.../paths.pkl`. | `datasets/train/{spatial,self}_{Country}_{City}.pkl`. |
| 3 | `extract/` | Per-image feature extraction with a pretrained Mocov3 backbone (street-view ViT-B and satellite ViT-B), then mean-aggregation to the neighborhood level. | Raw imagery, pretrained checkpoints. | `features/Raw/{Country}/{City}/{SV,RS}/*.h5`, `features/Unit/{Country}/{City}/{SV,RS}/*.pkl`. |
| 4 | `fuse/` | Concatenate features across the two modalities (SV ⊕ RS) and across two learning objectives (self ⊕ spatial). Optional PCA-99% reduction. | Unit-level features. | `features/Unit/{Country}/{City}/Fuse/Concat_*.pkl`. |
| 5 | `regression/` | Token regression mapping fused features to SDG indicators. Three experimental settings: K-fold cross-validation, train-ratio sweep, and feature-guided sampling. | Fused features, labels. | `regression_outputs/{Fold,Ratio,Sampling}/...` — CSV (R²/MAE/MSE), HDF5 (per-neighborhood predictions), and trained regression-head checkpoints. |
| - | `analysis/` | Off-pipeline post-hoc analyses: CLIP-concept interpretability scoring, residual decomposition, t-SNE export. Used to produce panels b–e of Figure 2. | Raw and Unit features. | `data/processed/sv_clip_scores.pkl`, `clip_concept_*.csv`. |

Bash launchers in `../scripts/` wrap each stage to iterate over the 20 regions listed in `../configs/cities.yaml`.
