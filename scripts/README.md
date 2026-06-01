# `scripts/` — bash launchers

Pipeline entry points. Each script sources `_lib.sh`, which reads `configs/paths.yaml` and `configs/cities.yaml` and iterates over the 20 regions.

## Setup

1. Copy `configs/paths.example.yaml` → `configs/paths.yaml` and edit for your machine.
2. Optional: override `CUDA_DEVICES`, `BATCH_SIZE_PER_GPU`, or `PYTHON` in your shell.

## End-to-end pipeline

```bash
# 1. Pretrain backbones (skip if downloading public checkpoints)
bash scripts/pretrain_sv.sh self    cities100-1m
bash scripts/pretrain_sv.sh spatial cities100-1m  pretrain_ckpts/self_cities100-1m.pth.tar
bash scripts/pretrain_rs.sh self    cities100-1m
bash scripts/pretrain_rs.sh spatial cities100-1m  pretrain_ckpts/self_rs_cities100-1m.pth.tar

# 2. Extract features (per-image -> per-neighborhood)
bash scripts/extract_features.sh SV self
bash scripts/extract_features.sh SV spatial
bash scripts/extract_features.sh RS self
bash scripts/extract_features.sh RS spatial
bash scripts/extract_features_imagenet.sh        # Fig 1a baseline
bash scripts/extract_features_segmentation.sh    # Fig 1a baseline

# 3. Fuse features
bash scripts/fuse_features.sh

# 4. Pre-compute sample-selection orderings
bash scripts/select_kcenter.sh

# 5. Regression — main results
bash scripts/run_fold.sh                         # Fig 1a "Ours" + Fig 2a fusion
bash scripts/run_fold_single_modal.sh            # Fig 2a single-modal bars
bash scripts/run_fold_imagenet.sh                # Fig 1a ImageNet
bash scripts/run_fold_segmentation.sh            # Fig 1a Segmentation
bash scripts/run_ratio.sh                        # Fig 1b/c/d random-sampling curves
bash scripts/run_ratio_no_geo.sh                 # Fig 2f baseline
bash scripts/run_sampling.sh                     # Fig 1b/c/d feature-guided sampling
bash scripts/run_sampling_no_geo.sh              # Fig 2f baseline

# 6. Side analysis (Fig 2 b-e inputs)
bash scripts/compute_clip_scores.sh
```

## Path conventions

All output paths are organized as:

```
${REGRESSION_OUT_DIR}/{fold|ratio|sampling}/{run_tag}/{Country}/{City}/{TYPE}/{Token|Multi}_<feature>/
```

`run_tag` is one of `main`, `main_no_geo`, `imagenet`, `segmentation`. This replaces the legacy `regmodels_*` directory naming where the directory name didn't match its contents (e.g. `regmodels_spatial/.../Token_Concat_spatial_self/` was actually the main framework output).

## Environment variables (overridable)

| Var | Default | Meaning |
|---|---|---|
| `SDG_CONFIG` | `configs/paths.yaml` | Path to the paths file. |
| `PYTHON` | `python` | Python interpreter to use. |
| `CUDA_DEVICES` | (from paths.yaml) | GPU IDs. |
| `BATCH_SIZE_PER_GPU` | 2048 | Per-GPU batch for feature extraction. |
| `TYPE` | `Fuse` (regression) / per-script (extraction) | Modality target. |
| `FEATURE_LEN` | 1536 | Feature dimensionality after fusion. |
