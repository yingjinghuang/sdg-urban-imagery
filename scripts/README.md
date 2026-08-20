# `scripts/` — bash launchers

Pipeline entry points. Each script sources `_lib.sh`, which reads
`configs/paths.yaml` and `configs/cities.yaml` and iterates over the configured
regions.

## Setup

1. Copy `configs/paths.example.yaml` to `configs/paths.yaml` and edit it for your machine.
2. Optional: override `CUDA_DEVICES`, `BATCH_SIZE_PER_GPU`, or `PYTHON` in your shell.

## Paper pretraining settings

The maintained representation-pretraining launchers are:

```bash
bash scripts/pretrain_sv.sh <self|spatial> <city_tag> [init_ckpt] [resume_ckpt]
bash scripts/pretrain_rs.sh <self|spatial> <tag>      [init_ckpt] [resume_ckpt]
```

They use the manuscript configuration: ViT-B, AdamW, total batch size 1024,
initial optimizer learning rate `1e-5`, weight decay `1e-6`, 100 street-view
epochs and 50 satellite epochs. A 10-epoch warm-up is retained from the
implementation; it is not separately reported in the manuscript.

The cleaned `pretrain/moco_{sv,rs}.py` entry points use `--lr` directly as the
optimizer learning rate; they do not apply the stock MoCo-v3
`batch_size / 256` rescaling. `init_ckpt` initializes model weights with a fresh
optimizer/epoch counter (for example when initializing spatial-contrastive
training from a self-contrastive checkpoint). `resume_ckpt` is reserved for
resuming an interrupted same run with optimizer/scaler/epoch state restored.

The release pretraining code has no external experiment-tracking dependency;
TensorBoard logging is local.

## Paper regression settings

The regression launchers explicitly pin the settings reported in Supplementary
Table 9: hidden dimension 256, 8 attention heads, 1 Transformer layer, batch
size 128, maximum 100 epochs, 10-epoch warmup, early-stopping patience 5,
learning rate `1e-4`, and weight decay `1e-4`. The model uses dropout 0.5 before
the prediction head and an early-stopping delta of 0.005.

The main fold, random-sampling, and feature-guided-sampling regressions all use
the same four visual representation branches: street-view spatial, satellite
spatial, street-view self, and satellite self. They are stored in
`Concat_spatial_self.pkl` as four consecutive 768-dimensional tokens (3072
dimensions total). `FEATURE_LEN=768` denotes the width of each token.

## Feature-guided sampling configuration

`scripts/select_kcenter.sh` implements the sample-selection representation
described in the manuscript Methods. It reads the **unreduced**
`Concat_spatial_self.pkl` representation, passes the corresponding `geo.pkl`,
and then `src/regression/select_kcenter.py`:

1. splits the 3072-d visual representation into four 768-d branches;
2. standardizes each branch and applies PCA separately, retaining 99% variance within that branch;
3. standardizes neighborhood-centroid longitude/latitude;
4. weights the geographic coordinates by `0.5 * sqrt(d_v)`, where `d_v` is the concatenated PCA-reduced visual dimensionality;
5. runs seeded k-center greedy selection and writes nested 10%–90% survey definitions to `samples/pcahierachy.pkl`.

The optional globally reduced `Concat_spatial_self_pca99.pkl` file produced by
`fuse_features.sh` is retained only for auxiliary analysis. It is **not** used
as input to the paper's feature-guided sampling procedure.

## End-to-end pipeline

```bash
# 1. Pretrain backbones (skip if using the deposited checkpoints)
bash scripts/pretrain_sv.sh self    cities100-1m
bash scripts/pretrain_sv.sh spatial cities100-1m  pretrain_ckpts/self_cities100-1m.pth.tar
bash scripts/pretrain_rs.sh self    cities100-1m
bash scripts/pretrain_rs.sh spatial cities100-1m  pretrain_ckpts/self_rs_cities100-1m.pth.tar

# 2. Extract features (per-image -> per-neighborhood)
bash scripts/extract_features.sh SV self
bash scripts/extract_features.sh SV spatial
bash scripts/extract_features.sh RS self
bash scripts/extract_features.sh RS spatial
bash scripts/extract_features_imagenet.sh        # manuscript Fig. 2a baseline
bash scripts/extract_features_segmentation.sh    # manuscript Fig. 2a baseline

# 3. Fuse features
bash scripts/fuse_features.sh

# 4. Pre-compute feature-guided sample-selection orderings
bash scripts/select_kcenter.sh

# 5. Regression — main results
bash scripts/run_fold.sh                         # manuscript Fig. 2a + Fig. 3a fusion
bash scripts/run_fold_single_modal.sh            # manuscript Fig. 3a single-modal comparison
bash scripts/run_fold_imagenet.sh                # manuscript Fig. 2a ImageNet
bash scripts/run_fold_segmentation.sh            # manuscript Fig. 2a Segmentation
bash scripts/run_ratio.sh                        # manuscript Fig. 2b-d random-sampling curves
bash scripts/run_ratio_no_geo.sh                 # optional non-spatial ratio baseline
bash scripts/run_sampling.sh                     # manuscript Fig. 2b-d + Fig. 3f-g
bash scripts/run_sampling_no_geo.sh              # manuscript Fig. 3f baseline

# 6. Representation analysis (manuscript Fig. 3b-c inputs)
bash scripts/compute_clip_scores.sh
```

> The repository retains the historical `figures/fig1` and `figures/fig2`
> folder names for notebook compatibility. They correspond to current
> manuscript Fig. 2 and Fig. 3, respectively.

## Path conventions

All current regression output paths are organized as:

```text
${REGRESSION_OUT_DIR}/{fold|ratio|sampling}/{run_tag}/{Country}/{City}/{TYPE}/{Token|Multi}_<feature>/
```

`run_tag` is one of `main`, `main_no_geo`, `imagenet`, or `segmentation` as
appropriate. This is the canonical layout used by the maintained launchers.

## Environment variables (overridable)

| Var | Default | Meaning |
|---|---|---|
| `SDG_CONFIG` | `configs/paths.yaml` | Path to the paths file. |
| `PYTHON` | `python` | Python interpreter to use. |
| `CUDA_DEVICES` | (from paths.yaml) | GPU IDs. |
| `BATCH_SIZE_PER_GPU` | 2048 | Per-GPU batch for feature extraction. |
| `TYPE` | `Fuse` (regression) / per-script (extraction) | Modality target. |
| `FEATURE_LEN` | 768 | Width of each visual token for sampling regression; `Concat_spatial_self` contains four such tokens. |
