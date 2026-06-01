# Data schema

Schema reference for every derived artifact in the public Zenodo deposit. Use this alongside `src/README.md` (which describes how each artifact is produced).

## `labels/{Country}.csv` — sustainability indicators

| Column | Type | Description |
|---|---|---|
| `GEOID` | str / int | Unique neighborhood identifier (block group for US, SA2 for Australia, custom for others). |
| `city` | str | City name (used to disaggregate country-level bundles for France and Portugal). |
| `geometry` | WKT (optional) | Polygon of the neighborhood (omitted in pkl version). |
| `<indicator codes>` | float | Country-specific indicator values; see code dictionaries in `processed/0labels/{Country}.csv`. |

## `processed/{Country}/{City}/paths.pkl` — image-to-neighborhood manifest

| Column | Type | Description |
|---|---|---|
| `path` | str | Absolute path to one street-view or satellite image. |
| `GEOID` | str | Neighborhood the image falls within. |
| `lat`, `lon` | float | Image capture location. |
| `city` | str | City within the country bundle (FR/PT only). |

## `features/Raw/{Country}/{City}/{SV,RS}/Mocov3VITB-{variant}-{Country}-{City}-ep{N}.h5` — per-image features

HDF5 with one key `data`. Two columns: `index` (image path) and 768 unnamed feature columns (ViT-B/16 final-layer mean-pooled embedding). Shape: `(N_images, 769)`.

| `variant` | Pretraining objective |
|---|---|
| `self` | Self-contrastive (MoCo-v3 baseline) |
| `spatial` | Spatial-contrastive (this paper's variant) |

Per-modality file sizes range from ~150 MB (small city) to ~4 GB (France-All SV).

## `features/Unit/{Country}/{City}/{SV,RS}/Mocov3VITB-{variant}-{Country}-{City}-ep{N}.pkl` — neighborhood features

Pickled `pandas.DataFrame`. Columns:

| Column | Description |
|---|---|
| `GEOID` | Neighborhood ID — matches `labels` |
| `0`, `1`, …, `767` | Mean-aggregated 768-dim embedding |
| `city` | City within country (FR/PT) |

## `features/Unit/{Country}/{City}/{SV,RS}/ImageNet.pkl`

Same schema as above but produced from an ImageNet-pretrained ViT-B (no contrastive finetuning). Used as the Fig 1a baseline.

## `features/Unit/{Country}/{City}/SV/segmentation.pkl`

Pickled `DataFrame`. 150-dim per neighborhood, where each dimension is the mean fraction of pixels in one ADE20K class across all images in the neighborhood. Produced from per-image segmentation predictions by `src/extract/aggregate_to_unit.py --arch segmentation`.

## `features/Unit/{Country}/{City}/Fuse/Concat_*.pkl`

Concatenated features across modalities (SV ⊕ RS). Variants:

| Variant | Composition | Dim |
|---|---|---|
| `Concat_self` | self-SV ⊕ self-RS | 1536 |
| `Concat_spatial` | spatial-SV ⊕ spatial-RS | 1536 |
| `Concat_spatial_self` | (spatial+self)-SV ⊕ (spatial+self)-RS | 3072 |
| `Concat_ImageNet` | ImageNet-SV ⊕ ImageNet-RS | 1536 |
| `Concat_spatial_self_pca99` | `Concat_spatial_self` with PCA to 99% variance | ~500 |

`Concat_spatial_self` is the main framework feature consumed by `scripts/run_fold.sh` and downstream sampling experiments.

## `regression_outputs/{Fold,Ratio,Sampling}/.../results.csv`

Per-target regression metrics:

| Column | Type | Description |
|---|---|---|
| `target` | str | SDG indicator code |
| `r2` | float | Test R² (mean across folds for Fold runs) |
| `mae` | float | Mean absolute error |
| `mse` | float | Mean squared error |
| `n_train`, `n_test` | int | Sample sizes |

## `regression_outputs/.../results.h5`

Per-neighborhood predictions, used for residual analysis and spatial-error mapping. Keys `F0`–`F4` for the 5 folds; each is a DataFrame with `GEOID`, `set` (train/test), and `pred_<target>` / `<target>` columns.

## City naming

In all `Country` and `City` filename fields, the strings are kebab-free CamelCase:
- `LosAngeles`, `NewYork`, `SanFrancisco`, `BeloHorizonte`, `RiodeJaneiro`, `PortoAlegre`, `HongKong`.

Country-bundled regions:
- `France/All`, `Portugal/All`.
