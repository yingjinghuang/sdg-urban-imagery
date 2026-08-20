# Data schema

Schema reference for the derived artifacts in the Zenodo deposit. Use this alongside `src/README.md` (which describes how each artifact is produced).

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

HDF5 containing an `index` column (image path) and 768 feature columns from the ViT-B encoder.

| `variant` | Pretraining objective |
|---|---|
| `self` | Self-contrastive (MoCo-v3) |
| `spatial` | Spatial-contrastive |

Per-modality file sizes vary substantially with city/image count.

## `features/Unit/{Country}/{City}/{SV,RS}/Mocov3VITB-{variant}-{Country}-{City}-ep{N}.pkl` — neighborhood features

Pickled `pandas.DataFrame`. Columns:

| Column | Description |
|---|---|
| `GEOID` | Neighborhood ID — matches `labels` |
| `0`, `1`, …, `767` | Mean-aggregated 768-dim embedding |
| `city` | City within country (FR/PT) |

## `features/Unit/{Country}/{City}/{SV,RS}/ImageNet.pkl`

Pickled `pandas.DataFrame` produced from the manuscript's ImageNet baseline: a torchvision ResNet-50 pretrained on ImageNet-1K with the classification head removed. Each neighborhood contains the mean of the 2,048-dimensional global-average-pooling features over its constituent images. Used for the ImageNet comparison in current manuscript Fig. 2a.

## `features/Unit/{Country}/{City}/SV/segmentation.pkl`

Pickled `DataFrame`. 150-dim per neighborhood, where each dimension is the mean fraction of pixels in one ADE20K class across all images in the neighborhood. Produced from per-image segmentation predictions by `src/extract/aggregate_to_unit.py --arch segmentation`.

## `features/Unit/{Country}/{City}/Fuse/Concat_*.pkl`

Concatenated features across modalities (SV ⊕ RS). Variants:

| Variant | Composition | Dim |
|---|---|---:|
| `Concat_self` | self-SV ⊕ self-RS | 1536 |
| `Concat_spatial` | spatial-SV ⊕ spatial-RS | 1536 |
| `Concat_spatial_self` | spatial-SV ⊕ spatial-RS ⊕ self-SV ⊕ self-RS | 3072 |
| `Concat_ImageNet` | ResNet50(ImageNet)-SV ⊕ ResNet50(ImageNet)-RS | 4096 |
| `Concat_spatial_self_pca99` | optional global-PCA derivative of `Concat_spatial_self` | data-dependent |

`Concat_spatial_self` is the main four-branch representation used by the fold, random-sampling, and feature-guided-sampling regression launchers. In token regression it is interpreted as four consecutive 768-dimensional visual tokens.

For the paper's feature-guided sample selection, `scripts/select_kcenter.sh` also starts from the unreduced `Concat_spatial_self.pkl`. `src/regression/select_kcenter.py` standardizes and applies PCA **separately to each 768-dimensional branch**, retaining 99% variance per branch, then concatenates the reduced branches with weighted standardized centroid coordinates before k-center selection. The optional `Concat_spatial_self_pca99.pkl` global-PCA derivative is retained for legacy/auxiliary analysis and is **not** an input to the paper's k-center selection pipeline.

## `processed/{Country}/{City}/samples/pcahierachy.pkl`

Feature-guided sampling definitions produced by `scripts/select_kcenter.sh`. Columns:

| Column | Description |
|---|---|
| `GEOID` | Neighborhood identifier |
| `geometry` | Neighborhood geometry |
| `set01` … `set09` | Nested 10% … 90% sampling definitions; `train` denotes selected/surveyed neighborhoods and `test` denotes unsampled neighborhoods to be estimated |

The ordering is generated once by k-center greedy using seed 42; lower sampling ratios are prefixes of the same ordering.

## `regression_outputs/{Fold,Ratio,Sampling}/.../results.csv`

Per-target regression metrics:

| Column | Type | Description |
|---|---|---|
| `target` | str | SDG indicator code |
| `r2` | float | Held-out-neighborhood R² for that split/fold |
| `mae` | float | Mean absolute error |
| `mse` | float | Mean squared error |
| `all_r2` | float | Citywide reconstruction R² when observed/train neighborhoods retain their ground-truth values and held-out neighborhoods use model estimates |

## `regression_outputs/.../results.h5`

Per-neighborhood predictions, used for residual analysis and spatial-error mapping. Each split is stored under its split key and contains the original targets, the train/test membership, and `pred_<target>` columns.

## City naming

In all `Country` and `City` filename fields, the strings are kebab-free CamelCase:
- `LosAngeles`, `NewYork`, `SanFrancisco`, `BeloHorizonte`, `RiodeJaneiro`, `PortoAlegre`, `HongKong`.

Country-bundled regions:
- `France/All`, `Portugal/All`.
