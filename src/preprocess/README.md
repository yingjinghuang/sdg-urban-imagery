# `src/preprocess/` — data preparation

This directory contains the maintained preprocessing utilities used by the
current repository. For each `(Country, City)` pair, the downstream pipeline
expects the following canonical files:

| File | Schema / role |
|---|---|
| `labels.pkl` | Per-neighborhood sustainability indicators plus `GEOID` and geometry |
| `labels_norm.pkl` | Standardized labels used by regression |
| `scalers.joblib` | Fitted label scalers, where applicable |
| `paths.pkl` | Street-view image-to-neighborhood manifest |
| `rs_paths.pkl` | Satellite image-to-neighborhood manifest |
| `geo.pkl` | `GEOID` and neighborhood geometry used for geographic coordinates |
| `samples/fold5.pkl` | Five-fold train/test definitions |
| `samples/ratio.pkl` | Random partial-survey train/test definitions |
| `samples/pcahierachy.pkl` | Feature-guided partial-survey train/test definitions |

## Maintained preprocessing utilities

### `compute_country_tiles.py`

Builds satellite-download bounding-box manifests from country/city boundary
polygons.

### `scale_labels.py`

Standardizes the country-specific sustainability indicators used by the
regression pipeline. Target lists are read from `configs/cities.yaml`.

### `src/datasets/spatial_contrastive.py`

Builds the image-pair datasets used for spatial-contrastive pretraining.

## Reproducibility entry point

The raw national indicator sources differ substantially across countries in
file format, geographic identifiers, variable naming, and access conditions.
The country-specific source-ingestion notebooks used during dataset construction
are therefore not maintained as part of this cleaned reproducibility repository.

For reproduction of the reported analyses, start from the processed
neighborhood-level files supplied in the Zenodo deposit. In particular, the
deposited `labels.pkl` / processed indicator tables preserve the exact indicator
values used in the study, while the maintained code in this repository covers
standardization, representation processing, regression, sampling analyses, and
figure reproduction.

This separation is intentional: reimplementing heterogeneous national-source
ingestion is not required to reproduce the reported downstream results and
could introduce differences in indicator definitions or source-version handling.
Researchers who wish to rebuild indicators from the original national sources
should follow the source descriptions and variable definitions given in the
manuscript and Supplementary Information, then produce files matching the schema
documented in `docs/data_schema.md`.

## Notes on imagery manifests

Street-view and satellite imagery use separate manifests (`paths.pkl` and
`rs_paths.pkl`, respectively). Feature extraction and neighborhood aggregation
must use the same modality-specific manifest. Raw Google imagery is not
redistributed; the downstream reproducibility workflow uses the deposited
derived representations instead.
