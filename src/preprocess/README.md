# `src/preprocess/` — country-level data preparation

The preprocess stage produces, for each (Country, City) pair, the
canonical artifacts that the rest of the pipeline consumes:

| File | Schema |
|---|---|
| `labels.pkl` | raw per-neighborhood indicators (one column per SDG indicator code, `GEOID`, `geometry`) |
| `labels_norm.pkl` | z-scored labels with |z|>3 outliers masked to NaN; sub-50-non-NaN columns dropped |
| `scalers.joblib` | fitted `StandardScaler` per indicator (for inversion) |
| `paths.pkl` | image-to-neighborhood manifest for street view |
| `rs_paths.pkl` | same, for satellite |
| `geo.pkl` | `GEOID`, `geometry` only — used for centroid coordinates in regression |
| `samples/fold5.pkl` / `samples/ratio.pkl` / `samples/pcahierachy.pkl` | train/test split definitions |

## What this directory provides

Three portable, country-agnostic steps:

| Script | Replaces (legacy) | Purpose |
|---|---|---|
| `compute_country_tiles.py` | `1preprocess_country_raw/{fr,pt,hk,ng}_boundary.ipynb` | Convert country polygons → satellite-download bounding-box manifest. |
| `scale_labels.py` | `1preprocess/label_scale_all.py`, `label_scale_{au,br,ch,fr,pt,ng,us}.ipynb` | Z-score outliers + standardize per indicator; reads per-country target list from `configs/cities.yaml`. |
| (existing) `src/datasets/spatial_contrastive.py` | `2build_train_datasets/spatial_contrastive_city{,_rs}.py` | Build same-neighborhood image-pair training set. |

## What this directory does NOT provide

The **raw-source-specific ingestion** that turns each country's source
data (US ACS XLS, AU ABS DataPacks, BR IBGE microdata, etc.) into
`labels.pkl`. These notebooks live in the original repo under
`1preprocess_country_raw/{br,fr,hk,ng,pt}.ipynb` and `1preprocess/label_scale_{country}.ipynb`,
and are intrinsically per-country (different sources, different schemas).

> **For reviewers / users reproducing the paper:** the processed
> `labels.pkl` for all 20 region-folders is included in the public
> Zenodo deposit, so you do not need to re-run the per-country
> ingestion. The pipeline can be reproduced starting from these files.

If you do want to re-ingest from raw national data sources, the original
country-specific notebooks are preserved at:

```
<legacy_repo>/scripts_sdg/1preprocess/label_scale_{country}.ipynb
<legacy_repo>/scripts_sdg/1preprocess_country_raw/{country}.ipynb
```

These were not migrated because their value is documentary (showing
which census columns map to which SDG indicators) rather than functional,
and porting 13 large, divergent notebooks into homogeneous .py would
either be near-mechanical (and thus low-value) or would risk altering the
indicator mappings in ways that diverge from the published values.

## Spatial encoding (excluded)

`scripts_sdg/3_3spatial/spatial.ipynb` builds an SVD-reduced adjacency-
matrix embedding and writes it to `Unit/.../Spatial/{queen,rook}.pkl`.
This output is **not consumed** by any current pipeline stage
(regression takes spatial information from the `geo.pkl` centroids;
spatial-contrastive pretraining samples pairs at the GEOID level).
Treated as deprecated and excluded from the release.
