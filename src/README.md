# `src/` — maintained pipeline stages

The reproducibility route for the paper starts from the deposited processed
neighborhood data and pretrained checkpoints. Country-specific ingestion from
raw national statistical sources is not maintained as a single portable
pipeline because the source formats, identifiers, and access conditions differ
substantially by country; see `preprocess/README.md` and the Supplementary
Information for that boundary.

The maintained code is organized as follows:

| Stage | What it does | Main inputs | Main outputs |
|---|---|---|---|
| `preprocess/` | Portable preprocessing helpers, including indicator scaling and satellite tiling utilities. | Deposited/locally prepared neighborhood tables matching `docs/data_schema.md`. | Normalized labels and helper tables used downstream. |
| `datasets/` | Construct self/spatial contrastive-training manifests and image pairs, including same-neighborhood street-view positives and adjacent satellite positives. | Modality-specific image manifests plus neighborhood membership. | Pretraining pair/manifests consumed by `pretrain/moco_{sv,rs}.py`. |
| `extract/` | Extract per-image ViT-B features from pretrained checkpoints and mean-aggregate them to neighborhoods. | Raw imagery, modality-specific manifests, pretrained checkpoints. | `features/Raw/.../*.h5` and `features/Unit/.../*.pkl`. |
| `fuse/` | Combine street-view/satellite and self/spatial feature branches. | Neighborhood-level feature tables. | `features/Unit/{Country}/{City}/Fuse/Concat_*.pkl`. |
| `regression/` | Train the location-aware token regressor for five-fold, random partial-survey, and feature-guided partial-survey experiments. | Four visual branches, labels, split definitions, geographic coordinates. | Canonical `fold/main`, `ratio/main`, and `sampling/main` result hierarchies under `regression_out_dir`. |
| `analysis/` | Post-hoc representation analyses such as CLIP concept scoring and residual analysis. | Raw imagery/features or deposited prepared analysis tables. | Inputs supporting current manuscript Fig. 3b–e and related analyses. |

The main regression representation is `Concat_spatial_self.pkl`: four separate
768-dimensional visual branches (SV-spatial, RS-spatial, SV-self, RS-self).
The feature-guided sampling selector starts from the same four unreduced
branches, applies standardization/PCA separately by branch, and then combines
them with weighted geographic coordinates before k-center selection.

Bash launchers under `../scripts/` wrap the maintained stages and read paths
from `../configs/paths.yaml`. The 20 configured country/region bundles in
`../configs/cities.yaml` collectively represent the 93 cities reported in the
manuscript; France and Portugal are stored as country-level bundles and resolved
by the `city` field in their neighborhood tables.

For exact file schemas see `../docs/data_schema.md`; for panel-by-panel
reproduction commands see `../docs/reproduce_main_figures.md`.
