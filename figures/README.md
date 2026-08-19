# `figures/` — figure regeneration

Each panel of each manuscript figure has its own notebook, so any single
panel can be re-run independently. Heavyweight data preparation lives
in `_data_prep/` and writes intermediate CSV/HDF5 tables that the panel
notebooks then load.

## Layout

```
figures/
├── _data_prep/        Data-prep notebooks (read regression outputs, write per-figure CSVs)
├── fig1/              Main Figure 1 — Model performance
├── fig2/              Main Figure 2 — Mechanisms of spatial-visual integration
├── extended/          Extended Data Figures
└── supplementary/     Per-city supplementary figures (batch-generated)
```

## Main Figure 1 (`fig:performance`) — 4 panels

| Panel | Notebook | Data prep |
|---|---|---|
| 1a — R² comparison (Ours / ImageNet / Segmentation) across 11 SDGs | `fig1/panel_a_r2_compare.ipynb` | `_data_prep/prep_fig1a_r2_{ours,imagenet,segmentation}.ipynb` |
| 1b — Scaling curves by SDG category | `fig1/panel_b_sdg_scaling.ipynb` | (inline) |
| 1c — Scaling curves by city | `fig1/panel_c_city_scaling.ipynb` | `_data_prep/prep_fig1c_spearman.ipynb` |
| 1d — Feature-based vs random sampling | `fig1/panel_d_sampling_compare.ipynb` (canonical) + `fig1/panel_d_dumbbell.py` (alt rendering) | (inline) |

Also: `fig1/scaling_curves.py` is an alternative standalone renderer
for the b/c panels.

## Main Figure 2 (`fig:reason`) — 5 panel groups (a, b–c, d–e, f, g)

| Panel | Notebook | Data prep |
|---|---|---|
| 2a — Modal comparison (SV / RS / Fuse) | `fig2/panel_a_modal.ipynb` | `_data_prep/prep_fig2a_modal.ipynb` |
| 2b–c — CLIP zero-shot concept R² | `fig2/panel_bc_clip.ipynb` | `_data_prep/prep_fig2bc_clip_concept.ipynb` |
| 2d–e — t-SNE of self vs spatial in LA | `fig2/panel_de_tsne.ipynb` | (loads raw features directly) |
| 2f — Spatial vs non-spatial R² curves | `fig2/panel_f_spatial_curve.ipynb` | (inline) |
| 2g — Moran's I × R² at 50% sampling | (see `_data_prep/prep_fig2g_moransi.ipynb`) | `_data_prep/prep_fig2g_moransi.ipynb` produces the data + figure |

## Extended Data Figures

| Figure | Notebook / script |
|---|---|
| Maps — global distribution | `extended/ed_maps.ipynb`, `extended/ed_world_map.ipynb` |
| Heatmap — city × SDG R² | `extended/ed_heatmap.py` |
| Indicator bars | `extended/ed_indicator_bars.py` |
| Distribution | `extended/ed_distribution.ipynb` |
| Random / strategic sampling — full sweeps | `extended/ed_sampling_results.ipynb`, `extended/ed_ratio_results.ipynb` |
| Train-epoch ablation | `extended/ed_train_epoch.ipynb` |
| Sampling panels (origin of Fig 3 in early drafts) | `extended/ed_sampling_panel_{a,b,c}.ipynb` |

## Supplementary Figures

`supplementary/batch_per_city_figures.py` regenerates the 57 per-city
PDFs (`S_{Country}_{City}_{fold,random,strategic}.pdf`) that the LaTeX
supplementary references. Reads from `${REGRESSION_OUT_DIR}/fold/`,
`ratio/`, and `sampling/`.

`supplementary/per_city_fold_box.py` and `per_city_fold_results.ipynb`
are auxiliary fold-level renderers.

## Path conventions inside the notebooks

All notebooks have been path-rewritten to read from `../../data/...`
relative to the notebook location (i.e. `data/` under the repository
root). Set up the data root once by symlinking your `${data_root}`
(from `configs/paths.yaml`) into the repo:

```bash
ln -s "${data_root}" data
```

The notebooks then read from:

| Logical | Path in notebooks |
|---|---|
| Pre-computed regression results (legacy server: `regmodels_*`) | `data/regression_outputs/regmodels_<variant>/...` |
| Newer regression results (legacy server: `models_new/`) | `data/regression_outputs_new/...` |
| Unit/Raw features | `data/features/{Unit,Raw}/...` |
| Processed label/path tables | `data/processed/...` |
| Pre-rendered figure assets (flags, fonts) | `data/figure_assets/...`, `data/assets/flags/...` |
| Raw street-view imagery (third-party, not redistributed) | `data/raw/GoogleSV/...` |
| Raw satellite imagery (third-party, not redistributed) | `data/raw/GoogleSatellite/...` |

The raw satellite imagery used in the paper was obtained through the Google Static Maps API at approximately 0.6 m spatial resolution; it is not Sentinel-2 imagery. Raw third-party imagery is not included in the reproducibility deposit.

If you prefer a different layout, set up the symlinks for those specific
subpaths instead. Notebook outputs (figures) save under `../../data/figure_assets/`
unless you change the `plt.savefig(...)` lines.

The `.py` renderers (`ed_heatmap.py`, `ed_indicator_bars.py`,
`panel_d_dumbbell.py`, `scaling_curves.py`, the supplementary scripts)
follow the same convention.

All notebook outputs have been stripped — re-running the notebooks
generates fresh outputs and keeps the git history clean.

## Dropped (orphans)

These were present in the legacy `0figure/` and `0figure_data_prepare/`
folders but excluded from the release:

- `fig1_sdg_r2_compare_self_token.ipynb`, `fig1_sdg_r2_compare_spatial_token.ipynb`
  — Fig 1a draws three bars (Ours / ImageNet / Segmentation), not five.
  The self-only and spatial-only data-prep notebooks produced unused CSVs.
- `fig1c.ipynb` — early three-way comparison sketch, `plt.show()`-only.
- `new.ipynb` — work-in-progress placeholder.
- `extended_tsne copy.ipynb`, `fig3_*_copy.ipynb` — name-duplicate files.
- `read_flag.py`, `read_nb.py`, `read_nb2.py` — notebook-conversion utilities,
  not figure code.
