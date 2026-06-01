# Reproducing the main figures

This document maps each figure in the manuscript to the minimal set of commands required to regenerate it from the public Zenodo deposit.

> **Prerequisites:** `environment.yml` installed, `configs/paths.yaml` configured, public dataset unpacked under `data_root`.

## Figure 1 — Model performance

### Panel 1a (R² comparison across 11 SDGs)
Inputs: K-fold regression outputs for three models (Ours, ImageNet, Segmentation).

```bash
bash scripts/run_fold.sh                            # Ours
bash scripts/run_fold_imagenet.sh                   # ImageNet baseline
bash scripts/run_fold_segmentation.sh               # Segmentation baseline
python figures/_data_prep/prep_fig1a_r2_ours.py
python figures/_data_prep/prep_fig1a_r2_imagenet.py
python figures/_data_prep/prep_fig1a_r2_segmentation.py
jupyter nbconvert --execute figures/fig1/panel_a_r2_compare.ipynb
```

### Panel 1b–c (scaling curves by SDG and by city)
```bash
bash scripts/run_sampling.sh
jupyter nbconvert --execute figures/fig1/panel_b_sdg_scaling.ipynb
jupyter nbconvert --execute figures/fig1/panel_c_city_scaling.ipynb
```

### Panel 1d (feature-based vs random sampling)
```bash
bash scripts/run_ratio.sh                           # produces random-baseline curves
jupyter nbconvert --execute figures/fig1/panel_d_sampling_compare.ipynb
```

## Figure 2 — Mechanisms

### Panel 2a (modal comparison)
```bash
bash scripts/run_fold_single_modal.sh
python figures/_data_prep/prep_fig2a_modal_compare.py
jupyter nbconvert --execute figures/fig2/panel_a_modal.ipynb
```

### Panel 2b–c (CLIP-concept interpretability)
```bash
python src/analysis/clip_concept.py                 # produces sv_clip_scores.pkl
python figures/_data_prep/prep_fig2bc_clip_concept.py
jupyter nbconvert --execute figures/fig2/panel_bc_clip.ipynb
```

### Panel 2d–e (t-SNE of Los Angeles)
```bash
jupyter nbconvert --execute figures/fig2/panel_de_tsne.ipynb
```

### Panel 2f (spatial vs non-spatial R²)
```bash
bash scripts/run_sampling.sh                        # with-geo
bash scripts/run_sampling_no_geo.sh                 # without-geo baseline
jupyter nbconvert --execute figures/fig2/panel_f_spatial_curve.ipynb
```

### Panel 2g (Moran's I × R²)
```bash
python figures/_data_prep/prep_fig2g_moransi.py
jupyter nbconvert --execute figures/fig2/panel_g_moransi.ipynb
```

## Extended Data Figures

See `figures/extended/` — each ED notebook is self-contained and reads from `regression_outputs/`.

## Supplementary per-city figures

```bash
python figures/supplementary/batch_per_city_figures.py --config configs/paths.yaml
```

Produces all 57 per-city PDFs (`S_{Country}_{City}_{fold,random,strategic}.pdf`).

## Total compute budget

End-to-end reproduction of all figures from raw features (no pretraining): ~24 GPU-hours on a single A100. Pretraining adds ~5 days on 8 A100s.
