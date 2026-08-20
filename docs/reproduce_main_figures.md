# Reproducing the main figures

This document maps the current manuscript figures to the minimal commands required to regenerate them from the Zenodo deposit.

> **Prerequisites:** `environment.yml` installed, `configs/paths.yaml` configured, and the deposited derived data unpacked under `data_root`.
>
> **Internal folder numbering:** the repository keeps the legacy directories `figures/fig1` and `figures/fig2` for compatibility with existing notebooks. In the current manuscript they correspond to **Fig. 2** and **Fig. 3**, respectively. Current manuscript Fig. 1 is the conceptual overview.

## Regression settings used for paper reproduction

All paper-reproduction launchers explicitly use the regression configuration reported in Supplementary Table 9:

- hidden dimension: 256
- attention heads: 8
- Transformer layers: 1
- dropout before the prediction head: 0.5
- optimizer: Adam
- initial learning rate: `1e-4`
- cosine learning-rate schedule with 10-epoch warmup
- weight decay: `1e-4`
- batch size: 128
- maximum epochs: 100
- early-stopping patience: 5
- early-stopping delta: 0.005

The same values are the defaults in `src/regression/token_reg.py`, but the launch scripts pass them explicitly to keep the reproduction configuration stable.

## Manuscript Figure 2 — Model performance

### Panel 2a (R² comparison across 11 SDGs)

Inputs: 5-fold regression outputs for the proposed representation and the ImageNet and semantic-segmentation baselines.

```bash
bash scripts/run_fold.sh                            # Ours
bash scripts/run_fold_imagenet.sh                   # ImageNet baseline
bash scripts/run_fold_segmentation.sh               # Segmentation baseline
python figures/_data_prep/prep_fig1a_r2_ours.py
python figures/_data_prep/prep_fig1a_r2_imagenet.py
python figures/_data_prep/prep_fig1a_r2_segmentation.py
jupyter nbconvert --execute figures/fig1/panel_a_r2_compare.ipynb
```

### Panels 2b–c (scaling curves by SDG and by city)

```bash
bash scripts/run_ratio.sh                           # random-sampling curves
bash scripts/run_sampling.sh                        # feature-guided sampling curves
jupyter nbconvert --execute figures/fig1/panel_b_sdg_scaling.ipynb
jupyter nbconvert --execute figures/fig1/panel_c_city_scaling.ipynb
```

### Panel 2d (feature-guided vs random sampling)

```bash
bash scripts/run_ratio.sh                           # random baseline
bash scripts/run_sampling.sh                        # feature-guided sampling
jupyter nbconvert --execute figures/fig1/panel_d_sampling_compare.ipynb
```

## Manuscript Figure 3 — Roles of visual scale, learning objective, and spatial context

### Panel 3a (modal comparison)

```bash
bash scripts/run_fold.sh                            # fused representation
bash scripts/run_fold_single_modal.sh               # SV-only and RS-only
python figures/_data_prep/prep_fig2a_modal_compare.py
jupyter nbconvert --execute figures/fig2/panel_a_modal.ipynb
```

### Panels 3b–c (CLIP-concept probing)

```bash
python src/analysis/clip_concept.py                 # produces sv_clip_scores.pkl
python figures/_data_prep/prep_fig2bc_clip_concept.py
jupyter nbconvert --execute figures/fig2/panel_bc_clip.ipynb
```

### Panels 3d–e (t-SNE of Los Angeles)

```bash
jupyter nbconvert --execute figures/fig2/panel_de_tsne.ipynb
```

### Panel 3f (spatial vs non-spatial reconstruction R²)

```bash
bash scripts/run_sampling.sh                        # with geo
bash scripts/run_sampling_no_geo.sh                 # without-geo baseline
jupyter nbconvert --execute figures/fig2/panel_f_spatial_curve.ipynb
```

### Panel 3g (Moran's I × R²)

```bash
python figures/_data_prep/prep_fig2g_moransi.py
jupyter nbconvert --execute figures/fig2/panel_g_moransi.ipynb
```

## Extended Data Figures

See `figures/extended/`. Each Extended Data notebook reads the relevant prepared regression outputs.

## Supplementary per-city figures

```bash
python figures/supplementary/batch_per_city_figures.py --config configs/paths.yaml
```

Produces the per-city fold, random-sampling, and feature-guided-sampling figures.

## Compute budget

End-to-end reproduction from the deposited derived features does not require rerunning imagery pretraining. Pretraining from raw imagery is substantially more expensive and also requires independent access to the third-party imagery APIs; pretrained checkpoints are therefore included in the Zenodo deposit.
