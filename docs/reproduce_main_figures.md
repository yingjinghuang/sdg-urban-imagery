# Reproducing the main figures

This document maps the current manuscript figures to commands that exist in the repository and to the current canonical regression-output layout.

> **Prerequisites**
>
> 1. Create/activate the environment from `environment.yml`.
> 2. Copy `configs/paths.example.yaml` to `configs/paths.yaml` and configure the local data/output roots.
> 3. Unpack the deposited derived data/checkpoints as described in the top-level README.
> 4. The historical plotting notebooks use paths relative to `<repo>/data/`. `data/` may be a directory or a symlink to your local data bundle; it is ignored by Git.
>
> **Internal folder numbering:** the repository keeps the legacy directories `figures/fig1` and `figures/fig2` for compatibility with the original notebooks. In the current manuscript they correspond to **Fig. 2** and **Fig. 3**, respectively. Current manuscript Fig. 1 is the conceptual overview.

## Canonical outputs vs legacy plotting paths

The cleaned training launchers write to the canonical layout configured by `regression_out_dir`, for example:

```text
fold/main/<Country>/<City>/...
ratio/main/<Country>/<City>/...
sampling/main/<Country>/<City>/...
sampling/main_no_geo/<Country>/<City>/...
```

Some original plotting notebooks still read the historical `data/regression_outputs/regmodels_*` paths. After running regression, stage small compatibility CSVs with:

```bash
python scripts/prepare_figure_inputs.py --strict
```

This utility **does not retrain models or change the reported metrics**. It maps the canonical outputs to the filenames/schema expected by the legacy plotting notebooks and stages the small metadata tables they need. For fold experiments it writes the per-indicator mean across the five folds, which is the quantity consumed by the figure-level comparisons.

You can rerun this command whenever regression outputs change.

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

The same values are the defaults in `src/regression/token_reg.py`, but the launch scripts pass them explicitly so later default changes cannot silently alter the paper configuration.

## Notebook execution convention

The commands below execute notebooks in place:

```bash
jupyter nbconvert --to notebook --execute --inplace path/to/notebook.ipynb
```

The original plotting notebooks use `data/Arial.ttf` for exact manuscript typography. The font file is not distributed by this repository. If you need pixel-identical typography, provide a locally licensed copy at that path; otherwise use a locally available sans-serif font. The numerical results are unaffected.

---

## Manuscript Figure 2 — Model performance and sampling efficiency

### Panel 2a — R² comparison across 11 SDGs

Run the three fold regressions:

```bash
bash scripts/run_fold.sh
bash scripts/run_fold_imagenet.sh
bash scripts/run_fold_segmentation.sh
python scripts/prepare_figure_inputs.py --strict
```

Prepare the three series and render the panel. These are notebooks (`.ipynb`), not Python scripts:

```bash
jupyter nbconvert --to notebook --execute --inplace figures/_data_prep/prep_fig1a_r2_ours.ipynb
jupyter nbconvert --to notebook --execute --inplace figures/_data_prep/prep_fig1a_r2_imagenet.ipynb
jupyter nbconvert --to notebook --execute --inplace figures/_data_prep/prep_fig1a_r2_segmentation.ipynb
jupyter nbconvert --to notebook --execute --inplace figures/fig1/panel_a_r2_compare.ipynb
```

### Panels 2b–c — scaling curves by SDG and by city

If the deposited feature-guided split definitions are already present, the `select_kcenter.sh` step can be skipped. Otherwise generate them first.

```bash
bash scripts/run_ratio.sh
bash scripts/select_kcenter.sh          # skip if pcahierachy.pkl is already deposited
bash scripts/run_sampling.sh
python scripts/prepare_figure_inputs.py --strict

jupyter nbconvert --to notebook --execute --inplace figures/fig1/panel_b_sdg_scaling.ipynb
jupyter nbconvert --to notebook --execute --inplace figures/fig1/panel_c_city_scaling.ipynb
```

### Panel 2d — feature-guided vs random sampling

This panel uses the same ratio and feature-guided runs as Panels 2b–c:

```bash
bash scripts/run_ratio.sh
bash scripts/select_kcenter.sh          # skip if pcahierachy.pkl is already deposited
bash scripts/run_sampling.sh
python scripts/prepare_figure_inputs.py --strict

jupyter nbconvert --to notebook --execute --inplace figures/fig1/panel_d_sampling_compare.ipynb
```

The random and feature-guided regressions use the same four 768-d visual representation branches; only the neighborhood-selection strategy changes.

---

## Manuscript Figure 3 — Roles of visual scale, learning objective, and spatial context

### Panel 3a — modal comparison

```bash
bash scripts/run_fold.sh
bash scripts/run_fold_single_modal.sh
python scripts/prepare_figure_inputs.py --strict

jupyter nbconvert --to notebook --execute --inplace figures/_data_prep/prep_fig2a_modal.ipynb
jupyter nbconvert --to notebook --execute --inplace figures/fig2/panel_a_modal.ipynb
```

### Panels 3b–c — CLIP concept probing

For figure-only reproduction, use the deposited prepared concept tables:

```text
data/processed/fig/clip_concept_sv.csv
data/processed/fig/clip_concept_rs.csv
```

Then run:

```bash
jupyter nbconvert --to notebook --execute --inplace figures/fig2/panel_bc_clip.ipynb
```

Recomputing CLIP scores from the original imagery is a separate, substantially heavier route because the raw Google imagery cannot be redistributed by the authors. The repository retains `src/analysis/clip_concept.py` and `scripts/compute_clip_scores.sh` for users who independently reconstruct the imagery dataset, but those steps are not required to regenerate the published panel from the deposited derived tables.

### Panels 3d–e — t-SNE analysis of Los Angeles

The plotting notebook reads the deposited prepared Los Angeles representation tables:

```bash
jupyter nbconvert --to notebook --execute --inplace figures/fig2/panel_de_tsne.ipynb
```

### Panels 3f–g — spatial vs non-spatial reconstruction and Moran's I

Run the feature-guided model with and without the geographic-coordinate token:

```bash
bash scripts/select_kcenter.sh          # skip if pcahierachy.pkl is already deposited
bash scripts/run_sampling.sh
bash scripts/run_sampling_no_geo.sh
```

Prepare the joint spatial/Moran table with the maintained Python script:

```bash
python figures/_data_prep/prep_fig2g_moransi.py
```

Then execute the plotting notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace figures/fig2/panel_f_spatial_curve.ipynb
```

Despite its historical filename, `panel_f_spatial_curve.ipynb` contains both the spatial-vs-non-spatial scaling plot (current Fig. 3f) and the Moran's-I correlation plot (current Fig. 3g). There is no separate `panel_g_moransi.ipynb`; older documentation referring to that filename was incorrect.

The older `figures/_data_prep/prep_fig2g_moransi.ipynb` is retained as a legacy record. For current reproduction use the maintained `.py` script above, which reads the canonical `sampling/main` and `sampling/main_no_geo` outputs and writes:

```text
data/processed/fig/fig2_geo_reg_moransi_results.csv
```

---

## Extended Data Figures

See `figures/extended/`. These files are retained from the original analysis workspace and are not all yet normalized to the cleaned canonical output layout. For the main-paper reproducibility check, use the maintained commands above. Extended Data notebook cleanup is tracked separately from the main-figure reproduction path.

## Supplementary per-city figures

The maintained batch generator is:

```bash
python figures/supplementary/batch_per_city_figures.py --config configs/paths.yaml
```

It produces the per-city fold, random-sampling, and feature-guided-sampling figures from the configured regression outputs.

## Compute scope

Reproducing figures from the deposited neighborhood-level representations and regression outputs does **not** require rerunning imagery pretraining. Full pretraining from raw imagery is substantially more expensive and requires independent access to the third-party imagery APIs. Pretrained checkpoints are therefore included in the data release/reviewer package so reviewers can reproduce the downstream analyses without reacquiring the original imagery.
