# Extended Data figure workflows

This directory contains both maintained reproduction scripts and analysis
notebooks retained from the original study workspace. They are separated below
so that a reviewer does not have to infer which historical paths are still
expected to work.

## Maintained workflows

### Full random and feature-guided sampling sweeps

Use:

```bash
python figures/extended/sampling_curves.py --strict
```

This script reads the canonical regression outputs configured by
`configs/paths.yaml`:

```text
ratio/main/<Country>/<City>/Fuse/Token_Concat_spatial_self/results.csv
sampling/main/<Country>/<City>/Fuse/Multi_Concat_pcahierachy/results.csv
```

It uses `all_r2` for the citywide partial-survey reconstruction metric and
writes:

```text
data/figure_assets/S_random_sampling.pdf
data/figure_assets/S_strategic_sampling.pdf
```

PNG copies are written alongside the PDFs. The historical
`ed_ratio_results.ipynb` and `ed_sampling_results.ipynb` notebooks were removed
after this maintained renderer replaced them.

### Other Extended Data panels

`ed_heatmap.py`, `ed_indicator_bars.py`, `ed_distribution.ipynb`,
`ed_maps.ipynb`, and `ed_world_map.ipynb` are retained because they operate on
prepared/deposited tables or geographic inputs rather than defining alternative
training configurations. Their inputs are described in the notebooks/scripts.

## Archival sensitivity analyses

The following notebooks are retained as records of additional sensitivity
analyses from the study workspace, but are **not** mapped onto the final-model
canonical output hierarchy because doing so mechanically would change the
experiment being plotted:

- `ed_sampling_panel_a.ipynb` — combines the historical sampling/ratio sweeps
  with a feature-entropy analysis based on an earlier fused representation.
- `ed_sampling_panel_b.ipynb` — compares historical `top-k` k-center variants
  that are not produced by the final canonical k-center launcher.
- `ed_sampling_panel_c.ipynb` — uses per-neighborhood historical prediction
  HDF5 files to compare low-income-area and citywide behavior.
- `ed_train_epoch.ipynb` — compares regressions run from many intermediate
  representation-pretraining checkpoints rather than only the final reported
  checkpoints.

These notebooks therefore should not be interpreted as one-command reproduction
scripts for the final pipeline. Where their prepared inputs are included in the
review/data package, they can still be inspected as analysis records. Replacing
their historical inputs with current final-model outputs would answer a
different question and is intentionally avoided.

## Typography

Some historical notebooks reference `../../data/Arial.ttf` to match manuscript
typography. That font is not distributed by this repository. Numerical results
do not depend on it; use a locally available sans-serif font if an exact local
copy is unavailable.
