# Extended Data figure workflows

This directory contains a mix of maintained reproduction scripts and original
analysis notebooks. They are classified here so a reviewer does not have to
infer which historical workspace assumptions are still expected to work.

## Maintained workflows

### Full random and feature-guided sampling sweeps

```bash
python figures/extended/sampling_curves.py --strict
```

Reads the canonical regression outputs configured by `configs/paths.yaml`:

```text
ratio/main/<Country>/<City>/Fuse/Token_Concat_spatial_self/results.csv
sampling/main/<Country>/<City>/Fuse/Multi_Concat_pcahierachy/results.csv
```

The script uses `all_r2` for citywide partial-survey reconstruction and writes
`S_random_sampling.{pdf,png}` and `S_strategic_sampling.{pdf,png}` under
`data/figure_assets/` by default. The older workspace-specific random and
strategic sampling notebooks were removed after this renderer replaced them.

### City-by-SDG heatmap

```bash
python figures/extended/ed_heatmap.py --strict
```

Reads canonical `fold/main/.../Token_Concat_spatial_self/results.csv` files,
averages held-out R² across folds/indicators within city-SDG pairs, and obtains
SDG metadata from `<processed_dir>/0labels/<Country>.csv`. Output paths are
repository-relative/configurable; machine-specific Windows paths and required
local fonts were removed.

### Representative-city indicator bars

```bash
python figures/extended/ed_indicator_bars.py
```

Reads the same canonical fold outputs for Philadelphia, Melbourne, Rio de
Janeiro, and Hong Kong, averages the five fold rows per indicator, and uses the
country metadata tables for indicator labels/SDG groups. It no longer requires
machine-specific font, flag, or output paths.

## Original/prepared-data notebooks

`ed_maps.ipynb`, `ed_world_map.ipynb`, and `ed_distribution.ipynb` are retained
as analysis notebooks around geographic/processed inputs. They are not claimed
here as canonical one-command scripts; their data cells document the prepared
inputs they were developed with. Main-paper reproduction does not depend on
rerunning them.

## Archival sensitivity analyses

The following notebooks are retained as records of additional sensitivity
analyses but are **not** mapped onto the final-model canonical output hierarchy,
because doing so mechanically would change the experiment being plotted:

- `ed_sampling_panel_a.ipynb` — combines historical sampling/ratio sweeps with
  a feature-entropy analysis based on an earlier fused representation.
- `ed_sampling_panel_b.ipynb` — compares historical `top-k` k-center variants
  that are not produced by the final canonical k-center launcher.
- `ed_sampling_panel_c.ipynb` — uses historical per-neighborhood prediction
  HDF5 layouts to compare low-income-area and citywide behavior.
- `ed_train_epoch.ipynb` — compares regressions run from many intermediate
  representation-pretraining checkpoints rather than only the final reported
  checkpoints.

These notebooks should not be interpreted as one-command reproduction scripts
for the final pipeline. Where their prepared inputs are available in the data
package, they can still be inspected as analysis records. Replacing their
historical inputs with current final-model outputs would answer a different
question and is intentionally avoided.

## Typography

Some retained notebooks reference `../../data/Arial.ttf` to match manuscript
typography. That font is not distributed by this repository. Numerical results
do not depend on it; use a locally available sans-serif font if an exact local
copy is unavailable.
