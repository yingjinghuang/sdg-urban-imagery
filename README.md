# Urban imagery reveals multidimensional sustainability of neighborhoods across global cities

Code and data-release repository for the paper.

> Huang Y., Li Y., Zhang F., et al. "Urban imagery reveals multidimensional sustainability of neighborhoods across global cities." *(Under review)*.

This repository provides the full pipeline to reproduce the main and supplementary analyses of the paper, from raw imagery preprocessing through self/spatial-contrastive representation learning to SDG indicator regression and figure generation.

---

## Repository layout

```
sdg-urban-imagery/
├── configs/                 Path and city manifests (edit paths.yaml for your machine)
├── pretrain/                Self/spatial-contrastive pretraining (derived from facebookresearch/moco-v3)
├── src/
│   ├── preprocess/          Country-level label preparation, scaling, geo-joining
│   ├── datasets/            Spatial-contrastive training-dataset construction
│   ├── extract/             Per-image feature extraction + neighborhood aggregation
│   ├── fuse/                Multi-modal feature fusion (SV ⊕ RS)
│   ├── analysis/            CLIP-concept probing, residual analysis
│   └── regression/          Token regression (Fold / Ratio / Sampling experiments)
├── scripts/                 Bash launchers (read paths from configs/paths.yaml)
├── figures/
│   ├── _data_prep/          Per-panel data preparation (.py)
│   ├── fig1/                Legacy internal folder for current manuscript Fig. 2 panels
│   ├── fig2/                Legacy internal folder for current manuscript Fig. 3 panels
│   ├── extended/            Extended Data Figures
│   └── supplementary/       Per-city Supplementary Figures
├── data/README.md           Data access (Zenodo DOI, reviewer access, raw-source pointers)
└── docs/
    ├── reproduce_main_figures.md
    └── data_schema.md
```

> **Figure-folder note.** The internal `figures/fig1` and `figures/fig2` directory names are retained for compatibility with the analysis notebooks. In the current manuscript, these correspond to **Fig. 2 (model performance)** and **Fig. 3 (mechanisms / representation analyses)**, respectively. Manuscript Fig. 1 is the conceptual overview.

---

## Data and model availability

| Asset | Size | Location | Access |
|---|---|---|---|
| Neighborhood features, processed labels, sampling splits, regression results | ~14 GB | Zenodo — DOI: [10.5281/zenodo.20483916](https://doi.org/10.5281/zenodo.20483916) | Restricted during peer review; public on acceptance |
| Per-image raw features (`Raw/`) | ~142 GB | — | Available on request |
| Trained regression weights (`*.pth.tar`) and per-neighborhood predictions (`results.h5`) | ~150 GB | — | Available on request |
| Pretrained MoCo-v3 ViT-B backbones | ~6 GB | Zenodo — DOI: [10.5281/zenodo.20483916](https://doi.org/10.5281/zenodo.20483916) | Restricted during peer review; public on acceptance |
| Raw satellite imagery | — | Google Static Maps API, ~0.6 m resolution | Via Google Maps Platform API access; not redistributed |
| Raw street view imagery | — | Google Street View Static API | Via API key; not redistributed |
| Sustainability indicators (raw) | — | Per-country statistical agencies | See `docs/data_schema.md` |

The Zenodo record is restricted during peer review. Editors and reviewers can be given a private Zenodo reviewer-access link by the corresponding author. The record will be made public upon acceptance.

The original satellite and street-view imagery is not redistributed because it was obtained through third-party APIs. The deposited neighborhood-level representations and processed tables are sufficient for reproducing the downstream regression, sampling, and figure-generation analyses without redistributing the original imagery.

---

## Reported regression configuration

The paper-reproduction launchers explicitly pin the downstream regression settings reported in Supplementary Table 9:

| Setting | Value |
|---|---:|
| Hidden dimension | 256 |
| Attention heads | 8 |
| Transformer layers | 1 |
| Dropout before prediction head | 0.5 |
| Optimizer | Adam |
| Initial learning rate | 1e-4 |
| Learning-rate schedule | cosine annealing, 10-epoch warmup |
| Weight decay | 1e-4 |
| Batch size | 128 |
| Maximum epochs | 100 |
| Early-stopping patience | 5 |
| Early-stopping delta | 0.005 |

These values are also the defaults in `src/regression/token_reg.py`. The launchers pass them explicitly so that the paper-reproduction configuration remains stable if generic defaults are changed in the future.

---

## Reproducing the headline results

### 1. Set up environment

```bash
conda env create -f environment.yml
conda activate sdg
```

### 2. Configure paths

Copy `configs/paths.example.yaml` to `configs/paths.yaml` and point each entry at your local data root. All scripts read paths from this file.

### 3. Pull the public dataset

```bash
# Restricted-access record during peer review; editors/reviewers can use
# the private Zenodo reviewer-access link provided by the corresponding author.
# Once openly accessible after acceptance:
#   zenodo_get 10.5281/zenodo.20483916 -o ./data
```

### 4. Run the regression pipeline (current manuscript Figs. 2–3)

```bash
bash scripts/run_fold.sh             # 5-fold explanatory analysis; manuscript Fig. 2a / Fig. 3a
bash scripts/run_ratio.sh            # random-sampling scaling; manuscript Fig. 2b-d
bash scripts/run_sampling.sh         # feature-guided sampling; manuscript Fig. 2b-d / Fig. 3f-g
bash scripts/run_fold_single_modal.sh
bash scripts/run_sampling_no_geo.sh  # non-spatial baseline for manuscript Fig. 3f
```

For the visual baselines in manuscript Fig. 2a:

```bash
bash scripts/run_fold_imagenet.sh
bash scripts/run_fold_segmentation.sh
```

### 5. Regenerate figures

```bash
jupyter notebook figures/fig1/   # legacy folder name; current manuscript Fig. 2
jupyter notebook figures/fig2/   # legacy folder name; current manuscript Fig. 3
```

Each panel notebook reads its inputs from `regression_outputs/` and writes a panel PDF to `figures/_out/`. See `docs/reproduce_main_figures.md` for the full per-figure mapping.

---

## Optional: rerun pretraining from scratch

Pretraining requires the raw imagery, which is not redistributed because it is obtained through third-party APIs. Satellite imagery in the paper was obtained through the Google Static Maps API at approximately 0.6 m spatial resolution, while street-view imagery was obtained through the Google Street View Static API. Pretrained checkpoints (~6 GB) are included in the Zenodo deposit, so most users can skip this step.

To rerun pretraining on a configured multi-GPU system:

```bash
bash scripts/pretrain_sv.sh self    cities100-1m
bash scripts/pretrain_sv.sh spatial cities100-1m  pretrain_ckpts/self_cities100-1m.pth.tar
bash scripts/pretrain_rs.sh self    cities100-1m
bash scripts/pretrain_rs.sh spatial cities100-1m  pretrain_ckpts/self_rs_cities100-1m.pth.tar
```

See `scripts/README.md` for the complete pipeline.

---

## License

Released under the Apache License 2.0 (see `LICENSE`). The `pretrain/` directory contains code derived from [facebookresearch/moco-v3](https://github.com/facebookresearch/moco-v3) (also Apache-2.0); see `pretrain/NOTICE.md` for modifications.

## Citation

```bibtex
@article{huang2026urban,
  title   = {Urban imagery reveals multidimensional sustainability of neighborhoods across global cities},
  author  = {Huang, Yingjing and Li, Yong and Zhang, Fan and others},
  journal = {(under review)},
  year    = {2026}
}
```

## Contact

Corresponding author: Fan Zhang — `fanzhanggis@pku.edu.cn`
