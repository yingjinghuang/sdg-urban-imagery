# Urban imagery reveals multidimensional sustainability of neighborhoods across global cities

Code and data-release repository for the paper.

> Huang Y., Li Y., Zhang F., et al. "Urban imagery reveals multidimensional sustainability of neighborhoods across global cities." *(Under review)*.

This repository provides the analysis pipeline used for the paper, from the deposited/processed neighborhood inputs through self/spatial-contrastive representation learning, SDG-indicator regression, sampling experiments, and figure generation. Raw-source-specific ingestion differs across countries and is documented separately; reviewers can reproduce the reported downstream analyses from the deposited processed tables and neighborhood representations without redistributing the original imagery.

---

## Repository layout

```
sdg-urban-imagery/
├── configs/                 Path and city manifests (edit paths.yaml for your machine)
├── pretrain/                Self/spatial-contrastive pretraining (derived from facebookresearch/moco-v3)
├── src/
│   ├── preprocess/          Portable label preparation/scaling helpers
│   ├── datasets/            Spatial-contrastive training-dataset construction
│   ├── extract/             Per-image feature extraction + neighborhood aggregation
│   ├── fuse/                Multi-modal feature fusion (SV ⊕ RS)
│   ├── analysis/            CLIP-concept probing, residual analysis
│   └── regression/          Token regression (Fold / Ratio / Sampling experiments)
├── scripts/                 Bash launchers + figure-input compatibility staging
├── figures/
│   ├── _data_prep/          Per-panel data preparation (.py / .ipynb)
│   ├── fig1/                Legacy internal folder for current manuscript Fig. 2 panels
│   ├── fig2/                Legacy internal folder for current manuscript Fig. 3 panels
│   ├── extended/            Extended Data Figures
│   └── supplementary/       Per-city Supplementary Figures
├── data/README.md           Data access (Zenodo DOI, reviewer access, raw-source pointers)
└── docs/
    ├── reproduce_main_figures.md
    └── data_schema.md
```

> **Figure-folder note.** The internal `figures/fig1` and `figures/fig2` directory names are retained for compatibility with the original analysis notebooks. In the current manuscript, these correspond to **Fig. 2 (model performance)** and **Fig. 3 (mechanisms / representation analyses)**, respectively. Manuscript Fig. 1 is the conceptual overview.

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
| Sustainability indicators (raw) | — | Per-country statistical agencies | See `docs/data_schema.md` and the manuscript Supplementary Information |

The Zenodo record is restricted during peer review. Editors and reviewers can be given a private Zenodo reviewer-access link by the corresponding author. The record will be made public upon acceptance.

The original satellite and street-view imagery is not redistributed because it was obtained through third-party APIs. The deposited neighborhood-level representations and processed tables are sufficient for reproducing the downstream regression, sampling, and main-figure analyses without redistributing the original imagery.

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

Copy `configs/paths.example.yaml` to `configs/paths.yaml` and point each entry at your local data/output roots. All maintained launchers read paths from this file.

### 3. Obtain the reviewer/public data bundle

During peer review, use the private Zenodo reviewer-access link supplied by the corresponding author. Once the record is public, download DOI `10.5281/zenodo.20483916` and unpack the deposited processed labels, neighborhood representations, split definitions, regression outputs, and checkpoints according to `data/README.md`.

### 4. Run the regression pipeline (current manuscript Figs. 2–3)

```bash
bash scripts/run_fold.sh             # 5-fold explanatory analysis; Fig. 2a / Fig. 3a
bash scripts/run_ratio.sh            # random-sampling scaling; Fig. 2b-d
bash scripts/select_kcenter.sh        # feature-guided split definitions (skip if deposited)
bash scripts/run_sampling.sh         # feature-guided sampling; Fig. 2b-d / Fig. 3f-g
bash scripts/run_fold_single_modal.sh
bash scripts/run_sampling_no_geo.sh  # non-spatial baseline for Fig. 3f-g
```

For the visual baselines in manuscript Fig. 2a:

```bash
bash scripts/run_fold_imagenet.sh
bash scripts/run_fold_segmentation.sh
```

### 5. Stage figure inputs and regenerate panels

The cleaned regression launchers use a canonical output hierarchy, while several original plotting notebooks retain their historical filenames/paths. Stage the small compatibility tables once after regression:

```bash
python scripts/prepare_figure_inputs.py --strict
```

Then follow the exact per-panel commands in [`docs/reproduce_main_figures.md`](docs/reproduce_main_figures.md). The guide uses executable `.ipynb`/`.py` filenames that exist in the repository and documents which panels use deposited prepared tables rather than raw third-party imagery.

---

## Optional: rerun pretraining from scratch

Pretraining requires the raw imagery, which is not redistributed because it is obtained through third-party APIs. Satellite imagery in the paper was obtained through the Google Static Maps API at approximately 0.6 m spatial resolution, while street-view imagery was obtained through the Google Street View Static API. Pretrained checkpoints (~6 GB) are included in the Zenodo deposit, so most reviewers/users can skip this step.

To rerun pretraining on a configured multi-GPU system, use the launchers under `scripts/`. See `scripts/README.md` for the complete pipeline and the manuscript/Supplementary Information for the reported pretraining configuration.

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
