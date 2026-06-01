# Urban imagery reveals multidimensional sustainability of neighborhoods across global cities

Code and data-release repository for the paper.

> Huang Y., Li Y., Zhang F., et al. "Urban imagery reveals multidimensional sustainability of neighborhoods across global cities." *(Under review)*.

This repository provides the full pipeline to reproduce the main and supplementary figures of the paper, from raw imagery preprocessing through self/spatial-contrastive representation learning to SDG indicator regression and figure generation.

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
│   ├── analysis/            CLIP-concept interpretability, residual analysis
│   └── regression/          Token regression (Fold / Ratio / Sampling experiments)
├── scripts/                 Bash launchers (read paths from configs/paths.yaml)
├── figures/
│   ├── _data_prep/          Per-panel data preparation (.py)
│   ├── fig1/                Main Fig 1 panels (a, b, c, d)
│   ├── fig2/                Main Fig 2 panels (a, b, c, d, e, f, g)
│   ├── extended/            Extended Data Figures
│   └── supplementary/       Per-city Supplementary Figures
├── data/README.md           Data access (Zenodo DOIs, raw-source pointers)
└── docs/
    ├── reproduce_main_figures.md
    └── data_schema.md
```

---

## Data and model availability

| Asset | Size | Location | Access |
|---|---|---|---|
| Neighborhood-level features (`Unit/`) | ~1.7 GB | Zenodo (DOI: TBD) | Public |
| Per-image raw features (`Raw/`) | ~142 GB | Zenodo large-data deposit (DOI: TBD) | Public (large) |
| Trained model weights — main framework | ~25 GB | Zenodo (DOI: TBD) | Public |
| Trained model weights — ablation/baselines | varies | — | Available on request |
| Raw satellite imagery | — | Sentinel-2 (Copernicus, free) | Via Google Earth Engine |
| Raw street view imagery | — | Google Street View Static API | Via API key (paid) |
| Sustainability indicators | — | Per-country sources | See `docs/data_schema.md` |

DOIs and links will be populated upon acceptance.

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
# (commands populated after Zenodo DOIs assigned)
```

### 4. Run the regression pipeline (Fig 1, Fig 2)

```bash
bash scripts/run_fold.sh         # 5-fold cross-validation R² (Fig 1a, Fig 2a)
bash scripts/run_sampling.sh     # Sampling experiments (Fig 1b, c, d)
bash scripts/run_ratio.sh        # Ratio experiments (Fig 2f, g)
```

### 5. Regenerate figures

```bash
jupyter notebook figures/fig1/
jupyter notebook figures/fig2/
```

Each panel notebook reads its inputs from `regression_outputs/` and writes a panel PDF to `figures/_out/`. See `docs/reproduce_main_figures.md` for the full per-figure mapping.

---

## Optional: rerun pretraining from scratch

Pretraining requires the raw imagery (not included in the public release; obtain via API). On a multi-GPU node:

```bash
bash scripts/run_pretrain.sh
```

Pretrained checkpoints (~6 GB) are included in the Zenodo deposit, so most users can skip this step.

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

Corresponding authors:
- Fan Zhang — `fanzhanggis@pku.edu.cn`
- Yu Liu — `yuliugis@pku.edu.cn`
