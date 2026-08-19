# Pretrain — derivative of facebookresearch/moco-v3

This directory contains contrastive-pretraining code derived from
[facebookresearch/moco-v3](https://github.com/facebookresearch/moco-v3),
originally licensed under Apache-2.0. This release repository inherits
that license — see the top-level `LICENSE` file.

## Provenance

| File | Origin |
|---|---|
| `moco/builder.py`, `moco/loader.py`, `moco/optimizer.py`, `moco/__init__.py` | Unmodified copy of the upstream `moco/` package. |
| `vits.py` | Unmodified copy of the upstream `vits.py`. |
| `datasets.py` | New — custom `PairsDataset` for spatial-contrastive sample selection (loads `(path1, path2, GEOID)` pairs from a pickle produced by `src/datasets/spatial_contrastive.py`). |
| `moco_sv.py` | Modified — multi-GPU street-view variant. Renamed from `moco_gsv_multi.py`. Differs from upstream `main_moco.py` in: dataset loader (PairsDataset instead of ImageFolder), optional Neptune logging hooks, custom CLI for `--save-folder` and resume semantics. |
| `moco_rs.py` | Modified — same as `moco_sv.py` but loads the high-resolution satellite imagery used in the paper (~0.6 m imagery obtained through the Google Static Maps API). Renamed from `moco_rs_multi.py`. |

## Files NOT included from upstream

The following upstream files were not relevant to this paper and were
excluded to keep the release minimal:

- `main_lincls.py` — supervised linear probing (we use a custom token
  regression head; see `src/regression/token_reg.py`).
- `main_moco.py` — single-GPU variant, superseded by `moco_sv.py` /
  `moco_rs.py`.
- `transfer/` — Oxford Flowers / Oxford Pets transfer benchmarks.
- `convert_to_deit.py` — DeiT conversion utility.

## Files dropped from the legacy SDG repo

- `moco-v3/moco_rs.py` — single-GPU duplicate of `moco_rs_multi.py`,
  superseded by the multi-GPU version which we kept as `moco_rs.py`.
- `moco-v3/CONFIG.md` — informal training notes, superseded by this
  NOTICE and by `scripts/README.md`.

## Reproducing the published checkpoints

The pretrained checkpoints used for this paper's results are distributed
via Zenodo (see `data/README.md`). Most downstream users do not need to
rerun pretraining — they can download the checkpoints and proceed
directly to feature extraction.

If you do want to rerun, see `../scripts/pretrain_sv.sh` and
`../scripts/pretrain_rs.sh`. Compute budget per modality with the paper's
training set: ~5 days on 8× A100.

## Citation

If you use this pretraining code, please cite both the original MoCo-v3
paper and this work:

```bibtex
@inproceedings{chen2021mocov3,
  title     = {An Empirical Study of Training Self-Supervised Vision Transformers},
  author    = {Chen, Xinlei and Xie, Saining and He, Kaiming},
  booktitle = {ICCV},
  year      = {2021}
}
```
