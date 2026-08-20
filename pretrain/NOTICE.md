# Pretrain — derivative of facebookresearch/moco-v3

This directory contains contrastive-pretraining code derived from
[facebookresearch/moco-v3](https://github.com/facebookresearch/moco-v3),
originally licensed under Apache-2.0. This release repository inherits that
license — see the top-level `LICENSE` file.

## Provenance

| File | Origin |
|---|---|
| `moco/builder.py`, `moco/loader.py`, `moco/optimizer.py`, `moco/__init__.py` | Unmodified copy of the upstream `moco/` package. |
| `vits.py` | Unmodified copy of the upstream `vits.py`. |
| `moco_sv.py` | Modified multi-GPU street-view entry point with the study-specific dataset loaders, checkpoint handling, and paper-reproduction learning-rate semantics. |
| `moco_rs.py` | Satellite counterpart using the channel normalization and image loaders used by the study. |

The study-specific self/spatial dataset loaders used during pretraining are
implemented directly in `moco_sv.py` and `moco_rs.py`. Spatial-pair dataset
construction is maintained separately under `src/datasets/`.

The cleaned release entry points intentionally contain no external
experiment-tracking service integration or embedded service credentials.
Training metrics are written locally through TensorBoard.

## Pretraining configuration

The paper-reproduction launchers are `../scripts/pretrain_sv.sh` and
`../scripts/pretrain_rs.sh`. They use ViT-B, AdamW, total batch size 1024,
initial optimizer learning rate `1e-5`, weight decay `1e-6`, 100 street-view
epochs and 50 satellite epochs. A 10-epoch warm-up is retained from the
implementation; the manuscript does not separately report that value.

Unlike the stock MoCo-v3 launcher, the cleaned entry points interpret `--lr` as
the actual initial optimizer learning rate and do **not** multiply it by
`batch_size / 256`. This keeps a launcher value of `1e-5` equal to the
manuscript-reported optimizer learning rate.

Two checkpoint modes are supported:

- `--pretrained`: initialize model weights but start a fresh optimizer/epoch
  counter. The shell launchers expose this as the optional `init_ckpt`, e.g.
  when starting a spatial-contrastive run from a self-contrastive checkpoint.
- `--resume`: restore model, optimizer, AMP scaler, and epoch for an interrupted
  run. The shell launchers expose this separately as `resume_ckpt`.

## Reproducing the published checkpoints

The pretrained checkpoints used for the paper are distributed via Zenodo (see
`data/README.md`). Most downstream users do not need to rerun pretraining: they
can use those checkpoints and proceed directly to feature extraction.

Full pretraining requires independent access to the third-party imagery and
substantial GPU compute. The downstream main-figure reproduction route therefore
starts from the deposited checkpoints/derived representations.

## Files not included from upstream

Upstream training/transfer utilities that are not used by this study were
excluded to keep the release focused; the original MoCo-v3 repository remains
the authoritative source for those utilities.

## Citation

If you use this pretraining code, please cite both the original MoCo-v3 paper
and this work:

```bibtex
@inproceedings{chen2021mocov3,
  title     = {An Empirical Study of Training Self-Supervised Vision Transformers},
  author    = {Chen, Xinlei and Xie, Saining and He, Kaiming},
  booktitle = {ICCV},
  year      = {2021}
}
```
