# Data access

The reproducibility data and pretrained model checkpoints for this paper are deposited on Zenodo:

- DOI: https://doi.org/10.5281/zenodo.20483916

During peer review, the Zenodo record is restricted. Editors and reviewers can access the deposited files through a private reviewer-access link provided directly by the corresponding author. The reviewer-only link is intentionally not published in this repository.

The Zenodo deposit includes the processed neighborhood-level data needed to reproduce the main analyses and figures, together with the pretrained MoCo-v3 ViT-B checkpoints used in the paper. The record will be made public upon acceptance.

## Raw imagery

Raw third-party imagery is not redistributed in the Zenodo deposit.

- **Satellite imagery:** high-resolution satellite imagery obtained through the Google Static Maps API at approximately 0.6 m spatial resolution, as described in the manuscript.
- **Street-view imagery:** Google Street View imagery obtained through the Google Street View Static API.

Users who wish to rerun imagery acquisition, pretraining, or feature extraction from scratch must obtain the corresponding imagery through the relevant Google Maps Platform APIs under their own access credentials and applicable terms of service.

## Derived data

See `../docs/data_schema.md` for the schema of the derived artifacts and `../docs/reproduce_main_figures.md` for the commands used to regenerate the reported figures.
