"""Training-dataset construction for the contrastive pretraining stage.

This stage emits pickled DataFrames of image pairs that the Mocov3
trainers consume directly:

    - self-contrastive: a single image gets two augmentations.
    - spatial-contrastive: two images sampled from the same neighborhood.

The self-contrastive dataset is just the list of image paths used by
``pretrain/moco_*.py`` with built-in augmentation. The spatial-contrastive
dataset requires explicit (path1, path2, GEOID) tuples and is built by
:mod:`spatial_contrastive_sv` and :mod:`spatial_contrastive_rs`.
"""
