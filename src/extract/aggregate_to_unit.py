"""Aggregate per-image features to per-neighborhood feature vectors.

The pipeline produces two kinds of per-image features:
    1. ViT-B embeddings (768-dim) from ``extract_feature.py``;
    2. ADE20K-class distributions (150-dim) from an off-the-shelf
       segmentation model (produced outside this repo, stored as CSV
       under ``${SEG_RESULTS_DIR}/{Country}/{City}.csv``).

For each neighborhood we take the mean feature across all images that
fall inside it. Optionally, the per-neighborhood std is concatenated to
double the feature dimension.

Replaces the original ``feature2unit.py``.

Usage:
    python -m src.extract.aggregate_to_unit \
        --feature_path ${FEATURES_RAW_DIR}/Australia/Adelaide/SV/Mocov3VITB-spatial-Australia-Adelaide-ep99.h5 \
        --meta_path ${PROCESSED_DIR}/Australia/Adelaide/paths.pkl \
        --save_path ${FEATURES_UNIT_DIR}/Australia/Adelaide/SV/Mocov3VITB-spatial-Australia-Adelaide-ep99.pkl \
        --arch VITB
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from src._common.io import read_feature_table
from src._common.paths import city_from_path, remap_image_paths

pd.options.mode.chained_assignment = None


FEATURE_DIMS = {
    "VITB": 768,
    "segmentation": 150,
    "ResNet50": 2048,
}


def _log(msg: str) -> None:
    print(f"[aggregate] {time.strftime('%H:%M:%S')} {msg}")


def aggregate(
    feature_path: str,
    meta_path: str,
    save_path: str,
    *,
    arch: str = "VITB",
    geoid_col: str = "GEOID",
    include_std: bool = False,
) -> None:
    """Group per-image features by neighborhood and write per-unit averages."""
    feature_size = FEATURE_DIMS.get(arch)
    if feature_size is None:
        raise ValueError(f"Unsupported arch {arch}; expected one of {list(FEATURE_DIMS)}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    _log(f"loading features from {feature_path}")
    features = read_feature_table(feature_path)

    # Some files use "path" as the path column; standardize to "index".
    if "index" not in features.columns and "path" in features.columns:
        features = features.rename(columns={"path": "index"})
    features["index"] = remap_image_paths(features["index"])

    _log(f"loading manifest from {meta_path}")
    meta = pd.read_pickle(meta_path)
    meta["city"] = city_from_path(meta["path"])

    _log("merging features with manifest")
    features = features.merge(
        meta[["path", geoid_col]],
        left_on="index",
        right_on="path",
        how="inner",
    )

    features = features.dropna(subset=[geoid_col]).copy()
    # Segmentation CSVs use 1-indexed class columns; ViT/ResNet HDF5 uses 0-indexed.
    if arch == "segmentation":
        feature_cols = list(range(1, feature_size + 1))
    else:
        feature_cols = list(range(feature_size))
    features = features[[geoid_col] + feature_cols]

    _log("computing per-neighborhood mean")
    mean_df = features.groupby(geoid_col).mean().reset_index()

    if include_std:
        _log("computing per-neighborhood std")
        std_df = features.groupby(geoid_col).std().reset_index(drop=True)
        std_df.columns = list(range(feature_size, 2 * feature_size))
        result = pd.concat([mean_df, std_df], axis=1)
    else:
        result = mean_df

    # Re-attach the city column (lost during the groupby).
    result = result.merge(
        meta[[geoid_col, "city"]].drop_duplicates(),
        on=geoid_col,
        how="inner",
    )

    # Normalize segmentation columns to 0..149 for downstream consistency.
    if arch == "segmentation":
        result.columns = [geoid_col] + list(range(feature_size)) + ["city"]

    _log(f"writing {len(result)} rows -> {save_path}")
    result.to_pickle(save_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--feature_path", required=True, help=".h5 / .pkl / .csv of per-image features")
    p.add_argument("--meta_path", required=True, help="paths.pkl manifest mapping image -> GEOID")
    p.add_argument("--save_path", required=True, help="output .pkl per-neighborhood feature table")
    p.add_argument("--arch", default="VITB", choices=sorted(FEATURE_DIMS.keys()))
    p.add_argument("--geoid_col", default="GEOID",
                   help="name of the neighborhood-id column in the manifest")
    p.add_argument("--std", action="store_true",
                   help="concatenate per-neighborhood std to the mean (doubles feature dim)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    aggregate(
        args.feature_path,
        args.meta_path,
        args.save_path,
        arch=args.arch,
        geoid_col=args.geoid_col,
        include_std=args.std,
    )


if __name__ == "__main__":
    main()
