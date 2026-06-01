"""Hierarchical k-center sample selection.

Implements the feature-guided sampling strategy used in Fig 1d and
Fig 3: greedily select neighborhoods that maximize feature-space
coverage, so each sampling ratio (10%, 20%, ..., 90%) contains the
previous one as a subset.

The "hierarchical" framing combines per-modality PCA (one per 768-d
block) with a spatially-weighted geographic component, then runs
classical k-center greedy on the concatenated representation.

Outputs a pickle with columns ``GEOID``, ``geometry``, and one
``setNN`` column per ratio whose value is ``"train"`` (selected) or
``"test"`` (unselected). This file is the ``--community_path`` for
the regression runs.

Replaces ``6regression_sampling/kcenter.py``.
"""

from __future__ import annotations

import argparse
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


# --- Core selection ---------------------------------------------------------


def k_center_greedy(features: np.ndarray, n_samples: int, seed: int = 42) -> np.ndarray:
    """Farthest-point sampling: pick ``n_samples`` indices that minimize
    the maximum distance from any unselected point to its nearest
    selected point."""
    n_total = features.shape[0]
    rng = np.random.RandomState(seed)
    first = int(rng.choice(n_total))
    selected = [first]
    min_dists = pairwise_distances(features, features[first].reshape(1, -1)).flatten()
    for _ in range(1, n_samples):
        new_idx = int(np.argmax(min_dists))
        selected.append(new_idx)
        new_dist = pairwise_distances(features, features[new_idx].reshape(1, -1)).flatten()
        min_dists = np.minimum(min_dists, new_dist)
        if len(selected) % 200 == 0:
            print(f"[kcenter] selected {len(selected)} / {n_samples}")
    return np.array(selected)


# --- Per-block PCA fusion ---------------------------------------------------


def block_pca(visual: np.ndarray, *, block_size: int = 768, variance: float = 0.99) -> np.ndarray:
    """Run a per-block PCA on a concatenated multi-modal feature matrix.

    Each ``block_size``-wide column slice is treated as one modality
    and reduced independently.
    """
    n_blocks = max(1, visual.shape[1] // block_size)
    if n_blocks <= 1:
        return PCA(n_components=variance).fit_transform(visual)
    pieces = []
    for i in range(n_blocks):
        block = visual[:, i * block_size: (i + 1) * block_size]
        pieces.append(PCA(n_components=variance).fit_transform(block))
    return np.concatenate(pieces, axis=1)


def fuse_with_geo(visual_pca: np.ndarray, coords_norm: np.ndarray, *, weight_factor: float = 0.5) -> np.ndarray:
    """Concatenate PCA-reduced visual features with re-weighted coordinates.

    The geographic component is multiplied by ``sqrt(visual_dim) * weight_factor``
    so its influence scales with the visual feature width (otherwise 2-D
    coordinates would be swamped by ~100-D visual features).
    """
    visual_dim = visual_pca.shape[1]
    spatial_weight = np.sqrt(visual_dim) * weight_factor
    print(f"[kcenter] applying spatial weight {spatial_weight:.2f}")
    return np.concatenate([visual_pca, coords_norm * spatial_weight], axis=1)


# --- CLI driver -------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f"[kcenter] loading {args.feature_path}")
    df = pd.read_pickle(args.feature_path)
    if "city" in df.columns:
        df = df.drop(columns=["city"])
    feature_len = df.shape[1] - 1  # GEOID + features

    geo = pd.read_pickle(args.geo_path)
    df = pd.merge(df, geo[["GEOID", "geometry"]], on="GEOID", how="inner")
    df = gpd.GeoDataFrame(df, geometry="geometry").reset_index(drop=True)

    visual = df.loc[:, list(range(feature_len))].values
    visual = StandardScaler().fit_transform(visual)

    print("[kcenter] running per-block PCA")
    visual_pca = block_pca(visual, block_size=args.block_size, variance=args.variance)
    print(f"[kcenter] PCA -> {visual_pca.shape}")

    centroids = df["geometry"].centroid
    coords = np.column_stack([centroids.x, centroids.y])
    coords_norm = StandardScaler().fit_transform(coords)
    features = fuse_with_geo(visual_pca, coords_norm, weight_factor=args.weight_factor)
    print(f"[kcenter] fused features {features.shape}")

    # Select once at the highest ratio; lower ratios are prefixes of this ordering.
    max_ratio = 0.9
    max_samples = int(len(df) * max_ratio)
    ordering = k_center_greedy(features, max_samples, seed=args.seed)

    out = df[["GEOID", "geometry"]].copy()
    for r in np.arange(0.1, 1.0, 0.1):
        r = round(r, 1)
        ratio_str = str(r).replace(".", "")
        n = int(round(len(df) * r))
        col = f"set{ratio_str}"
        out[col] = "test"
        out.iloc[ordering[:n], out.columns.get_loc(col)] = "train"

    out.to_pickle(args.output_path)
    print(f"[kcenter] saved {args.output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--feature_path", required=True,
                   help="Concat_*.pkl (typically Concat_spatial_self_pca99.pkl).")
    p.add_argument("--geo_path", required=True,
                   help="Pickle with GEOID + geometry.")
    p.add_argument("--output_path", required=True,
                   help="Output pickle path (e.g. samples/pcahierachy.pkl).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--block_size", type=int, default=768,
                   help="Per-modality block width for the block-PCA stage.")
    p.add_argument("--variance", type=float, default=0.99,
                   help="Variance retained by each per-block PCA.")
    p.add_argument("--weight_factor", type=float, default=0.5,
                   help="Multiplier for the geographic-feature weight.")
    p.add_argument("--strategy", default="hierarchical",
                   choices=["hierarchical"],
                   help="Reserved for future selection strategies.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
