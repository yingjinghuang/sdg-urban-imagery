"""Hierarchical k-center sample selection.

Implements the feature-guided sampling strategy used in manuscript Fig. 2d
and the downstream sampling analyses in Fig. 3. Neighborhoods are selected
to maximize coverage in a joint visual/geographic representation, with each
sampling ratio (10%, 20%, ..., 90%) nested within the next.

For the paper configuration, ``--feature_path`` is the unreduced 3072-d
``Concat_spatial_self.pkl`` representation containing four 768-d branches:
SV-spatial, RS-spatial, SV-self, and RS-self. Each branch is standardized and
reduced independently with PCA to retain 99% of its variance. The reduced
visual branches are concatenated, standardized centroid coordinates are
weighted by ``0.5 * sqrt(d_v)``, and classical k-center greedy selection is
then applied to the joint representation.

Outputs a pickle with columns ``GEOID``, ``geometry``, and one ``setNN``
column per ratio whose value is ``"train"`` (selected/surveyed) or ``"test"``
(unselected/estimated). This file is the ``--community_path`` for the
sampling-regression runs.

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
    """Farthest-point sampling with a deterministic random initial center."""
    n_total = features.shape[0]
    if n_total == 0:
        raise ValueError("Cannot run k-center on an empty feature matrix.")
    if not 1 <= n_samples <= n_total:
        raise ValueError(f"n_samples must be in [1, {n_total}], got {n_samples}.")

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
    return np.asarray(selected, dtype=int)


# --- Per-branch PCA fusion --------------------------------------------------


def block_pca(
    visual: np.ndarray,
    *,
    block_size: int = 768,
    variance: float = 0.99,
) -> np.ndarray:
    """Standardize and PCA-reduce each representation branch independently.

    ``visual`` must be a concatenation of equal-width branches. For the paper,
    its shape is ``(N, 3072)`` and ``block_size=768``, yielding four branches.
    PCA is fitted separately in each branch and retains ``variance`` of that
    branch's variance, matching the manuscript Methods description.
    """
    if visual.ndim != 2:
        raise ValueError(f"Expected a 2-D visual matrix, got shape {visual.shape}.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if visual.shape[1] % block_size != 0:
        raise ValueError(
            f"Visual width {visual.shape[1]} is not divisible by block_size={block_size}; "
            "cannot identify complete representation branches."
        )

    n_blocks = visual.shape[1] // block_size
    if n_blocks == 0:
        raise ValueError("No visual representation branches found.")

    print(f"[kcenter] {n_blocks} visual branches x {block_size} dimensions")
    pieces: list[np.ndarray] = []
    for i in range(n_blocks):
        start = i * block_size
        stop = (i + 1) * block_size
        block = visual[:, start:stop]
        block = StandardScaler().fit_transform(block)
        reduced = PCA(n_components=variance).fit_transform(block)
        print(f"[kcenter] branch {i}: {block.shape[1]} -> {reduced.shape[1]} dims")
        pieces.append(reduced)

    return np.concatenate(pieces, axis=1)


def fuse_with_geo(
    visual_pca: np.ndarray,
    coords_norm: np.ndarray,
    *,
    weight_factor: float = 0.5,
) -> np.ndarray:
    """Concatenate reduced visual features with re-weighted coordinates.

    The geographic component is multiplied by ``sqrt(d_v) * weight_factor``,
    where ``d_v`` is the dimensionality of the concatenated PCA-reduced visual
    representation.
    """
    visual_dim = visual_pca.shape[1]
    spatial_weight = np.sqrt(visual_dim) * weight_factor
    print(f"[kcenter] applying spatial weight {spatial_weight:.2f}")
    return np.concatenate([visual_pca, coords_norm * spatial_weight], axis=1)


# --- CLI driver -------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    output_parent = os.path.dirname(args.output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    print(f"[kcenter] loading {args.feature_path}")
    df = pd.read_pickle(args.feature_path)
    if "city" in df.columns:
        df = df.drop(columns=["city"])
    if "country" in df.columns:
        df = df.drop(columns=["country"])
    if "GEOID" not in df.columns:
        raise ValueError("Feature file must contain a 'GEOID' column.")

    feature_cols = [c for c in df.columns if c != "GEOID"]
    if not feature_cols:
        raise ValueError("Feature file contains no visual feature columns.")

    print(f"[kcenter] loading geography {args.geo_path}")
    geo = pd.read_pickle(args.geo_path)
    if "GEOID" not in geo.columns or "geometry" not in geo.columns:
        raise ValueError("Geo file must contain 'GEOID' and 'geometry' columns.")

    df = pd.merge(df, geo[["GEOID", "geometry"]], on="GEOID", how="inner")
    df = gpd.GeoDataFrame(df, geometry="geometry").reset_index(drop=True)
    if df.empty:
        raise ValueError("No common GEOIDs between visual features and geography.")

    visual = df[feature_cols].to_numpy()

    print("[kcenter] running branch-wise PCA")
    visual_pca = block_pca(
        visual,
        block_size=args.block_size,
        variance=args.variance,
    )
    print(f"[kcenter] concatenated PCA visual features -> {visual_pca.shape}")

    centroids = df["geometry"].centroid
    coords = np.column_stack([centroids.x, centroids.y])
    coords_norm = StandardScaler().fit_transform(coords)
    features = fuse_with_geo(
        visual_pca,
        coords_norm,
        weight_factor=args.weight_factor,
    )
    print(f"[kcenter] joint visual/geographic features -> {features.shape}")

    # Select once at the highest ratio; lower ratios are prefixes of this ordering.
    max_ratio = 0.9
    max_samples = int(len(df) * max_ratio)
    if max_samples < 1:
        raise ValueError("Too few neighborhoods to construct sampling ratios.")
    ordering = k_center_greedy(features, max_samples, seed=args.seed)

    out = df[["GEOID", "geometry"]].copy()
    for r in np.arange(0.1, 1.0, 0.1):
        r = round(float(r), 1)
        ratio_str = str(r).replace(".", "")
        n = int(round(len(df) * r))
        n = min(n, len(ordering))
        col = f"set{ratio_str}"
        out[col] = "test"
        if n > 0:
            out.iloc[ordering[:n], out.columns.get_loc(col)] = "train"

    out.to_pickle(args.output_path)
    print(f"[kcenter] saved {args.output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--feature_path",
        required=True,
        help=(
            "Unreduced concatenated representation. For paper reproduction, use "
            "Fuse/Concat_spatial_self.pkl (four 768-d branches)."
        ),
    )
    p.add_argument("--geo_path", required=True, help="Pickle with GEOID + geometry.")
    p.add_argument(
        "--output_path",
        required=True,
        help="Output pickle path (e.g. samples/pcahierachy.pkl).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--block_size",
        type=int,
        default=768,
        help="Width of each representation branch (768 in the paper).",
    )
    p.add_argument(
        "--variance",
        type=float,
        default=0.99,
        help="Variance retained independently within each branch PCA.",
    )
    p.add_argument(
        "--weight_factor",
        type=float,
        default=0.5,
        help="Multiplier in the geographic weight: weight_factor * sqrt(d_v).",
    )
    p.add_argument(
        "--strategy",
        default="hierarchical",
        choices=["hierarchical"],
        help="Reserved for future selection strategies.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
