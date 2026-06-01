"""Shared logic for the fuse-stage concatenation scripts.

All Concat_*.pkl outputs are formed by:
    1. Loading 2 or more per-neighborhood feature DataFrames.
    2. Taking the GEOID intersection.
    3. Horizontally stacking the feature columns, renumbering them
       contiguously starting at 0 so downstream regression code can treat
       the result as a single feature block of known size.

This helper centralizes that pattern.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src._common.io import feature_columns


def hstack_on_geoid(
    pkls: list[str],
    save_path: str,
    *,
    expected_dim_per_block: int | None = None,
    preserve_city: bool = True,
) -> pd.DataFrame:
    """Load and horizontally-stack feature pkls on the GEOID intersection.

    Output columns are: ``GEOID``, integer-indexed feature columns
    ``0..D-1`` where ``D`` is the sum of the input feature dimensions,
    and optionally ``city``.
    """
    if len(pkls) < 2:
        raise ValueError("Need at least two feature files to concatenate.")
    for p in pkls:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    dfs = [pd.read_pickle(p) for p in pkls]
    for i, df in enumerate(dfs):
        if "GEOID" not in df.columns:
            raise ValueError(f"GEOID column missing in {pkls[i]}")

    # Intersect GEOIDs across all inputs.
    common = set(dfs[0]["GEOID"])
    for df in dfs[1:]:
        common &= set(df["GEOID"])
    if not common:
        raise ValueError(f"No common GEOIDs across inputs: {pkls}")
    common_sorted = sorted(common)

    # Pull out feature blocks in input order.
    blocks: list[np.ndarray] = []
    city_series: pd.Series | None = None
    for df in dfs:
        sub = df[df["GEOID"].isin(common)].sort_values("GEOID").reset_index(drop=True)
        feat_cols = feature_columns(sub)
        block = sub[feat_cols].values
        if expected_dim_per_block is not None and block.shape[1] != expected_dim_per_block:
            raise ValueError(
                f"Expected {expected_dim_per_block} feature columns per block, "
                f"got {block.shape[1]} in one of {pkls}"
            )
        blocks.append(block)
        if preserve_city and city_series is None and "city" in sub.columns:
            city_series = sub["city"].reset_index(drop=True)

    X = np.hstack(blocks)
    out = pd.DataFrame(X, columns=list(range(X.shape[1])))
    out.insert(0, "GEOID", common_sorted)
    if city_series is not None and len(city_series) == len(out):
        out["city"] = city_series.values

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(save_path)
    print(f"[fuse] {len(out)} rows x {X.shape[1]} feat -> {save_path}")
    return out
