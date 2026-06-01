"""Label / feature / community / geo alignment for regression training."""

from __future__ import annotations

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _align(df: pd.DataFrame, geoids: list) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["GEOID"])
    return df.set_index("GEOID").reindex(geoids).reset_index()


def _coerce_geoid(df: pd.DataFrame, as_int: bool) -> pd.DataFrame:
    df["GEOID"] = df["GEOID"].astype(int).astype(str) if as_int else df["GEOID"].astype(str)
    return df


def load_aligned(
    label_path: str,
    feature_path: str,
    community_path: str,
    geo_path: str | None,
    output_dir: str,
    feature_len: int,
    *,
    use_geo: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """Load all four input tables and align them on the GEOID intersection.

    Returns:
        df_labels: aligned label table with `set*` split columns merged in.
        X_visual: (N, n_modalities, feature_len) standardized visual features.
        X_coords: (N, 2) standardized lat/lon, or None if ``use_geo=False``.

    Also persists the fitted StandardScalers (per-modality and geo) under
    ``output_dir`` so the same transform can be reapplied at inference.
    """
    df_labels = pd.read_pickle(label_path)
    df_features = pd.read_pickle(feature_path)
    df_community = pd.read_pickle(community_path)

    # GEOID typing: HongKong and Lagos use string IDs natively; everything else
    # is numeric and we coerce to canonical int->string to align reliably.
    as_int = ("HongKong" not in community_path) and ("Lagos" not in community_path)
    df_labels = _coerce_geoid(df_labels, as_int)
    df_features = _coerce_geoid(df_features, as_int)
    df_community = _coerce_geoid(df_community, as_int)

    # Keep only the split columns in the community table.
    set_cols = [c for c in df_community.columns if "set" in str(c)]
    df_community = df_community[["GEOID"] + set_cols]

    common = set(df_labels["GEOID"]) & set(df_features["GEOID"]) & set(df_community["GEOID"])

    if use_geo:
        if geo_path is None:
            raise ValueError("geo_path is required when use_geo=True")
        df_geo = pd.read_pickle(geo_path)
        df_geo = _coerce_geoid(df_geo, as_int)
        if "geometry" not in df_geo.columns:
            raise ValueError("Geo file must contain a 'geometry' column.")
        common &= set(df_geo["GEOID"])

    common = sorted(common)
    print(f"[regression] {len(common)} common GEOIDs")

    df_labels = _align(df_labels, common)
    df_features = _align(df_features, common)
    df_community = _align(df_community, common)
    df_labels = df_labels.merge(df_community, on="GEOID", how="left")

    # --- Coordinates ---
    if use_geo:
        df_geo = _align(df_geo, common)
        centroids = gpd.GeoSeries(df_geo["geometry"]).centroid
        coords = np.column_stack((centroids.x, centroids.y))
        scaler_geo = StandardScaler()
        X_coords = scaler_geo.fit_transform(coords)
        joblib.dump(scaler_geo, f"{output_dir}/scaler_geo.pkl")
    else:
        X_coords = None

    # --- Visual features ---
    feat_cols = [c for c in df_features.columns if c not in ("GEOID", "city", "country")]
    X_all = df_features[feat_cols].values
    if X_all.shape[1] % feature_len != 0:
        raise ValueError(
            f"Visual feature width {X_all.shape[1]} is not a multiple of feature_len={feature_len}."
        )
    n_modalities = X_all.shape[1] // feature_len

    blocks = []
    for i in range(n_modalities):
        block = X_all[:, i * feature_len: (i + 1) * feature_len]
        scaler = StandardScaler()
        block = scaler.fit_transform(block)
        joblib.dump(scaler, f"{output_dir}/scaler_modality_{i}.pkl")
        blocks.append(block)
    X_visual = np.stack(blocks, axis=1)  # (N, n_modalities, feature_len)

    if use_geo:
        assert len(df_labels) == len(X_visual) == len(X_coords)
    else:
        assert len(df_labels) == len(X_visual)
    print(f"[regression] visual {X_visual.shape}, coords {None if X_coords is None else X_coords.shape}")

    return df_labels, X_visual, X_coords
