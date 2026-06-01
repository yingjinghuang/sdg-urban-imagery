"""Per-target outlier removal + standardization of country indicator tables.

For each indicator listed for a country:
    1. Drop values whose |z-score| > 3 (set to NaN).
    2. Standardize the remaining values with sklearn ``StandardScaler``.
    3. Drop the column entirely if fewer than 50 non-NaN values survive.

Writes:
    {labels_dir}/{Country}/{City}/labels_norm.pkl   normalized labels
    {labels_dir}/{Country}/{City}/scalers.joblib    fitted per-target scalers

Replaces ``1preprocess/label_scale_all.py``. Country/city iteration moves
to ``scripts/scale_labels.sh``; per-country indicator lists move to
``configs/cities.yaml`` (``targets`` block) so this script reads them at
runtime instead of hard-coding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

MIN_NON_NAN = 50


def mask_outliers(series: pd.Series, *, z_threshold: float = 3.0) -> pd.Series:
    """Return ``series`` with |z| > threshold replaced by NaN."""
    non_nan = series.dropna()
    if len(non_nan) < 2:
        return series
    z = stats.zscore(non_nan)
    bad = non_nan.index[np.abs(z) > z_threshold]
    series = series.copy()
    series.loc[bad] = np.nan
    return series


def scale_one_city(
    labels_pkl: Path,
    targets: list[str],
    *,
    out_pkl: Path,
    out_scalers: Path,
) -> None:
    df = pd.read_pickle(labels_pkl)
    scalers: dict[str, StandardScaler] = {}
    for target in targets:
        if target not in df.columns:
            print(f"  [skip] {target} not in {labels_pkl.name}")
            continue
        df[target] = mask_outliers(df[target])
        scaler = StandardScaler()
        df[target] = scaler.fit_transform(df[[target]])
        scalers[target] = scaler
        if df[target].dropna().shape[0] < MIN_NON_NAN:
            df = df.drop(columns=[target])
            scalers.pop(target, None)
            print(f"  [drop] {target} (<{MIN_NON_NAN} non-NaN values)")

    # Drop columns that are entirely NaN (e.g. all-NaN auxiliary fields).
    df = df.dropna(axis=1, how="all")

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_pkl)
    joblib.dump(scalers, out_scalers)
    print(f"  saved {out_pkl}")


def load_targets(cities_yaml: Path, country: str) -> list[str]:
    import yaml
    with open(cities_yaml) as f:
        cfg = yaml.safe_load(f)
    return list(cfg.get("targets", {}).get(country, []))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--country", required=True)
    p.add_argument("--city", required=True)
    p.add_argument("--labels_dir", required=True,
                   help="Directory containing per-country/city labels.pkl.")
    p.add_argument("--cities_yaml", required=True,
                   help="configs/cities.yaml (provides per-country target list).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    targets = load_targets(Path(args.cities_yaml), args.country)
    if not targets:
        raise ValueError(f"No targets defined for country={args.country} in {args.cities_yaml}")

    base = Path(args.labels_dir) / args.country / args.city
    scale_one_city(
        base / "labels.pkl",
        targets,
        out_pkl=base / "labels_norm.pkl",
        out_scalers=base / "scalers.joblib",
    )


if __name__ == "__main__":
    main()
