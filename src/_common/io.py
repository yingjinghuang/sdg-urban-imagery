"""Tolerant readers for the mix of pickle / HDF5 / CSV feature files this
pipeline emits.

The original codebase reads features from .pkl, .h5, and .csv in different
stages, with subtle column-handling differences. Centralizing here avoids
divergence (and the column-coercion-with-fallback boilerplate that was
copy-pasted into several scripts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def read_feature_table(path: str | Path, *, h5_key: str = "data") -> pd.DataFrame:
    """Load a feature table from .pkl, .h5, or .csv.

    For CSV files, attempt to coerce column labels to int (feature
    dimensions are stored as integer column names in this pipeline);
    fall back to leaving uncoerceable labels as-is.
    """
    path = str(path)
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    if path.endswith(".h5"):
        return pd.read_hdf(path, key=h5_key)
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        df.columns = [_maybe_int(c) for c in df.columns]
        return df
    raise ValueError(f"Unsupported feature format: {path}")


def _maybe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to .pkl, .h5, or .csv based on extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix
    if suffix == ".pkl":
        df.to_pickle(path)
    elif suffix == ".h5":
        df.to_hdf(path, key="data", mode="w")
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def feature_columns(df: pd.DataFrame, *, exclude: Iterable[str] = ("GEOID", "city", "country", "index", "path")) -> list:
    """Return the column labels that hold actual feature dimensions."""
    excluded = set(exclude)
    return [c for c in df.columns if c not in excluded]
