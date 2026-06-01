"""PCA-99% reduction of a Concat_*.pkl feature file.

Used to compress ``Concat_spatial_self.pkl`` (3072-d) into a smaller
feature space that retains 99% of the variance. The PCA-reduced
feature feeds the Fig 3 sampling experiments via the
``Multi_Concat_pcahierachy`` regression run.

Output schema mirrors the input: GEOID, contiguous integer feature
columns, and (when present) city.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src._common.io import feature_columns


def pca_reduce(input_path: str, output_path: str, variance: float = 0.99) -> None:
    df = pd.read_pickle(input_path)
    feat_cols = feature_columns(df)
    X = df[feat_cols].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    pca = PCA(n_components=variance, svd_solver="full")
    Z = pca.fit_transform(X)
    n_comp = Z.shape[1]
    print(f"[pca] {input_path} {X.shape[1]}d -> {n_comp}d (variance={variance})")

    out = pd.DataFrame(Z, columns=list(range(n_comp)))
    out.insert(0, "GEOID", df["GEOID"].values)
    if "city" in df.columns:
        out["city"] = df["city"].values

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(output_path)

    # Persist the fitted scaler + PCA alongside the output so the same
    # transform can be reapplied if needed.
    joblib.dump({"scaler": scaler, "pca": pca}, output_path.replace(".pkl", "_transform.joblib"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", required=True, help="Concat feature pkl to reduce.")
    p.add_argument("--output", required=True, help="Reduced feature pkl path.")
    p.add_argument("--variance", type=float, default=0.99,
                   help="Fraction of variance to retain.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pca_reduce(args.input, args.output, variance=args.variance)


if __name__ == "__main__":
    main()
