"""Prepare the spatial-vs-non-spatial and Moran's-I table for Fig. 3f-g.

Reads the canonical feature-guided sampling outputs written by
``scripts/run_sampling.sh`` and ``scripts/run_sampling_no_geo.sh`` and combines
them with per-neighborhood indicator geometries. The output is the table read
by ``figures/fig2/panel_f_spatial_curve.ipynb`` (that notebook contains both
the spatial/non-spatial scaling panel and the Moran's-I correlation panel).

Usage:
    python figures/_data_prep/prep_fig2g_moransi.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from esda.moran import Moran
from libpysal.weights import KNN


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_paths(config_path: Path) -> dict[str, str]:
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    pattern = re.compile(r"\$\{([^}]+)\}")
    resolved: dict[str, str] = {}
    for key, value in cfg.items():
        if not isinstance(value, str):
            resolved[key] = value
            continue
        while True:
            m = pattern.search(value)
            if m is None:
                break
            ref = m.group(1)
            if ref not in resolved:
                raise KeyError(f"Unresolved variable ${{{ref}}} in {config_path}")
            value = value.replace(m.group(0), str(resolved[ref]))
        resolved[key] = value
    return resolved


def load_result_family(root: Path, run_tag: str, value_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    pattern = f"sampling/{run_tag}/*/*/Fuse/Multi_Concat_pcahierachy/results.csv"
    for path in sorted(root.glob(pattern)):
        rel = path.relative_to(root).parts
        country, city = rel[2], rel[3]
        df = pd.read_csv(path)
        required = {"target", "ratio", "all_r2"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")
        tmp = df[["target", "ratio", "all_r2"]].copy()
        tmp["country"] = country
        tmp["city"] = city
        tmp = tmp.rename(columns={"all_r2": value_name})
        frames.append(tmp)
    if not frames:
        raise FileNotFoundError(
            f"No sampling outputs found for run_tag={run_tag!r} under {root}. "
            "Run the corresponding sampling launcher first."
        )
    return pd.concat(frames, ignore_index=True)


def load_sdg_map(processed_dir: Path, country: str) -> pd.DataFrame | None:
    path = processed_dir / "0labels" / f"{country}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if not {"ID", "SDG"}.issubset(df.columns):
        return None
    return df[["ID", "SDG"]].drop_duplicates()


def find_labels_file(paths: dict[str, str], country: str, city: str) -> Path:
    candidates = [
        Path(paths["labels_dir"]) / country / city / "labels.pkl",
        Path(paths["processed_dir"]) / country / city / "labels.pkl",
        REPO_ROOT / "data" / "processed" / country / city / "labels.pkl",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No labels.pkl found for {country}/{city}; checked: "
        + ", ".join(str(p) for p in candidates)
    )


def moran_for_region(
    labels_path: Path,
    targets: list[str],
    country: str,
    city: str,
    *,
    k_neighbors: int = 6,
) -> pd.DataFrame:
    df = pd.read_pickle(labels_path)
    if "geometry" not in df.columns:
        raise ValueError(f"{labels_path}: missing geometry column")
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    gdf = gdf[gdf.geometry.notna()].copy()

    rows: list[dict] = []
    for target in targets:
        if target not in gdf.columns:
            continue
        current = gdf.dropna(subset=[target]).copy()
        if len(current) < max(5, k_neighbors + 1):
            continue
        try:
            w = KNN.from_dataframe(current, k=k_neighbors)
            w.transform = "r"
            moran = Moran(current[target].to_numpy(), w)
        except Exception as exc:
            print(f"[moran] skip {country}/{city}/{target}: {exc}")
            continue
        rows.append(
            {
                "target": target,
                "Moran_I": moran.I,
                "Expected_I": moran.EI,
                "p_value": moran.p_sim,
                "country": country,
                "city": city,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "paths.yaml",
        help="Path configuration used by the regression launchers.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "fig" / "fig2_geo_reg_moransi_results.csv",
    )
    p.add_argument("--k-neighbors", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(args.config)
    regression_root = Path(paths["regression_out_dir"])
    processed_dir = Path(paths["processed_dir"])

    ours = load_result_family(regression_root, "main", "Ours")
    no_geo = load_result_family(regression_root, "main_no_geo", "Non-spatial")
    df = ours.merge(
        no_geo,
        on=["target", "ratio", "country", "city"],
        how="inner",
        validate="one_to_one",
    )
    if df.empty:
        raise RuntimeError("Spatial and non-spatial sampling outputs have no overlapping rows.")

    # Add SDG labels where the country dictionaries are available.
    mapped: list[pd.DataFrame] = []
    for country, part in df.groupby("country", sort=False):
        sdg_map = load_sdg_map(processed_dir, country)
        if sdg_map is not None:
            part = part.merge(sdg_map, left_on="target", right_on="ID", how="left")
            part = part.drop(columns=["ID"])
        else:
            part = part.copy()
            part["SDG"] = pd.NA
        mapped.append(part)
    df = pd.concat(mapped, ignore_index=True)

    # Moran's I is a property of each region/indicator and therefore only needs
    # to be computed once, then joined to all sampling ratios.
    moran_parts: list[pd.DataFrame] = []
    for (country, city), part in df.groupby(["country", "city"], sort=False):
        labels_path = find_labels_file(paths, country, city)
        targets = sorted(part["target"].dropna().unique().tolist())
        moran_parts.append(
            moran_for_region(
                labels_path,
                targets,
                country,
                city,
                k_neighbors=args.k_neighbors,
            )
        )
    moran_df = pd.concat(moran_parts, ignore_index=True)
    df = df.merge(moran_df, on=["target", "country", "city"], how="left")

    df["ratio"] = df["ratio"] * 100.0
    df["delta"] = df["Ours"] - df["Non-spatial"]
    columns = [
        "ratio",
        "target",
        "SDG",
        "Ours",
        "Non-spatial",
        "country",
        "city",
        "delta",
        "Moran_I",
        "Expected_I",
        "p_value",
    ]
    df = df[[c for c in columns if c in df.columns]]
    df = df.sort_values(["country", "city", "target", "ratio"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[moran] saved {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
