"""Regenerate the maintained per-region Supplementary figures.

This script reads the canonical regression outputs written by the current
launchers rather than the historical ``regmodels`` directory layout.

For each region in ``configs/cities.yaml`` it can generate:

- ``S_<country>_<city>_fold.pdf``
    Five-fold held-out R² by indicator.
- ``S_<country>_<city>_random.pdf``
    Citywide reconstruction R² across random partial-survey ratios.
- ``S_<country>_<city>_strategic.pdf``
    Citywide reconstruction R² across feature-guided partial-survey ratios.

Canonical inputs:

- ``fold/main/<Country>/<City>/Fuse/Token_Concat_spatial_self/results.csv``
- ``ratio/main/<Country>/<City>/Fuse/Token_Concat_spatial_self/results.csv``
- ``sampling/main/<Country>/<City>/Fuse/Multi_Concat_pcahierachy/results.csv``

Indicator display labels and SDG groups are read from
``<processed_dir>/0labels/<Country>.csv`` when available. If a metadata table is
missing, raw target IDs are used so the numerical figure can still be produced.

Usage:

    python figures/supplementary/batch_per_city_figures.py
    python figures/supplementary/batch_per_city_figures.py --strict
    python figures/supplementary/batch_per_city_figures.py --country US --city Boston
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paths.yaml"
CITIES_CONFIG = REPO_ROOT / "configs" / "cities.yaml"
DEFAULT_OUT = REPO_ROOT / "data" / "figure_assets"

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

SDG_COLORS = {
    1: "#E5243B",
    3: "#4C9F38",
    4: "#C5192D",
    5: "#FF3A21",
    6: "#26BDE2",
    8: "#A21942",
    9: "#FD6925",
    10: "#DD1367",
    11: "#FD9D24",
    13: "#3F7E44",
    16: "#00689D",
}


def load_paths(config_path: Path) -> dict[str, str]:
    """Load paths.yaml and resolve ${key} references in declaration order."""
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    pattern = re.compile(r"\$\{([^}]+)\}")
    resolved: dict[str, str] = {}
    for key, value in cfg.items():
        if not isinstance(value, str):
            resolved[key] = value
            continue
        while True:
            match = pattern.search(value)
            if match is None:
                break
            ref = match.group(1)
            if ref not in resolved:
                raise KeyError(f"Unresolved variable ${{{ref}}} in {config_path}")
            value = value.replace(match.group(0), str(resolved[ref]))
        resolved[key] = value
    return resolved


def load_regions() -> list[tuple[str, str]]:
    with CITIES_CONFIG.open() as f:
        cfg = yaml.safe_load(f)
    return [
        (country, region["name"])
        for country, regions in cfg["regions"].items()
        for region in regions
    ]


def load_meta(meta_root: Path, country: str) -> pd.DataFrame | None:
    path = meta_root / f"{country}.csv"
    if not path.exists():
        return None
    meta = pd.read_csv(path)
    required = {"ID", "Code", "SDG"}
    if not required.issubset(meta.columns):
        print(
            f"[supplementary] metadata {path} lacks {sorted(required)}; "
            "using raw target IDs"
        )
        return None
    return meta[["ID", "Code", "SDG"]].drop_duplicates("ID")


def annotate(df: pd.DataFrame, meta: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    if meta is None:
        out["Code"] = out["target"].astype(str)
        out["SDG"] = 0
    else:
        out = out.merge(meta, left_on="target", right_on="ID", how="left")
        out["Code"] = out["Code"].fillna(out["target"].astype(str))
        out["SDG"] = pd.to_numeric(out["SDG"], errors="coerce").fillna(0).astype(int)
        out = out.drop(columns=["ID"])
    return out.sort_values(["SDG", "Code"]).reset_index(drop=True)


def read_results(path: Path, required: set[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    return df


def fold_results(reg_root: Path, country: str, city: str) -> pd.DataFrame | None:
    path = (
        reg_root
        / "fold"
        / "main"
        / country
        / city
        / "Fuse"
        / "Token_Concat_spatial_self"
        / "results.csv"
    )
    df = read_results(path, {"target", "r2"})
    if df is None:
        return None
    if "set" in df.columns:
        df = df.copy()
        df["fold"] = df["set"].astype(str).str.replace("setF", "", regex=False)
    elif "fold" not in df.columns:
        df = df.copy()
        df["fold"] = np.arange(len(df))
    return df[["target", "fold", "r2"]].dropna(subset=["r2"])


def _ratio_column(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if "ratio" in df.columns:
        return df
    if "set" not in df.columns:
        raise ValueError(f"{path}: missing both 'ratio' and 'set' columns")
    out = df.copy()
    suffix = out["set"].astype(str).str.replace("set", "", regex=False)
    out["ratio"] = pd.to_numeric(suffix, errors="coerce") / 10.0
    return out


def sampling_results(
    reg_root: Path,
    country: str,
    city: str,
    *,
    strategy: str,
) -> pd.DataFrame | None:
    if strategy == "random":
        path = (
            reg_root
            / "ratio"
            / "main"
            / country
            / city
            / "Fuse"
            / "Token_Concat_spatial_self"
            / "results.csv"
        )
    elif strategy == "strategic":
        path = (
            reg_root
            / "sampling"
            / "main"
            / country
            / city
            / "Fuse"
            / "Multi_Concat_pcahierachy"
            / "results.csv"
        )
    else:
        raise ValueError(strategy)

    df = read_results(path, {"target"})
    if df is None:
        return None
    df = _ratio_column(df, path)

    # Partial-survey figures describe reconstruction of the full city: observed
    # neighborhoods keep their measured values and unobserved neighborhoods use
    # model estimates. ``all_r2`` is therefore the canonical metric. Fall back
    # to held-out ``r2`` only for older result files that predate ``all_r2``.
    metric = "all_r2" if "all_r2" in df.columns else "r2"
    if metric not in df.columns:
        raise ValueError(f"{path}: missing 'all_r2'/'r2' metric")
    out = df[["target", "ratio", metric]].rename(columns={metric: "r2"})
    return out.dropna(subset=["ratio", "r2"])


def palette_for(df: pd.DataFrame) -> dict[str, str]:
    code_to_sdg = dict(zip(df["Code"], df["SDG"]))
    return {
        code: SDG_COLORS.get(int(code_to_sdg.get(code, 0)), "#808080")
        for code in df["Code"].drop_duplicates()
    }


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7)
    ax.axhline(0, linewidth=0.5, linestyle="--", zorder=0)


def add_sdg_legend(ax, df: pd.DataFrame) -> None:
    sdgs = sorted(s for s in set(df["SDG"]) if s in SDG_COLORS)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            color=SDG_COLORS[sdg],
            label=f"SDG {sdg}",
        )
        for sdg in sdgs
    ]
    if handles:
        ax.legend(handles=handles, frameon=False, fontsize=7, loc="best")


def plot_fold(df: pd.DataFrame, out_path: Path) -> None:
    order = df.drop_duplicates("Code")["Code"].tolist()
    palette = palette_for(df)

    fig, ax = plt.subplots(figsize=(18 / 2.5, 8 / 2.5))
    sns.boxplot(
        data=df,
        x="Code",
        y="r2",
        hue="Code",
        order=order,
        palette=palette,
        width=0.55,
        linewidth=0.8,
        fliersize=0,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="Code",
        y="r2",
        hue="Code",
        order=order,
        palette=palette,
        size=2.8,
        alpha=0.85,
        jitter=0.15,
        edgecolor="white",
        linewidth=0.3,
        legend=False,
        ax=ax,
    )
    style_axis(ax)
    ax.set_ylabel(r"Held-out $R^2$", fontsize=8)
    ax.set_xlabel("")
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    add_sdg_legend(ax, df)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sampling(df: pd.DataFrame, out_path: Path) -> None:
    order = df.drop_duplicates("Code")["Code"].tolist()
    palette = palette_for(df)

    fig, ax = plt.subplots(figsize=(18 / 2.5, 8 / 2.5))
    for code in order:
        sub = df[df["Code"] == code]
        grouped = sub.groupby("ratio", as_index=False)["r2"].agg(["mean", "std"]).reset_index()
        grouped = grouped.sort_values("ratio")
        ax.plot(
            grouped["ratio"],
            grouped["mean"],
            linewidth=1.1,
            marker="o",
            markersize=3,
            color=palette[code],
        )
        if grouped["std"].notna().any():
            std = grouped["std"].fillna(0)
            ax.fill_between(
                grouped["ratio"],
                grouped["mean"] - std,
                grouped["mean"] + std,
                color=palette[code],
                alpha=0.18,
                linewidth=0,
            )

    style_axis(ax)
    ax.set_xlabel("Surveyed-neighborhood ratio", fontsize=8)
    ax.set_ylabel(r"Citywide reconstruction $R^2$", fontsize=8)
    ax.set_xlim(0.05, 0.95)
    add_sdg_legend(ax, df)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--country", default=None, help="Optional country filter")
    p.add_argument("--city", default=None, help="Optional city/region filter")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested canonical result file is missing.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(
            f"{args.config} not found. Copy configs/paths.example.yaml to "
            "configs/paths.yaml and configure it first."
        )

    paths = load_paths(args.config)
    reg_root = Path(paths["regression_out_dir"])
    meta_root = Path(paths["processed_dir"]) / "0labels"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    regions = load_regions()
    if args.country is not None:
        regions = [item for item in regions if item[0] == args.country]
    if args.city is not None:
        regions = [item for item in regions if item[1] == args.city]
    if not regions:
        raise ValueError("No regions match the requested filters")

    written = 0
    missing: list[str] = []
    for country, city in regions:
        meta = load_meta(meta_root, country)

        fold = fold_results(reg_root, country, city)
        if fold is None:
            missing.append(f"fold: {country}/{city}")
        else:
            fold = annotate(fold, meta)
            out = args.output_dir / f"S_{country}_{city}_fold.pdf"
            plot_fold(fold, out)
            print(f"[supplementary] {out}")
            written += 1

        for strategy in ("random", "strategic"):
            df = sampling_results(reg_root, country, city, strategy=strategy)
            if df is None:
                missing.append(f"{strategy}: {country}/{city}")
                continue
            df = annotate(df, meta)
            out = args.output_dir / f"S_{country}_{city}_{strategy}.pdf"
            plot_sampling(df, out)
            print(f"[supplementary] {out}")
            written += 1

    if missing:
        print("[supplementary] missing canonical inputs:")
        for item in missing:
            print(f"  - {item}")
    print(f"[supplementary] wrote {written} figures")

    if args.strict and missing:
        raise RuntimeError(f"Missing {len(missing)} requested canonical inputs")
    if args.strict and written == 0:
        raise RuntimeError("No supplementary figures were generated")


if __name__ == "__main__":
    main()
