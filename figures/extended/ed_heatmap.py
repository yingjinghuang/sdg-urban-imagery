"""Render the Extended Data city-by-SDG performance heatmap.

Reads the canonical five-fold regression results configured by
`configs/paths.yaml`, averages held-out R² across folds/indicators within each
city-SDG pair, and writes the figure under `data/figure_assets` by default.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paths.yaml"
CITIES_CONFIG = REPO_ROOT / "configs" / "cities.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "figure_assets" / "heatmap_city_sdg.pdf"
DEFAULT_FLAG_DIR = REPO_ROOT / "data" / "assets" / "flags"

FLAG_NAMES = {
    "Australia": "au.png",
    "Brazil": "br.png",
    "China": "cn.png",
    "France": "fr.png",
    "Nigeria": "ng.png",
    "Portugal": "pt.png",
    "US": "us.png",
}

SDG_ORDER = [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 16]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def load_paths(config_path: Path) -> dict[str, str]:
    with config_path.open() as handle:
        cfg = yaml.safe_load(handle)
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
    with CITIES_CONFIG.open() as handle:
        cfg = yaml.safe_load(handle)
    return [
        (country, region["name"])
        for country, regions in cfg["regions"].items()
        for region in regions
    ]


def display_city(country: str, city: str) -> str:
    if city == "All":
        return country
    replacements = {
        "BeloHorizonte": "Belo Horizonte",
        "HongKong": "Hong Kong",
        "LosAngeles": "Los Angeles",
        "NewYork": "New York",
        "PortoAlegre": "Porto Alegre",
        "RiodeJaneiro": "Rio de Janeiro",
        "SanFrancisco": "San Francisco",
    }
    return replacements.get(city, city)


def collect_results(reg_root: Path, processed_root: Path, *, strict: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[Path] = []

    for country, city in load_regions():
        results_path = (
            reg_root
            / "fold"
            / "main"
            / country
            / city
            / "Fuse"
            / "Token_Concat_spatial_self"
            / "results.csv"
        )
        if not results_path.exists():
            missing.append(results_path)
            continue

        df = pd.read_csv(results_path)
        if not {"target", "r2"}.issubset(df.columns):
            raise ValueError(f"{results_path}: expected target/r2 columns")
        df = df.groupby("target", as_index=False)["r2"].mean()

        meta_path = processed_root / "0labels" / f"{country}.csv"
        if not meta_path.exists():
            if strict:
                raise FileNotFoundError(meta_path)
            continue
        meta = pd.read_csv(meta_path)
        if not {"ID", "SDG"}.issubset(meta.columns):
            raise ValueError(f"{meta_path}: expected ID/SDG columns")

        df = df.merge(
            meta[["ID", "SDG"]].drop_duplicates("ID"),
            left_on="target",
            right_on="ID",
            how="inner",
        ).drop(columns=["ID"])
        df["SDG"] = pd.to_numeric(df["SDG"], errors="coerce")
        df = df.dropna(subset=["SDG", "r2"])
        df["SDG"] = df["SDG"].astype(int)
        df["country"] = country
        df["city"] = display_city(country, city)
        frames.append(df)

    if strict and missing:
        preview = "\n".join(f"  - {p}" for p in missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} canonical fold result files.\n{preview}"
        )
    if not frames:
        raise RuntimeError("No canonical fold results were found")
    return pd.concat(frames, ignore_index=True)


def render(df: pd.DataFrame, output_path: Path, flag_dir: Path) -> None:
    grouped = df.groupby(["country", "city", "SDG"], as_index=False)["r2"].mean()
    grouped["SDG_label"] = grouped["SDG"].map(lambda value: f"SDG {value}")

    city_country = (
        grouped[["country", "city"]]
        .drop_duplicates()
        .sort_values(["country", "city"])
        .reset_index(drop=True)
    )
    city_order = city_country["city"].tolist()
    sdg_labels = [f"SDG {value}" for value in SDG_ORDER]

    pivot = grouped.pivot(index="city", columns="SDG_label", values="r2")
    pivot = pivot.reindex(index=city_order, columns=sdg_labels)
    available = [column for column in sdg_labels if not pivot[column].isna().all()]
    pivot = pivot[available]

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = pivot.isna()
    ax.set_facecolor("#f0f0f0")
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8},
        cmap="RdYlGn",
        mask=mask,
        vmin=0,
        vmax=0.9,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": r"$R^2$", "shrink": 0.8},
        ax=ax,
    )

    for row in range(len(pivot)):
        for column in range(len(pivot.columns)):
            if mask.iloc[row, column]:
                ax.add_patch(
                    plt.Rectangle(
                        (column, row),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor="#cccccc",
                        linewidth=0,
                    )
                )

    ax.set_xlabel("SDG category", fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=45, labelsize=10)
    ax.tick_params(axis="y", labelrotation=0, labelsize=9)

    # Add country flags only when the optional local assets are available.
    countries = city_country["country"].tolist()
    for row, country in enumerate(countries):
        flag_name = FLAG_NAMES.get(country)
        flag_path = flag_dir / flag_name if flag_name else None
        if flag_path is not None and flag_path.exists():
            image = mpimg.imread(flag_path)
            box = OffsetImage(image, zoom=0.02)
            ax.add_artist(
                AnnotationBbox(
                    box,
                    (-0.02, row + 0.5),
                    xycoords=("axes fraction", "data"),
                    frameon=False,
                    box_alignment=(1.0, 0.5),
                )
            )

    previous = None
    for row, country in enumerate(countries):
        if previous is not None and country != previous:
            ax.axhline(row, linewidth=1.5)
        previous = country

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[extended] {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--flag-dir", type=Path, default=DEFAULT_FLAG_DIR)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(args.config)
    paths = load_paths(args.config)
    df = collect_results(
        Path(paths["regression_out_dir"]),
        Path(paths["processed_dir"]),
        strict=args.strict,
    )
    render(df, args.output, args.flag_dir)


if __name__ == "__main__":
    main()
