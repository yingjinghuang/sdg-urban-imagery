"""Render the Extended Data random and feature-guided sampling sweeps.

This maintained renderer replaces the historical ``ed_ratio_results.ipynb``
and ``ed_sampling_results.ipynb`` notebooks, which read the old
``data/regression_outputs_new`` workspace layout and the obsolete ``R2``
column. The current pipeline reads canonical regression outputs configured by
``configs/paths.yaml`` and uses ``all_r2`` for the citywide partial-survey
reconstruction metric.

By default both figures are generated:

- ``S_random_sampling.pdf/png``
- ``S_strategic_sampling.pdf/png``

Usage:
    python figures/extended/sampling_curves.py
    python figures/extended/sampling_curves.py --strategy random
    python figures/extended/sampling_curves.py --strategy feature-guided --strict
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.optimize import curve_fit


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paths.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "figure_assets"
CITIES_CONFIG = REPO_ROOT / "configs" / "cities.yaml"

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


def canonical_result_path(
    regression_root: Path, country: str, city: str, strategy: str
) -> Path:
    if strategy == "random":
        return (
            regression_root
            / "ratio"
            / "main"
            / country
            / city
            / "Fuse"
            / "Token_Concat_spatial_self"
            / "results.csv"
        )
    if strategy == "feature-guided":
        return (
            regression_root
            / "sampling"
            / "main"
            / country
            / city
            / "Fuse"
            / "Multi_Concat_pcahierachy"
            / "results.csv"
        )
    raise ValueError(strategy)


def load_region_data(
    path: Path, metadata_path: Path, *, country: str, city: str
) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"target", "ratio"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")

    metric = "all_r2" if "all_r2" in df.columns else "r2"
    if metric not in df.columns:
        raise ValueError(f"{path}: missing all_r2/r2 metric")

    out = df[["target", "ratio", metric]].rename(columns={metric: "R2"}).copy()
    out["country"] = country
    out["city"] = city

    if metadata_path.exists():
        meta = pd.read_csv(metadata_path)
        if {"ID", "SDG"}.issubset(meta.columns):
            out = out.merge(
                meta[["ID", "SDG"]].drop_duplicates("ID"),
                left_on="target",
                right_on="ID",
                how="left",
            ).drop(columns=["ID"])
        else:
            out["SDG"] = 0
    else:
        out["SDG"] = 0

    out["SDG"] = pd.to_numeric(out["SDG"], errors="coerce").fillna(0).astype(int)
    return out.dropna(subset=["ratio", "R2"])


def log_fit(x, a, b):
    return a + b * np.log(x)


def display_city(country: str, city: str) -> str:
    if city == "All":
        return f"Selected cities in {country}"
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


def render_strategy(
    regression_root: Path,
    processed_root: Path,
    output_dir: Path,
    strategy: str,
    *,
    strict: bool,
) -> int:
    region_frames: list[tuple[str, str, pd.DataFrame]] = []
    missing: list[Path] = []

    for country, city in load_regions():
        result_path = canonical_result_path(regression_root, country, city, strategy)
        if not result_path.exists():
            missing.append(result_path)
            continue
        frame = load_region_data(
            result_path,
            processed_root / "0labels" / f"{country}.csv",
            country=country,
            city=city,
        )
        if not frame.empty:
            region_frames.append((country, city, frame))

    if strict and missing:
        preview = "\n".join(f"  - {p}" for p in missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} canonical {strategy} result files.\n{preview}"
        )
    if not region_frames:
        raise RuntimeError(f"No canonical {strategy} result files were found")

    region_frames.sort(key=lambda item: (item[0], item[1]))
    n_cols = 4
    n_rows = math.ceil(len(region_frames) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), squeeze=False)

    for panel_index, (country, city, df) in enumerate(region_frames):
        ax = axes.flat[panel_index]
        plot_df = df.copy()
        plot_df["ratio_percent"] = plot_df["ratio"] * 100

        sdgs = sorted(s for s in plot_df["SDG"].unique() if s != 0)
        palette = {s: SDG_COLORS.get(s, "#999999") for s in sdgs}
        if not sdgs:
            sdgs = [0]
            palette = {0: "#999999"}

        sns.scatterplot(
            data=plot_df,
            x="ratio_percent",
            y="R2",
            hue="SDG",
            palette=palette,
            legend=False,
            alpha=0.55,
            ax=ax,
        )

        for offset, sdg in enumerate(sdgs):
            group = plot_df[plot_df["SDG"] == sdg].dropna(subset=["ratio", "R2"])
            if len(group) < 3 or group["ratio"].nunique() < 2:
                continue
            x = group["ratio"].to_numpy(dtype=float)
            y = group["R2"].to_numpy(dtype=float)
            if np.any(x <= 0):
                continue
            try:
                params, covariance = curve_fit(log_fit, x, y, maxfev=10000)
            except (RuntimeError, ValueError):
                continue

            color = palette[sdg]
            x_fit = np.linspace(max(0.01, x.min()), min(1.0, x.max()), 500)
            y_fit = log_fit(x_fit, *params)
            ax.plot(x_fit * 100, y_fit, color=color, linewidth=1.4)

            if np.all(np.isfinite(covariance)):
                errors = np.sqrt(np.diag(covariance))
                upper = log_fit(x_fit, *(params + 1.96 * errors))
                lower = log_fit(x_fit, *(params - 1.96 * errors))
                ax.fill_between(x_fit * 100, lower, upper, color=color, alpha=0.18)

            if params[1] != 0:
                ratio_at_08 = float(np.exp((0.8 - params[0]) / params[1]))
                if 0 < ratio_at_08 <= 1:
                    ax.scatter(ratio_at_08 * 100, 0.8, color=color, s=20, zorder=5)
                    ax.annotate(
                        f"SDG {sdg}: {ratio_at_08 * 100:.1f}%",
                        xy=(ratio_at_08 * 100, 0.8),
                        xytext=(ratio_at_08 * 100 - 8, 0.08 + 0.07 * offset),
                        fontsize=7,
                        color=color,
                    )

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(display_city(country, city), fontsize=10)
        ax.set_xlabel("Sampling ratio (%)" if panel_index // n_cols == n_rows - 1 else "")
        ax.set_ylabel(r"Citywide reconstruction $R^2$" if panel_index % n_cols == 0 else "")
        ax.tick_params(labelsize=8)

    for empty_index in range(len(region_frames), n_rows * n_cols):
        fig.delaxes(axes.flat[empty_index])

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "S_random_sampling" if strategy == "random" else "S_strategic_sampling"
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[extended] {strategy}: {len(region_frames)} regions -> {output_dir / (stem + '.pdf')}")
    if missing:
        print(f"[extended] skipped {len(missing)} regions with missing canonical inputs")
    return len(region_frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--strategy",
        choices=["random", "feature-guided", "both"],
        default="both",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(
            f"{args.config} not found. Copy configs/paths.example.yaml to configs/paths.yaml first."
        )
    paths = load_paths(args.config)
    regression_root = Path(paths["regression_out_dir"])
    processed_root = Path(paths["processed_dir"])

    strategies = (
        ["random", "feature-guided"] if args.strategy == "both" else [args.strategy]
    )
    for strategy in strategies:
        render_strategy(
            regression_root,
            processed_root,
            args.output_dir,
            strategy,
            strict=args.strict,
        )


if __name__ == "__main__":
    main()
