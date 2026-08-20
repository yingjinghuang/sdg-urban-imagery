"""Render per-indicator held-out R² bars for four representative cities.

The script reads the canonical five-fold regression outputs and indicator
metadata configured by `configs/paths.yaml`. Fold rows are averaged per target
before plotting. No machine-specific font, flag, or output paths are required.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paths.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "figure_assets" / "indicator_bars.pdf"

CITIES = [
    ("US", "Philadelphia", "Philadelphia, US"),
    ("Australia", "Melbourne", "Melbourne, Australia"),
    ("Brazil", "RiodeJaneiro", "Rio de Janeiro, Brazil"),
    ("China", "HongKong", "Hong Kong, China"),
]

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

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False


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


def load_city_data(
    regression_root: Path,
    processed_root: Path,
    country: str,
    city: str,
) -> pd.DataFrame:
    result_path = (
        regression_root
        / "fold"
        / "main"
        / country
        / city
        / "Fuse"
        / "Token_Concat_spatial_self"
        / "results.csv"
    )
    if not result_path.exists():
        raise FileNotFoundError(result_path)

    df = pd.read_csv(result_path)
    if not {"target", "r2"}.issubset(df.columns):
        raise ValueError(f"{result_path}: expected target/r2 columns")
    df = df.groupby("target", as_index=False)["r2"].mean()

    meta_path = processed_root / "0labels" / f"{country}.csv"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = pd.read_csv(meta_path)
    if not {"ID", "SDG"}.issubset(meta.columns):
        raise ValueError(f"{meta_path}: expected ID/SDG columns")

    keep = [column for column in ("ID", "Code", "SDG") if column in meta.columns]
    df = df.merge(
        meta[keep].drop_duplicates("ID"),
        left_on="target",
        right_on="ID",
        how="left",
    ).drop(columns=["ID"])
    df["SDG"] = pd.to_numeric(df["SDG"], errors="coerce").fillna(0).astype(int)
    if "Code" in df.columns:
        df["label"] = df["Code"].fillna(df["target"]).astype(str)
    else:
        df["label"] = df["target"].astype(str)
    return df.sort_values("r2").reset_index(drop=True)


def render(
    regression_root: Path,
    processed_root: Path,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    used_sdgs: set[int] = set()

    for index, (country, city, title) in enumerate(CITIES):
        ax = axes[index]
        df = load_city_data(regression_root, processed_root, country, city)
        used_sdgs.update(sdg for sdg in df["SDG"].unique() if sdg in SDG_COLORS)

        colors = [SDG_COLORS.get(sdg, "#808080") for sdg in df["SDG"]]
        positions = np.arange(len(df))
        bars = ax.barh(
            positions,
            df["r2"],
            height=0.7,
            color=colors,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )

        for row, (value, bar) in enumerate(zip(df["r2"], bars)):
            offset = 0.01 if value >= 0 else -0.01
            alignment = "left" if value >= 0 else "right"
            ax.text(value + offset, row, f"{value:.2f}", va="center", ha=alignment, fontsize=7)

        ax.set_yticks(positions)
        ax.set_yticklabels(df["label"], fontsize=8)
        ax.set_xlabel(r"Held-out $R^2$", fontsize=10)
        lower = min(0.0, float(df["r2"].min()) - 0.05)
        upper = max(1.0, float(df["r2"].max()) + 0.08)
        ax.set_xlim(lower, upper)
        if lower < 0:
            ax.axvline(0, linewidth=0.5)
        ax.xaxis.grid(True, linestyle="--", alpha=0.35, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"({chr(ord('a') + index)}) {title}", loc="left", fontsize=11)

    handles = [
        Patch(facecolor=SDG_COLORS[sdg], edgecolor="none", label=f"SDG {sdg}")
        for sdg in sorted(used_sdgs)
    ]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[extended] {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(args.config)
    paths = load_paths(args.config)
    render(
        Path(paths["regression_out_dir"]),
        Path(paths["processed_dir"]),
        args.output,
    )


if __name__ == "__main__":
    main()
