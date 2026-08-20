"""Prepare the legacy figure-notebook input layout from canonical outputs.

The training/reproduction launchers in this repository now write to a clean
canonical layout under ``regression_out_dir`` (for example
``fold/main/<Country>/<City>/...`` and ``sampling/main/<Country>/<City>/...``).
Several plotting notebooks predate that cleanup and still read the historical
``data/regression_outputs/regmodels_*`` paths.

This utility creates *small CSV compatibility copies* under ``<repo>/data`` so
those notebooks can be executed without changing their plotting code. It does
not retrain models or alter reported metrics. For fold experiments, the five
fold rows are averaged per indicator before staging because the legacy figure
preparation notebooks expect one city/indicator row.

It also stages the small indicator-code tables and per-city ``labels.pkl``
files needed by the Moran's-I preparation notebook when they are available in
the configured processed/labels directories.

Usage:
    python scripts/prepare_figure_inputs.py
    python scripts/prepare_figure_inputs.py --config configs/paths.yaml --strict
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False)
    print(f"[figure-inputs] {path.relative_to(REPO_ROOT)}")


def fold_mean(src: Path, *, add_fold_column: bool = False) -> pd.DataFrame:
    """Average current per-fold results to one row per target for legacy plots."""
    df = pd.read_csv(src)
    required = ["target", "mae", "mse", "r2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{src}: missing required columns {missing}")
    out = df[required].groupby("target", as_index=False).mean(numeric_only=True)
    if add_fold_column:
        # One legacy prep notebook explicitly drops this column.
        out["fold"] = "mean"
    return out


def ratio_table(src: Path) -> pd.DataFrame:
    """Convert current ratio/sampling results to the legacy plotting schema."""
    df = pd.read_csv(src)
    if "ratio" not in df.columns and "set" in df.columns:
        suffix = df["set"].astype(str).str.replace("set", "", regex=False)
        df["ratio"] = pd.to_numeric(suffix, errors="coerce") / 10.0

    keep = [
        "target",
        "mae",
        "mse",
        "r2",
        "all_mae",
        "all_mse",
        "all_r2",
        "ratio",
    ]
    missing = [c for c in ("target", "all_r2", "ratio") if c not in df.columns]
    if missing:
        raise ValueError(f"{src}: missing required columns {missing}")
    out = df[[c for c in keep if c in df.columns]].copy()
    # Historical sampling notebooks expect a ``k`` column, but do not use it in
    # the plotted quantities. A constant placeholder preserves that schema.
    out["k"] = 0
    return out


def stage_fold_outputs(reg_root: Path, legacy_root: Path) -> int:
    count = 0

    # Proposed model: Fig. 2a prep uses regmodels_spatial, while Fig. 3a prep
    # uses regmodels_spatial_self. Both should point to the same canonical run.
    for src in reg_root.glob("fold/main/*/*/Fuse/Token_Concat_spatial_self/results.csv"):
        rel = src.relative_to(reg_root).parts
        country, city = rel[2], rel[3]
        out = fold_mean(src)
        for family in ("regmodels_spatial", "regmodels_spatial_self"):
            dst = legacy_root / family / "Fold" / country / city / "Fuse" / "Token_Concat_spatial_self" / "results.csv"
            write_csv(out, dst)
        count += 1

    # Single-modal Fig. 3a runs.
    for modality in ("SV", "RS"):
        for src in reg_root.glob(f"fold/main/*/*/{modality}/Token_Concat_self_spatial/results.csv"):
            rel = src.relative_to(reg_root).parts
            country, city = rel[2], rel[3]
            dst = legacy_root / "regmodels_spatial_self" / "Fold" / country / city / modality / "Token_Concat_self_spatial" / "results.csv"
            write_csv(fold_mean(src), dst)
            count += 1

    # ImageNet baseline.
    for src in reg_root.glob("fold/imagenet/*/*/Fuse/Token_Concat_ImageNet/results.csv"):
        rel = src.relative_to(reg_root).parts
        country, city = rel[2], rel[3]
        dst = legacy_root / "regmodels_imagenet" / "Fold" / country / city / "Fuse" / "Token_Concat_ImageNet" / "results.csv"
        write_csv(fold_mean(src), dst)
        count += 1

    # Segmentation baseline. Its historical prep notebook explicitly removes a
    # ``fold`` column, so retain a harmless marker after averaging.
    for src in reg_root.glob("fold/segmentation/*/*/SV/Multi_segmentation/results.csv"):
        rel = src.relative_to(reg_root).parts
        country, city = rel[2], rel[3]
        dst = legacy_root / "regmodels_seg" / "Fold" / country / city / "SV" / "Multi_segmentation" / "results.csv"
        write_csv(fold_mean(src, add_fold_column=True), dst)
        count += 1

    return count


def stage_sampling_outputs(reg_root: Path, legacy_root: Path) -> int:
    count = 0

    # Feature-guided sampling: current canonical four-token regression.
    for src in reg_root.glob("sampling/main/*/*/Fuse/Multi_Concat_pcahierachy/results.csv"):
        rel = src.relative_to(reg_root).parts
        country, city = rel[2], rel[3]
        dst = legacy_root / "regmodels_spatial_self" / "Sampling_kcenter" / country / city / "Fuse" / "Token_Concat_spatial_self_Spatial" / "results.csv"
        write_csv(ratio_table(src), dst)
        count += 1

    # Random sampling baseline.
    for src in reg_root.glob("ratio/main/*/*/Fuse/Token_Concat_spatial_self/results.csv"):
        rel = src.relative_to(reg_root).parts
        country, city = rel[2], rel[3]
        dst = legacy_root / "regmodels_spatial_self" / "Ratio" / country / city / "Fuse" / "Token_Concat_spatial_self_Spatial_Random" / "results.csv"
        write_csv(ratio_table(src), dst)
        count += 1

    # Non-spatial feature-guided run used by the Fig. 3f-g preparation notebook.
    for src in reg_root.glob("sampling/main_no_geo/*/*/Fuse/Multi_Concat_pcahierachy/results.csv"):
        rel = src.relative_to(reg_root).parts
        country, city = rel[2], rel[3]
        dst = legacy_root / "regmodels_spatial_self" / "Ratio_no_geo" / country / city / "Fuse" / "results.csv"
        write_csv(ratio_table(src), dst)
        count += 1

    return count


def copy_if_missing(src: Path, dst: Path) -> bool:
    if not src.exists() or dst.exists():
        return False
    ensure_parent(dst)
    shutil.copy2(src, dst)
    print(f"[figure-inputs] copied metadata -> {dst.relative_to(REPO_ROOT)}")
    return True


def stage_metadata(paths: dict[str, str], data_dir: Path) -> int:
    """Stage small metadata files referenced by the historical notebooks."""
    count = 0
    processed_src = Path(paths["processed_dir"])
    labels_src = Path(paths["labels_dir"])

    # SDG indicator-code dictionaries used by most figure notebooks.
    for src in (processed_src / "0labels").glob("*.csv"):
        count += int(copy_if_missing(src, data_dir / "processed" / "0labels" / src.name))

    # Per-city unnormalized labels used to compute Moran's I. Prefer labels_dir,
    # but fall back to processed_dir if a local release stores them there.
    candidates = list(labels_src.glob("*/*/labels.pkl"))
    candidates += list(processed_src.glob("*/*/labels.pkl"))
    seen: set[tuple[str, str]] = set()
    for src in candidates:
        country, city = src.parent.parent.name, src.parent.name
        key = (country, city)
        if key in seen:
            continue
        seen.add(key)
        count += int(copy_if_missing(src, data_dir / "processed" / country / city / "labels.pkl"))

    (data_dir / "processed" / "fig").mkdir(parents=True, exist_ok=True)
    (data_dir / "figure_assets").mkdir(parents=True, exist_ok=True)
    return count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "paths.yaml",
        help="Path configuration used by the training launchers.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if no canonical regression outputs were found.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(
            f"{args.config} not found. Copy configs/paths.example.yaml to configs/paths.yaml first."
        )

    paths = load_paths(args.config)
    reg_root = Path(paths["regression_out_dir"])
    data_dir = REPO_ROOT / "data"
    legacy_root = data_dir / "regression_outputs"
    legacy_root.mkdir(parents=True, exist_ok=True)

    print(f"[figure-inputs] canonical regression root: {reg_root}")
    print(f"[figure-inputs] notebook data root: {data_dir}")

    n_fold = stage_fold_outputs(reg_root, legacy_root)
    n_sampling = stage_sampling_outputs(reg_root, legacy_root)
    n_meta = stage_metadata(paths, data_dir)

    print(
        f"[figure-inputs] done: {n_fold} fold sources, "
        f"{n_sampling} sampling/ratio sources, {n_meta} metadata files staged"
    )
    if args.strict and (n_fold + n_sampling == 0):
        raise RuntimeError(
            "No canonical regression outputs were found. Run the relevant regression launchers "
            "or unpack the deposited regression outputs first."
        )


if __name__ == "__main__":
    main()
