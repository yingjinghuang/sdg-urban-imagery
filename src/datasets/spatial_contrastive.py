"""Build a spatial-contrastive training set.

For each neighborhood (GEOID), sample pairs of images taken inside that
neighborhood. The resulting (path1, path2, GEOID) pickle is consumed by
``pretrain/moco_sv.py`` and ``pretrain/moco_rs.py``.

Replaces ``2build_train_datasets/spatial_contrastive_city.py`` and
``spatial_contrastive_city_rs.py`` (95% duplicate code; the only real
difference was input-path naming, factored out here as ``--modality``).

Usage:
    python -m src.datasets.spatial_contrastive \
        --country China --city HongKong --modality sv \
        --input  ${PROCESSED_DIR}/China/HongKong/paths.pkl \
        --output ${PROCESSED_DIR}/train_datasets/spatial_China_HongKong.pkl \
        --sample_size 1000000
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

GEOID_COL = "GEOID"


def _sample_for_group(args: tuple) -> pd.DataFrame:
    """Pair-sample within one GEOID. Worker for the multiprocessing pool."""
    geoid, paths, target_n = args
    n = len(paths)
    if n < 2:
        return pd.DataFrame(columns=["path1", "path2", GEOID_COL])

    total = n * (n - 1) // 2
    if total <= target_n:
        pairs = list(itertools.combinations(paths, 2))
    else:
        # Random index pairs with i1 < i2; oversample by 2x to compensate for
        # rejected ties / wrong-order draws, then truncate.
        idx1 = np.random.randint(0, n, size=target_n * 2)
        idx2 = np.random.randint(0, n, size=target_n * 2)
        keep = (idx1 < idx2) & (idx1 != idx2)
        paths_arr = np.asarray(paths)
        pairs = list(zip(paths_arr[idx1[keep]], paths_arr[idx2[keep]]))[:target_n]

    out = pd.DataFrame(pairs, columns=["path1", "path2"])
    out[GEOID_COL] = geoid
    return out


def build(
    input_path: str,
    output_path: str,
    *,
    sample_size: int = 1_000_000,
    seed: int = 42,
    n_workers: int | None = None,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[spatial-contrastive] reading {input_path}")
    df = pd.read_pickle(input_path)
    n_groups = df[GEOID_COL].nunique()
    per_group = int((sample_size / n_groups) * 2)  # see truncation note in _sample_for_group
    print(f"[spatial-contrastive] {n_groups} GEOIDs, target {per_group} pairs/group")

    grouped = df.groupby(GEOID_COL)["path"].apply(list).reset_index(name="paths")
    args = [(row.GEOID, row.paths, per_group) for row in grouped.itertuples(index=False)]

    n_workers = n_workers or mp.cpu_count()
    with mp.Pool(n_workers) as pool:
        per_group_dfs = list(tqdm(pool.imap(_sample_for_group, args), total=len(args)))

    all_pairs = pd.concat(per_group_dfs, ignore_index=True)
    print(f"[spatial-contrastive] {len(all_pairs)} pairs before global cap")

    if len(all_pairs) > sample_size:
        all_pairs = all_pairs.sample(n=sample_size, random_state=seed)

    all_pairs.to_pickle(output_path)
    print(f"[spatial-contrastive] saved {len(all_pairs)} pairs -> {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", required=True,
                   help="Image-manifest pickle (paths.pkl for SV, rs_paths.pkl for RS).")
    p.add_argument("--output", required=True,
                   help="Output pickle path.")
    p.add_argument("--sample_size", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_workers", type=int, default=None,
                   help="Workers in the multiprocessing pool (default: cpu_count).")
    # These two are informational/bookkeeping — used by callers to label outputs.
    p.add_argument("--country", default="", help="Country tag (for logging).")
    p.add_argument("--city", default="", help="City tag (for logging).")
    p.add_argument("--modality", default="sv", choices=["sv", "rs"],
                   help="Modality tag (for logging).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build(args.input, args.output, sample_size=args.sample_size,
          seed=args.seed, n_workers=args.n_workers)


if __name__ == "__main__":
    main()
