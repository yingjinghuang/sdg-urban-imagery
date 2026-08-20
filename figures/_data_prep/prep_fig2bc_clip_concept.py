"""Prepare the CLIP concept-probe tables for manuscript Fig. 3b-c.

The manuscript/Supplementary Information defines the concept sets and the final
figure displays four concepts per modality:

* RS: concrete, rooftop, vegetation, soil
* SV: building, window, road, tree

CLIP scoring itself is performed by ``scripts/compute_clip_scores.sh``. This
script then measures how strongly the per-image self- and spatial-contrastive
768-d features explain the CLIP scores using the RidgeCV analysis retained from
the original study code.

The historical analysis used a 10,000-image working subset. The old notebook did
not record a random seed; this maintained release uses seed 42 so raw-score
recomputation is deterministic. The deposited ``clip_concept_{sv,rs}.csv``
tables remain the authoritative inputs for exact figure-only reproduction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src._common.io import feature_columns, read_feature_table
from src._common.paths import remap_image_paths


DEFAULT_DATA_ROOT = REPO_ROOT / "data"

DISPLAY_CONCEPTS = {
    "RS": ["concrete", "rooftop", "vegetation", "soil"],
    "SV": ["building", "window", "road", "tree"],
}
SCORED_CONCEPTS = {
    "RS": ["concrete", "rooftop", "vegetation", "soil"],
    "SV": [
        "building",
        "car",
        "fence",
        "pole",
        "window",
        "road",
        "tree",
        "chaotic",
        "orderly",
        "depressing",
        "lively",
        "safe",
        "dilapidated",
        "wealthy",
    ],
}
ALPHAS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modality", choices=["SV", "RS", "sv", "rs"], required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_feature_branch(feature_dir: Path, variant: str) -> pd.DataFrame:
    files = sorted(feature_dir.glob(f"Mocov3VITB-{variant}-*.h5"))
    if not files:
        raise FileNotFoundError(
            f"No {variant} raw feature files found under {feature_dir}"
        )

    frames = []
    for path in files:
        frame = read_feature_table(path)
        if "index" not in frame.columns:
            raise KeyError(f"{path} has no 'index' image-path column")
        frame = frame.copy()
        frame["index"] = remap_image_paths(frame["index"].astype(str))
        frames.append(frame)

    output = pd.concat(frames, ignore_index=True)
    if output["index"].duplicated().any():
        output = output.drop_duplicates("index", keep="first")
    return output


def align_inputs(
    scores: pd.DataFrame,
    self_features: pd.DataFrame,
    spatial_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = scores.copy()
    scores["path"] = remap_image_paths(scores["path"].astype(str))
    scores = scores.drop_duplicates("path", keep="first").set_index("path")

    self_features = self_features.set_index("index")
    spatial_features = spatial_features.set_index("index")
    common = scores.index.intersection(self_features.index).intersection(spatial_features.index)
    if common.empty:
        raise RuntimeError(
            "No common image paths between CLIP scores and self/spatial feature tables."
        )

    common = common.sort_values()
    return (
        scores.loc[common],
        self_features.loc[common],
        spatial_features.loc[common],
    )


def select_rows(
    scores: pd.DataFrame,
    self_features: pd.DataFrame,
    spatial_features: pd.DataFrame,
    *,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if sample_size <= 0 or len(scores) <= sample_size:
        return scores, self_features, spatial_features

    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(scores), size=sample_size, replace=False))
    return (
        scores.iloc[positions],
        self_features.iloc[positions],
        spatial_features.iloc[positions],
    )


def probe(
    scores: pd.DataFrame,
    self_features: pd.DataFrame,
    spatial_features: pd.DataFrame,
    concepts: list[str],
) -> pd.DataFrame:
    self_cols = feature_columns(self_features.reset_index())
    spatial_cols = feature_columns(spatial_features.reset_index())
    if not self_cols or not spatial_cols:
        raise RuntimeError("Feature tables contain no numeric embedding columns")

    x_self = self_features[self_cols].fillna(0).to_numpy()
    x_spatial = spatial_features[spatial_cols].fillna(0).to_numpy()
    results = []

    for concept in concepts:
        y = scores[concept].to_numpy()

        self_model = make_pipeline(
            StandardScaler(), RidgeCV(alphas=ALPHAS, scoring="r2", cv=5)
        )
        self_model.fit(x_self, y)
        r2_self = float(self_model.score(x_self, y))

        spatial_model = make_pipeline(
            StandardScaler(), RidgeCV(alphas=ALPHAS, scoring="r2", cv=5)
        )
        spatial_model.fit(x_spatial, y)
        r2_spatial = float(spatial_model.score(x_spatial, y))

        print(
            f"[clip-probe] {concept}: self={r2_self:.4f} spatial={r2_spatial:.4f}"
        )
        results.append(
            {"concept": concept, "r2_self": r2_self, "r2_spatial": r2_spatial}
        )

    return pd.DataFrame(results)


def main() -> None:
    args = parse_args()
    modality = args.modality.upper()
    data_root = args.data_root.resolve()

    score_file = data_root / "processed" / f"{modality.lower()}_clip_scores.pkl"
    feature_dir = data_root / "features" / "Raw" / "US" / "LosAngeles" / modality
    output_file = (
        data_root / "processed" / "fig" / f"clip_concept_{modality.lower()}.csv"
    )

    if not score_file.exists():
        raise FileNotFoundError(
            f"Missing {score_file}; run `bash scripts/compute_clip_scores.sh {modality}` first."
        )

    scores = pd.read_pickle(score_file)
    if "path" not in scores.columns:
        raise KeyError(f"{score_file} has no 'path' column")

    expected = SCORED_CONCEPTS[modality]
    missing = [concept for concept in expected if concept not in scores.columns]
    if missing:
        raise KeyError(
            f"{score_file} is missing manuscript {modality} concepts: {missing}"
        )

    self_features = load_feature_branch(feature_dir, "self")
    spatial_features = load_feature_branch(feature_dir, "spatial")
    scores, self_features, spatial_features = align_inputs(
        scores, self_features, spatial_features
    )
    scores, self_features, spatial_features = select_rows(
        scores,
        self_features,
        spatial_features,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    print(
        f"[clip-probe] modality={modality} aligned_images={len(scores)} "
        f"sample_size={args.sample_size} seed={args.seed}"
    )
    result = probe(
        scores,
        self_features,
        spatial_features,
        DISPLAY_CONCEPTS[modality],
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    print(f"[clip-probe] saved {output_file}")


if __name__ == "__main__":
    main()
