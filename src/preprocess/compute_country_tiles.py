"""Compute satellite-download bounding-box tiles for one country.

Reads the country's neighborhood-boundary GeoDataFrame, dissolves it to
a single union geometry, splits that union into its individual polygon
components (one per disconnected region), and emits one line per
component in the format expected by the satellite tile downloader:

    PART_ID|minx,miny|maxx,maxy|zoom

For example France produces ~83 components (mainland + overseas + islands).

Replaces the per-country ``fr_boundary.ipynb`` / ``pt_boundary.ipynb`` /
``hk_boundary.ipynb`` / ``ng_boundary.ipynb`` notebooks, which all
implemented this same routine with country-specific input paths.

Usage:
    python -m src.preprocess.compute_country_tiles \
        --labels  ${PROCESSED_DIR}/France/All/labels.pkl \
        --output  ${DATA_ROOT}/rs_download_txt/France.txt \
        --zoom    19
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def country_to_tiles(
    labels_path: str,
    output_path: str,
    *,
    zoom: int = 19,
) -> int:
    """Read ``labels_path`` neighborhood polygons and write a tile manifest.

    Returns the number of tiles written.
    """
    gdf = pd.read_pickle(labels_path)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

    # Dissolve all neighborhood polygons into a single (multi-)polygon, then
    # split into individual contiguous regions. Tiles are bounding boxes of
    # each region, which is much cheaper to download than tiling the union
    # bounding box (especially for countries with islands).
    union = gdf["geometry"].unary_union
    if union.geom_type == "MultiPolygon":
        regions = list(union.geoms)
    elif union.geom_type == "Polygon":
        regions = [union]
    else:
        raise ValueError(f"Unexpected geometry type: {union.geom_type}")

    regions_gdf = gpd.GeoDataFrame(geometry=regions, crs=gdf.crs)
    bounds = regions_gdf.geometry.envelope.bounds
    regions_gdf = pd.concat([regions_gdf, bounds], axis=1)
    regions_gdf["PART"] = [f"P{i:02d}" for i in regions_gdf.index]

    lines = [
        f"{row.PART}|{row.minx},{row.miny}|{row.maxx},{row.maxy}|{zoom}"
        for row in regions_gdf.itertuples(index=False)
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines))
    print(f"[boundary] {len(lines)} tiles -> {output_path}")
    return len(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--labels", required=True,
                   help="Country labels pkl (GeoDataFrame with 'geometry').")
    p.add_argument("--output", required=True,
                   help="Output .txt manifest (one tile per line).")
    p.add_argument("--zoom", type=int, default=19,
                   help="Satellite imagery zoom level.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    country_to_tiles(args.labels, args.output, zoom=args.zoom)


if __name__ == "__main__":
    main()
