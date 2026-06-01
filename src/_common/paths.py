"""Image-path remapping helpers.

Some intermediate manifests (paths.pkl, feature h5 files) store absolute
paths from the machine where they were generated. When the same artifact
is consumed on a machine with a different filesystem layout (e.g. when
features are extracted on one server and labels merged on another), the
paths inside the file refer to non-existent locations.

This module centralizes the remap so the pipeline can be portable.
The original scripts inlined hardcoded substitutions like
``/home/huangyingjing/GoogleSV -> /data/GoogleSV`` — these become a
configurable mapping here.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd


# Default remappings preserved for backward compatibility with the
# original training-server manifests. New deployments should pass an
# explicit mapping instead.
DEFAULT_REMAPS: dict[str, str] = {
    "/home/huangyingjing/GoogleSV": "/data/GoogleSV",
    "/nas/huangyj/GoogleSV": "/data/GoogleSV",
}


def remap_image_paths(
    series: pd.Series,
    remaps: Mapping[str, str] | None = None,
) -> pd.Series:
    """Apply a chain of ``old_prefix -> new_prefix`` substitutions to a path Series.

    Substitutions are applied in iteration order. Passing ``None`` uses
    :data:`DEFAULT_REMAPS`.
    """
    if remaps is None:
        remaps = DEFAULT_REMAPS
    out = series
    for old, new in remaps.items():
        out = out.str.replace(old, new, regex=False)
    return out


def city_from_path(series: pd.Series, *, depth: int = 5) -> pd.Series:
    """Extract the city name from an image path by indexing into the path components.

    The original convention places the city name at depth 5 in paths like
    ``/data/GoogleSV/images/Country/City/<hash>/<file>.jpg``. ``depth`` is
    exposed so callers can override it for non-standard layouts.
    """
    return series.str.split("/").str[depth]
