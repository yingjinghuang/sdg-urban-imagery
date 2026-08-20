"""Compatibility entry point for multi-feature regression experiments.

This module is intentionally a thin alias to :mod:`token_reg`. It is kept
as a separate entry point because the existing sampling and baseline figure
notebooks expect output directories with the historical ``Multi_*`` naming.
The model architecture, optimization, early stopping, and metric computation
are therefore identical to ``token_reg``.

For the paper-reproduction sampling experiments, ``run_sampling.sh`` and
``run_sampling_no_geo.sh`` use ``Concat_spatial_self.pkl``: four 768-dimensional
visual tokens (street-view spatial, satellite spatial, street-view self, and
satellite self). This matches the representation used by the main regression
and random-sampling experiments. The no-geo launcher disables only the
geographic-coordinate token.

Paper-reproduction launchers pass the regression hyperparameters reported in
Supplementary Table 9 explicitly; the same values are also the defaults in
``token_reg.py``.
"""

from __future__ import annotations

from src.regression.token_reg import main

if __name__ == "__main__":
    main()
