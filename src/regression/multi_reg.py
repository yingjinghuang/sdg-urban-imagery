"""Compatibility entry point for multi-feature regression experiments.

This module is intentionally a thin alias to :mod:`token_reg`. It is kept
as a separate entry point because the existing sampling and baseline figure
notebooks expect output directories with the historical ``Multi_*`` naming.
The model architecture, optimization, early stopping, and metric computation
are therefore identical to ``token_reg``.

The concrete input representation is selected by each launcher. In the
current paper-reproduction scripts, ``run_sampling.sh`` and
``run_sampling_no_geo.sh`` use the 1536-dimensional ``Concat_spatial.pkl``
input together with the feature-guided sampling definitions, while the
segmentation baseline supplies its own 150-dimensional feature input.

Paper-reproduction launchers pass the regression hyperparameters reported in
Supplementary Table 9 explicitly; the same values are also the defaults in
``token_reg.py``.
"""

from __future__ import annotations

from src.regression.token_reg import main

if __name__ == "__main__":
    main()
