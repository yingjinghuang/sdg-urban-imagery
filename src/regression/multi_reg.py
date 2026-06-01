"""Multi-feature regression entry point — Fig 3 sampling experiments.

Functionally a thin alias to :mod:`token_reg`, kept as a separate entry
point so the output-path naming used by the figure-prep notebooks
(``Multi_Concat_pcahierachy/results.csv``) remains stable. The model
and training loop are identical to ``token_reg``; the distinguishing
feature is that this entry is called with PCA-reduced inputs
(``Concat_spatial_self_pca99.pkl``) and a sampling-derived community
file (``samples/pcahierachy.pkl``).

If you are starting fresh, just use ``token_reg`` and choose your own
output naming. This alias exists for compatibility with the existing
figure notebooks.
"""

from __future__ import annotations

from src.regression.token_reg import main

if __name__ == "__main__":
    main()
