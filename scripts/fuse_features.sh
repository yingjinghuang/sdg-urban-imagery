#!/usr/bin/env bash
# Build all Concat_*.pkl fused feature files for every region.
#
# Produces, for each (Country, City):
#   Fuse/Concat_self.pkl                 self-SV ⊕ self-RS
#   Fuse/Concat_spatial.pkl              spatial-SV ⊕ spatial-RS
#   Fuse/Concat_spatial_self.pkl         four 768-d branches, 3072-d total [MAIN]
#   Fuse/Concat_ImageNet.pkl             ImageNet-SV ⊕ ImageNet-RS
#   Fuse/Concat_spatial_self_pca99.pkl   optional global-PCA derivative
#   SV/Concat_self_spatial.pkl           self-SV ⊕ spatial-SV [single-modal]
#   RS/Concat_self_spatial.pkl           self-RS ⊕ spatial-RS
#
# Important for paper reproduction: feature-guided sampling does NOT use the
# global-PCA derivative above. scripts/select_kcenter.sh reads the unreduced
# Concat_spatial_self.pkl and applies PCA separately to each 768-d branch, as
# described in the manuscript Methods.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

list_regions | while read -r COUNTRY CITY SV_EPOCH RS_EPOCH; do
    echo "[fuse] ${COUNTRY}/${CITY}"

    "${PYTHON}" "${REPO_ROOT}/src/fuse/concat_cross_modal.py" \
        --root "${FEATURES_UNIT_DIR}" \
        --country "${COUNTRY}" --city "${CITY}" \
        --sv_epoch "${SV_EPOCH}" --rs_epoch "${RS_EPOCH}"

    "${PYTHON}" "${REPO_ROOT}/src/fuse/concat_within_modal.py" \
        --root "${FEATURES_UNIT_DIR}" \
        --country "${COUNTRY}" --city "${CITY}" \
        --sv_epoch "${SV_EPOCH}" --rs_epoch "${RS_EPOCH}"

    # Optional global-PCA artifact retained for compatibility with legacy
    # analysis. It is not consumed by the paper's k-center sampling pipeline.
    "${PYTHON}" "${REPO_ROOT}/src/fuse/pca_reduce.py" \
        --input "${FEATURES_UNIT_DIR}/${COUNTRY}/${CITY}/Fuse/Concat_spatial_self.pkl" \
        --output "${FEATURES_UNIT_DIR}/${COUNTRY}/${CITY}/Fuse/Concat_spatial_self_pca99.pkl" \
        --variance 0.99
done
