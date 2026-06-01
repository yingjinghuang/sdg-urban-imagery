#!/usr/bin/env bash
# Build all Concat_*.pkl fused feature files for every region.
#
# Produces, for each (Country, City):
#   Fuse/Concat_self.pkl              self-SV  ⊕ self-RS
#   Fuse/Concat_spatial.pkl           spatial-SV ⊕ spatial-RS
#   Fuse/Concat_spatial_self.pkl      (self+spatial)-SV ⊕ (self+spatial)-RS  [MAIN]
#   Fuse/Concat_ImageNet.pkl          ImageNet-SV ⊕ ImageNet-RS
#   Fuse/Concat_spatial_self_pca99.pkl  PCA-99% of Concat_spatial_self  [for Fig 3]
#   SV/Concat_self_spatial.pkl        self-SV ⊕ spatial-SV  [for Fig 2a single-modal]
#   RS/Concat_self_spatial.pkl        self-RS ⊕ spatial-RS

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

    "${PYTHON}" "${REPO_ROOT}/src/fuse/pca_reduce.py" \
        --input "${FEATURES_UNIT_DIR}/${COUNTRY}/${CITY}/Fuse/Concat_spatial_self.pkl" \
        --output "${FEATURES_UNIT_DIR}/${COUNTRY}/${CITY}/Fuse/Concat_spatial_self_pca99.pkl" \
        --variance 0.99
done
