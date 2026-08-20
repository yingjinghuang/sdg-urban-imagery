#!/usr/bin/env bash
# Hierarchical k-center sample-selection step.
# Selects which neighborhoods to "survey" based on joint visual/geographic
# feature-space coverage. Produces
#   ${PROCESSED_DIR}/{Country}/{City}/samples/pcahierachy.pkl
# which is consumed by run_sampling.sh and run_sampling_no_geo.sh.
#
# Paper configuration:
#   - start from the unreduced 3072-d Concat_spatial_self representation
#   - treat it as four 768-d branches (SV-spatial, RS-spatial, SV-self, RS-self)
#   - standardize and apply PCA separately to each branch, retaining 99% variance
#   - concatenate the reduced branches with standardized centroid coordinates
#     weighted by 0.5 * sqrt(d_v)
#   - run hierarchical k-center selection on the resulting joint representation
#
# The branch-wise PCA is performed inside src/regression/select_kcenter.py.
# Do NOT pass Concat_spatial_self_pca99.pkl here; that would apply a global PCA
# before the branch-wise PCA described in the manuscript.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

FEATURE_NAME="Concat_spatial_self"

list_regions | while read -r COUNTRY CITY _ _; do
    FEATURE_PATH="${FEATURES_UNIT_DIR}/${COUNTRY}/${CITY}/Fuse/${FEATURE_NAME}.pkl"
    GEO_PATH="${PROCESSED_DIR}/${COUNTRY}/${CITY}/geo.pkl"
    OUTPUT_PATH="${PROCESSED_DIR}/${COUNTRY}/${CITY}/samples/pcahierachy.pkl"
    mkdir -p "$(dirname "${OUTPUT_PATH}")"
    echo "[select-kcenter] ${COUNTRY}/${CITY}"
    "${PYTHON}" "${REPO_ROOT}/src/regression/select_kcenter.py" \
        --feature_path "${FEATURE_PATH}" \
        --geo_path "${GEO_PATH}" \
        --output_path "${OUTPUT_PATH}" \
        --block_size 768 \
        --variance 0.99 \
        --weight_factor 0.5 \
        --seed 42 \
        --strategy hierarchical
done
