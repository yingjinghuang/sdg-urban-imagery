#!/usr/bin/env bash
# Hierarchical k-center sample-selection step.
# Selects which neighborhoods to "survey" based on feature-space coverage.
# Produces ${PROCESSED_DIR}/{Country}/{City}/samples/pcahierachy.pkl
# which is consumed by run_sampling.sh.
#
# Selection uses the Concat_spatial_self_pca99 features.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

FEATURE_NAME="Concat_spatial_self_pca99"

list_regions | while read -r COUNTRY CITY _ _; do
    FEATURE_PATH="${FEATURES_UNIT_DIR}/${COUNTRY}/${CITY}/Fuse/${FEATURE_NAME}.pkl"
    OUTPUT_PATH="${PROCESSED_DIR}/${COUNTRY}/${CITY}/samples/pcahierachy.pkl"
    mkdir -p "$(dirname "${OUTPUT_PATH}")"
    echo "[select-kcenter] ${COUNTRY}/${CITY}"
    "${PYTHON}" "${REPO_ROOT}/src/regression/select_kcenter.py" \
        --feature_path "${FEATURE_PATH}" \
        --output_path "${OUTPUT_PATH}" \
        --strategy hierarchical
done
