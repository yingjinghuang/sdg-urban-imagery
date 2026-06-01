#!/usr/bin/env bash
# Feature-guided sampling regression — main framework, with geo.
# Uses the PCA-99 reduced Concat_spatial_self feature and the hierarchical
# k-center sample-selection result. Feeds Fig 1b/c/d and Fig 2f/g.
#
# Output:
#   ${REGRESSION_OUT_DIR}/sampling/main/{Country}/{City}/Fuse/Multi_Concat_pcahierachy/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

TYPE="Fuse"
# Verified against the legacy Multi_Concat_pcahierachy/scaler.pkl: the
# regression that produced the published Fig 3 numbers was trained on
# Concat_spatial.pkl (1536-d). The bash placeholder "Concat.pkl" in the
# legacy repo was a runtime symlink to this file.
FEATURE_NAME="Concat_spatial"
RUN_TAG="main"
FEATURE_LEN="${FEATURE_LEN:-1536}"

list_regions | while read -r COUNTRY CITY _ _; do
    TARGETS="$(country_targets "${COUNTRY}")"
    [ -z "${TARGETS}" ] && continue
    FEATURE_PATH="${FEATURES_UNIT_DIR}/${COUNTRY}/${CITY}/${TYPE}/${FEATURE_NAME}.pkl"
    OUTPUT_DIR="${REGRESSION_OUT_DIR}/sampling/${RUN_TAG}/${COUNTRY}/${CITY}/${TYPE}/Multi_Concat_pcahierachy"
    mkdir -p "${OUTPUT_DIR}"
    echo "[sampling] ${COUNTRY}/${CITY}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON}" "${REPO_ROOT}/src/regression/multi_reg.py" \
        --feature_path "${FEATURE_PATH}" \
        --label_path   "${LABELS_DIR}/${COUNTRY}/${CITY}/labels_norm.pkl" \
        --community_path "${PROCESSED_DIR}/${COUNTRY}/${CITY}/samples/pcahierachy.pkl" \
        --geo_path     "${PROCESSED_DIR}/${COUNTRY}/${CITY}/geo.pkl" \
        --targets      "${TARGETS}" \
        --output_dir   "${OUTPUT_DIR}" \
        --feature_len  "${FEATURE_LEN}" \
        --device 0
done
