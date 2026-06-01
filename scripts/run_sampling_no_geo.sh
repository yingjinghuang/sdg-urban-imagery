#!/usr/bin/env bash
# Non-spatial sampling baseline — Fig 2f comparison curve.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

TYPE="Fuse"
# Matches run_sampling.sh; see comment there about the 1536-d Concat_spatial input.
FEATURE_NAME="Concat_spatial"
RUN_TAG="main_no_geo"
FEATURE_LEN="${FEATURE_LEN:-1536}"

list_regions | while read -r COUNTRY CITY _ _; do
    TARGETS="$(country_targets "${COUNTRY}")"
    [ -z "${TARGETS}" ] && continue
    FEATURE_PATH="${FEATURES_UNIT_DIR}/${COUNTRY}/${CITY}/${TYPE}/${FEATURE_NAME}.pkl"
    OUTPUT_DIR="${REGRESSION_OUT_DIR}/sampling/${RUN_TAG}/${COUNTRY}/${CITY}/${TYPE}/Multi_Concat_pcahierachy"
    mkdir -p "${OUTPUT_DIR}"
    echo "[sampling-no-geo] ${COUNTRY}/${CITY}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON}" "${REPO_ROOT}/src/regression/multi_reg.py" \
        --feature_path "${FEATURE_PATH}" \
        --label_path   "${LABELS_DIR}/${COUNTRY}/${CITY}/labels_norm.pkl" \
        --community_path "${PROCESSED_DIR}/${COUNTRY}/${CITY}/samples/pcahierachy.pkl" \
        --targets      "${TARGETS}" \
        --output_dir   "${OUTPUT_DIR}" \
        --feature_len  "${FEATURE_LEN}" \
        --no_geo \
        --device 0
done
