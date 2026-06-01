#!/usr/bin/env bash
# Fold regression with the ImageNet baseline — Fig 1a blue bar.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

TYPE="Fuse"
FEATURE_NAME="Concat_ImageNet"
RUN_TAG="imagenet"

list_regions | while read -r COUNTRY CITY _ _; do
    TARGETS="$(country_targets "${COUNTRY}")"
    [ -z "${TARGETS}" ] && continue
    FEATURE_PATH="${FEATURES_UNIT_DIR}/${COUNTRY}/${CITY}/${TYPE}/${FEATURE_NAME}.pkl"
    OUTPUT_DIR="${REGRESSION_OUT_DIR}/fold/${RUN_TAG}/${COUNTRY}/${CITY}/${TYPE}/Token_${FEATURE_NAME}"
    mkdir -p "${OUTPUT_DIR}"
    echo "[fold-imagenet] ${COUNTRY}/${CITY}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON}" "${REPO_ROOT}/src/regression/token_reg.py" \
        --feature_path "${FEATURE_PATH}" \
        --label_path   "${LABELS_DIR}/${COUNTRY}/${CITY}/labels_norm.pkl" \
        --community_path "${PROCESSED_DIR}/${COUNTRY}/${CITY}/samples/fold5.pkl" \
        --geo_path     "${PROCESSED_DIR}/${COUNTRY}/${CITY}/geo.pkl" \
        --targets      "${TARGETS}" \
        --output_dir   "${OUTPUT_DIR}" \
        --mode         "${TYPE}" \
        --device 0
done
