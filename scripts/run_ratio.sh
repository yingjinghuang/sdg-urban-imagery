#!/usr/bin/env bash
# Variable train-ratio regression on the main framework feature.
# Sweeps the training-set ratio from low to high to characterize how
# much ground-truth survey data is needed for reliable estimation.
# Feeds Fig 1b (SDG scaling), Fig 1c (city scaling), and Fig 1d (random baseline).
#
# Output:
#   ${REGRESSION_OUT_DIR}/ratio/main/{Country}/{City}/Fuse/Token_Concat_spatial_self/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

TYPE="Fuse"
FEATURE_NAME="Concat_spatial_self"
RUN_TAG="main"

list_regions | while read -r COUNTRY CITY _ _; do
    TARGETS="$(country_targets "${COUNTRY}")"
    [ -z "${TARGETS}" ] && continue
    FEATURE_PATH="${FEATURES_UNIT_DIR}/${COUNTRY}/${CITY}/${TYPE}/${FEATURE_NAME}.pkl"
    OUTPUT_DIR="${REGRESSION_OUT_DIR}/ratio/${RUN_TAG}/${COUNTRY}/${CITY}/${TYPE}/Token_${FEATURE_NAME}"
    mkdir -p "${OUTPUT_DIR}"
    echo "[ratio] ${COUNTRY}/${CITY}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON}" "${REPO_ROOT}/src/regression/token_reg.py" \
        --feature_path "${FEATURE_PATH}" \
        --label_path   "${LABELS_DIR}/${COUNTRY}/${CITY}/labels_norm.pkl" \
        --community_path "${PROCESSED_DIR}/${COUNTRY}/${CITY}/samples/ratio.pkl" \
        --geo_path     "${PROCESSED_DIR}/${COUNTRY}/${CITY}/geo.pkl" \
        --targets      "${TARGETS}" \
        --output_dir   "${OUTPUT_DIR}" \
        --mode         "${TYPE}" \
        --device 0
done
