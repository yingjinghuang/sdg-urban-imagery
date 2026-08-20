#!/usr/bin/env bash
# Fold regression with the segmentation baseline — manuscript Fig. 2a gray line.
# Uses SV-only 150-dim ADE20K class-distribution features and the same
# downstream regression configuration as the proposed model.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

TYPE="SV"
FEATURE_NAME="segmentation"
RUN_TAG="segmentation"
FEATURE_LEN=150

list_regions | while read -r COUNTRY CITY _ _; do
    TARGETS="$(country_targets "${COUNTRY}")"
    [ -z "${TARGETS}" ] && continue
    FEATURE_PATH="${FEATURES_UNIT_DIR}/${COUNTRY}/${CITY}/${TYPE}/${FEATURE_NAME}.pkl"
    if [ ! -f "${FEATURE_PATH}" ]; then
        echo "[fold-seg] SKIP ${COUNTRY}/${CITY} — run extract_features_segmentation.sh first"
        continue
    fi
    OUTPUT_DIR="${REGRESSION_OUT_DIR}/fold/${RUN_TAG}/${COUNTRY}/${CITY}/${TYPE}/Multi_${FEATURE_NAME}"
    mkdir -p "${OUTPUT_DIR}"
    echo "[fold-seg] ${COUNTRY}/${CITY}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON}" "${REPO_ROOT}/src/regression/multi_reg.py" \
        --feature_path "${FEATURE_PATH}" \
        --label_path   "${LABELS_DIR}/${COUNTRY}/${CITY}/labels_norm.pkl" \
        --community_path "${PROCESSED_DIR}/${COUNTRY}/${CITY}/samples/fold5.pkl" \
        --targets      "${TARGETS}" \
        --output_dir   "${OUTPUT_DIR}" \
        --feature_len  "${FEATURE_LEN}" \
        --hidden_dim 256 \
        --num_heads 8 \
        --num_layers 1 \
        --lr 1e-4 \
        --batch_size 128 \
        --num_epochs 100 \
        --warmup_epochs 10 \
        --patience 5 \
        --weight_decay 1e-4 \
        --mode         "${TYPE}" \
        --device 0
done
