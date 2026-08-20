#!/usr/bin/env bash
# Fold regression with the ImageNet-1K ResNet-50 baseline — manuscript Fig. 2a.
# Each modality contributes a 2,048-d global-average-pooling feature block.
# Uses the same downstream regression configuration as the proposed model,
# matching Supplementary Table 9.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

TYPE="Fuse"
FEATURE_NAME="Concat_ImageNet"
RUN_TAG="imagenet"
FEATURE_LEN=2048

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
        --feature_len "${FEATURE_LEN}" \
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
