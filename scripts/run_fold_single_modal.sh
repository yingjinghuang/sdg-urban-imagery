#!/usr/bin/env bash
# Single-modal (SV-only, RS-only) fold regression — manuscript Fig. 3a.
# Uses the SV-or-RS-level Concat_self_spatial.pkl features (NOT the Fuse-level)
# and the regression configuration reported in Supplementary Table 9.
#
# Output:
#   ${REGRESSION_OUT_DIR}/fold/main/{Country}/{City}/{SV|RS}/Token_Concat_self_spatial/...

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

FEATURE_NAME="Concat_self_spatial"
RUN_TAG="main"

for TYPE in SV RS; do
    list_regions | while read -r COUNTRY CITY _ _; do
        TARGETS="$(country_targets "${COUNTRY}")"
        [ -z "${TARGETS}" ] && continue

        FEATURE_PATH="${FEATURES_UNIT_DIR}/${COUNTRY}/${CITY}/${TYPE}/${FEATURE_NAME}.pkl"
        OUTPUT_DIR="${REGRESSION_OUT_DIR}/fold/${RUN_TAG}/${COUNTRY}/${CITY}/${TYPE}/Token_${FEATURE_NAME}"

        mkdir -p "${OUTPUT_DIR}"
        echo "[fold-single] ${COUNTRY}/${CITY}/${TYPE}"
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON}" "${REPO_ROOT}/src/regression/token_reg.py" \
            --feature_path "${FEATURE_PATH}" \
            --label_path   "${LABELS_DIR}/${COUNTRY}/${CITY}/labels_norm.pkl" \
            --community_path "${PROCESSED_DIR}/${COUNTRY}/${CITY}/samples/fold5.pkl" \
            --geo_path     "${PROCESSED_DIR}/${COUNTRY}/${CITY}/geo.pkl" \
            --targets      "${TARGETS}" \
            --output_dir   "${OUTPUT_DIR}" \
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
done
