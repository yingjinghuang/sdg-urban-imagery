#!/usr/bin/env bash
# 5-fold cross-validation regression on the FUSE-level main feature.
# Produces R² values that feed manuscript Fig. 2a (Ours) and Fig. 3a (fusion).
#
# Regression hyperparameters below are pinned to Supplementary Table 9.
#
# Output:
#   ${REGRESSION_OUT_DIR}/fold/main/{Country}/{City}/Fuse/{Token_Concat_spatial_self}/
#     ├── results.csv      ← per-target R²/MAE/MSE
#     ├── results.h5       ← per-neighborhood predictions
#     ├── S*.pth.tar       ← best checkpoint per split
#     ├── scaler.pkl
#     └── scaler_geo.pkl

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

TYPE="${TYPE:-Fuse}"
FEATURE_NAME="Concat_spatial_self"
RUN_TAG="main"

list_regions | while read -r COUNTRY CITY _ _; do
    TARGETS="$(country_targets "${COUNTRY}")"
    [ -z "${TARGETS}" ] && { echo "[fold] SKIP ${COUNTRY}/${CITY} — no targets"; continue; }

    FEATURE_PATH="${FEATURES_UNIT_DIR}/${COUNTRY}/${CITY}/${TYPE}/${FEATURE_NAME}.pkl"
    OUTPUT_DIR="${REGRESSION_OUT_DIR}/fold/${RUN_TAG}/${COUNTRY}/${CITY}/${TYPE}/Token_${FEATURE_NAME}"

    mkdir -p "${OUTPUT_DIR}"
    echo "[fold] ${COUNTRY}/${CITY}"
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
