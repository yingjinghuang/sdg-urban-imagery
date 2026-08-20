#!/usr/bin/env bash
# Non-spatial sampling baseline — manuscript Fig. 3f comparison curve.
# Uses the same four visual tokens as run_sampling.sh, but disables the
# geographic-coordinate token so the comparison isolates spatial context.
# Regression hyperparameters are pinned to Supplementary Table 9.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

TYPE="Fuse"
FEATURE_NAME="Concat_spatial_self"
RUN_TAG="main_no_geo"
FEATURE_LEN="${FEATURE_LEN:-768}"

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
        --hidden_dim 256 \
        --num_heads 8 \
        --num_layers 1 \
        --lr 1e-4 \
        --batch_size 128 \
        --num_epochs 100 \
        --warmup_epochs 10 \
        --patience 5 \
        --weight_decay 1e-4 \
        --no_geo \
        --device 0
done
