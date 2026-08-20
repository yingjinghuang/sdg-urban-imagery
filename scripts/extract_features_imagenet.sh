#!/usr/bin/env bash
# Extract ImageNet-1K-pretrained ResNet-50 features (no contrastive finetuning)
# for the manuscript Fig. 2a ImageNet baseline. Runs for both SV and RS.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-2048}"
N_GPUS=$(echo "${CUDA_DEVICES}" | awk -F',' '{print NF}')
BATCH_SIZE=$((BATCH_SIZE_PER_GPU * N_GPUS))

for MODALITY in SV RS; do
    list_regions | while read -r COUNTRY CITY _ _; do
        UNIT_OUT="${FEATURES_UNIT_DIR}/${COUNTRY}/${CITY}/${MODALITY}/ImageNet.pkl"
        IMG_MANIFEST="${PROCESSED_DIR}/${COUNTRY}/${CITY}/paths.pkl"
        mkdir -p "$(dirname "${UNIT_OUT}")"
        echo "[extract-imagenet] ${COUNTRY}/${CITY}/${MODALITY}"
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON}" "${REPO_ROOT}/src/extract/extract_feature.py" \
            --batch_size "${BATCH_SIZE}" \
            --pretrained_model_path imagenet \
            --data_path "${IMG_MANIFEST}" \
            --save_path "${UNIT_OUT}.tmp.h5"
        "${PYTHON}" "${REPO_ROOT}/src/extract/aggregate_to_unit.py" \
            --feature_path "${UNIT_OUT}.tmp.h5" \
            --meta_path "${IMG_MANIFEST}" \
            --save_path "${UNIT_OUT}" \
            --arch ResNet50
        rm -f "${UNIT_OUT}.tmp.h5"
    done
done
