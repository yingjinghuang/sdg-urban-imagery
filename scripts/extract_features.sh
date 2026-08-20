#!/usr/bin/env bash
# Extract per-image features with a pretrained Mocov3 backbone, for one
# modality (SV or RS) across all 20 regions. Then aggregate to neighborhood.
#
# Usage:
#   bash scripts/extract_features.sh SV spatial      # street view, spatial-contrastive
#   bash scripts/extract_features.sh RS self         # satellite, self-contrastive
#
# The script expects pretrained checkpoints at:
#   ${MOCOV3_CKPT_DIR}/Mocov3VITB-${VARIANT}-cbg-${CITY}/checkpoint_${EPOCH}.pth.tar
# (SV uses EPOCH=99, RS uses EPOCH=49 by convention; see cities.yaml.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

MODALITY="${1:?usage: extract_features.sh <SV|RS> <self|spatial>}"
VARIANT="${2:?usage: extract_features.sh <SV|RS> <self|spatial>}"

if [ "${MODALITY}" != "SV" ] && [ "${MODALITY}" != "RS" ]; then
    echo "[extract] modality must be SV or RS, got: ${MODALITY}" >&2
    exit 2
fi
if [ "${VARIANT}" != "self" ] && [ "${VARIANT}" != "spatial" ]; then
    echo "[extract] variant must be self or spatial, got: ${VARIANT}" >&2
    exit 2
fi

BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-2048}"
N_GPUS=$(echo "${CUDA_DEVICES}" | awk -F',' '{print NF}')
BATCH_SIZE=$((BATCH_SIZE_PER_GPU * N_GPUS))

list_regions | while read -r COUNTRY CITY SV_EPOCH RS_EPOCH; do
    if [ "${MODALITY}" = "SV" ]; then
        EPOCH="${SV_EPOCH}"
        IMG_MANIFEST="${PROCESSED_DIR}/${COUNTRY}/${CITY}/paths.pkl"
    else
        EPOCH="${RS_EPOCH}"
        IMG_MANIFEST="${PROCESSED_DIR}/${COUNTRY}/${CITY}/rs_paths.pkl"
    fi

    MODEL_NAME="Mocov3VITB-${VARIANT}-${COUNTRY}-${CITY}-ep${EPOCH}"
    CKPT="${MOCOV3_CKPT_DIR}/Mocov3VITB-${VARIANT}-cbg-${CITY}/checkpoint_${EPOCH}.pth.tar"

    RAW_OUT="${FEATURES_RAW_DIR}/${COUNTRY}/${CITY}/${MODALITY}/${MODEL_NAME}.h5"
    UNIT_OUT="${FEATURES_UNIT_DIR}/${COUNTRY}/${CITY}/${MODALITY}/${MODEL_NAME}.pkl"

    mkdir -p "$(dirname "${RAW_OUT}")" "$(dirname "${UNIT_OUT}")"

    if [ ! -f "${IMG_MANIFEST}" ]; then
        echo "[extract] SKIP ${COUNTRY}/${CITY}/${MODALITY} — missing manifest ${IMG_MANIFEST}" >&2
        continue
    fi

    echo "[extract] ${COUNTRY}/${CITY}/${MODALITY}/${VARIANT} ep${EPOCH}"

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON}" "${REPO_ROOT}/src/extract/extract_feature.py" \
        --batch_size "${BATCH_SIZE}" \
        --pretrained_model_path "${CKPT}" \
        --data_path "${IMG_MANIFEST}" \
        --save_path "${RAW_OUT}"

    # Aggregate with the same modality-specific manifest used for extraction.
    # In particular, RS features must be joined against rs_paths.pkl rather
    # than the street-view paths.pkl manifest.
    "${PYTHON}" "${REPO_ROOT}/src/extract/aggregate_to_unit.py" \
        --feature_path "${RAW_OUT}" \
        --meta_path "${IMG_MANIFEST}" \
        --save_path "${UNIT_OUT}" \
        --arch VITB
done
