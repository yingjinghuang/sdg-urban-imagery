#!/usr/bin/env bash
# Aggregate per-image ADE20K segmentation distributions to neighborhood-level
# 150-dim feature vectors. Used for the Fig 1a segmentation baseline.
#
# Assumes raw per-image segmentation outputs already exist at
#   ${SEG_RESULTS_DIR}/${COUNTRY}/${CITY}.csv
# (One row per image, 150 ADE20K class columns plus the image path/GEOID.)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

list_regions | while read -r COUNTRY CITY _ _; do
    FEATURE_CSV="${SEG_RESULTS_DIR}/${COUNTRY}/${CITY}.csv"
    UNIT_OUT="${FEATURES_UNIT_DIR}/${COUNTRY}/${CITY}/SV/segmentation.pkl"
    if [ ! -f "${FEATURE_CSV}" ]; then
        echo "[extract-seg] SKIP ${COUNTRY}/${CITY} — missing ${FEATURE_CSV}"
        continue
    fi
    mkdir -p "$(dirname "${UNIT_OUT}")"
    echo "[extract-seg] ${COUNTRY}/${CITY}"
    "${PYTHON}" "${REPO_ROOT}/src/extract/aggregate_to_unit.py" \
        --feature_path "${FEATURE_CSV}" \
        --meta_path "${PROCESSED_DIR}/${COUNTRY}/${CITY}/paths.pkl" \
        --save_path "${UNIT_OUT}" \
        --arch segmentation
done
