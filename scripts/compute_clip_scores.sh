#!/usr/bin/env bash
# Score Los Angeles imagery against the modality-specific CLIP concept sets
# reported in the manuscript/Supplementary Information.
#
# Usage:
#   bash scripts/compute_clip_scores.sh SV
#   bash scripts/compute_clip_scores.sh RS
#
# SV concepts (14 scored in the Supplementary analysis):
#   building, car, fence, pole, window, road, tree,
#   chaotic, orderly, depressing, lively, safe, dilapidated, wealthy
# RS concepts (4):
#   concrete, rooftop, vegetation, soil
#
# The final main Fig. 3b-c displays four concepts per modality:
#   RS: concrete, rooftop, vegetation, soil
#   SV: building, window, road, tree

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

MODALITY="${1:?usage: compute_clip_scores.sh <SV|RS>}"
MODALITY="${MODALITY^^}"

case "${MODALITY}" in
    SV)
        INPUT_PKL="${PROCESSED_DIR}/US/LosAngeles/paths.pkl"
        OUTPUT_PKL="${PROCESSED_DIR}/sv_clip_scores.pkl"
        ;;
    RS)
        INPUT_PKL="${PROCESSED_DIR}/US/LosAngeles/rs_paths.pkl"
        OUTPUT_PKL="${PROCESSED_DIR}/rs_clip_scores.pkl"
        ;;
    *)
        echo "[clip-scores] modality must be SV or RS, got: ${MODALITY}" >&2
        exit 2
        ;;
esac

if [ ! -f "${INPUT_PKL}" ]; then
    echo "[clip-scores] missing Los Angeles ${MODALITY} manifest: ${INPUT_PKL}" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PKL}")"
echo "[clip-scores] ${MODALITY}: ${INPUT_PKL} -> ${OUTPUT_PKL}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON}" -m src.analysis.clip_concept \
    --modality "${MODALITY}" \
    --input-pkl "${INPUT_PKL}" \
    --output-pkl "${OUTPUT_PKL}" \
    --batch-size 256
