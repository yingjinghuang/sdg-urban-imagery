#!/usr/bin/env bash
# Score every street-view image against the 14 visual concepts used in
# Fig 2b–c (objects: building, car, fence, pole, window, road, tree;
# attributes: chaotic, orderly, depressing, lively, safe, dilapidated,
# wealthy). Uses OpenAI CLIP zero-shot scoring.
#
# Output: ${PROCESSED_DIR}/sv_clip_scores.pkl
#   columns: path, building, car, ..., wealthy

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

INPUT_PKL="${PROCESSED_DIR}/sv_paths_all.pkl"
OUTPUT_PKL="${PROCESSED_DIR}/sv_clip_scores.pkl"
mkdir -p "$(dirname "${OUTPUT_PKL}")"

if [ ! -f "${INPUT_PKL}" ]; then
    echo "[clip-scores] missing input manifest: ${INPUT_PKL}" >&2
    echo "[clip-scores] concat per-city paths.pkl into sv_paths_all.pkl first." >&2
    exit 1
fi

echo "[clip-scores] writing ${OUTPUT_PKL}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON}" -m src.analysis.clip_concept \
    --input_pkl  "${INPUT_PKL}" \
    --output_pkl "${OUTPUT_PKL}" \
    --batch_size 256
