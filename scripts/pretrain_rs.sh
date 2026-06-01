#!/usr/bin/env bash
# Pretrain a ViT-B backbone on satellite imagery (self- or spatial-contrastive).
#
# Usage:
#   bash scripts/pretrain_rs.sh self    <tag>
#   bash scripts/pretrain_rs.sh spatial <tag>

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

OBJECTIVE="${1:?usage: pretrain_rs.sh <self|spatial> <tag> [resume_ckpt]}"
TAG="${2:?usage: pretrain_rs.sh <self|spatial> <tag> [resume_ckpt]}"
RESUME_CKPT="${3:-}"

DATASET_PATH="${PROCESSED_DIR}/train_datasets/${OBJECTIVE}_rs_${TAG}.pkl"
SAVE_DIR="${PRETRAIN_OUT_DIR}/Mocov3VITB-${OBJECTIVE}-rs-${TAG}"
PORT=$((10000 + RANDOM % 9000))

if [ ! -f "${DATASET_PATH}" ]; then
    echo "[pretrain-rs] missing dataset: ${DATASET_PATH}" >&2
    exit 1
fi

mkdir -p "${SAVE_DIR}"

RESUME_ARG=""
if [ -n "${RESUME_CKPT}" ]; then
    RESUME_ARG="--pretrained ${RESUME_CKPT}"
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON}" "${REPO_ROOT}/pretrain/moco_rs.py" \
    -a vit_base -b 1024 \
    --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
    --save-folder "${SAVE_DIR}" \
    --epochs=300 --warmup-epochs=40 \
    --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
    --dist-url "tcp://localhost:${PORT}" \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    ${RESUME_ARG} \
    "${DATASET_PATH}"
