#!/usr/bin/env bash
# Pretrain a ViT-B backbone on satellite imagery using either the
# self-contrastive or spatial-contrastive objective.
#
# Usage:
#   bash scripts/pretrain_rs.sh self    <tag> [resume_ckpt]
#   bash scripts/pretrain_rs.sh spatial <tag> [resume_ckpt]
#
# Manuscript pretraining settings:
#   architecture:         ViT-B
#   optimizer:            AdamW
#   total batch size:     1024
#   initial optimizer LR: 1e-5
#   weight decay:         1e-6
#   epochs:               50 for satellite imagery
#
# The MoCo-v3 training code uses a 10-epoch warm-up. The manuscript does not
# report a separate warm-up value, so the implementation default is retained
# explicitly here. The --lr value is used directly by the optimizer and is not
# rescaled by batch_size/256 in the cleaned pretraining entry point.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

OBJECTIVE="${1:?usage: pretrain_rs.sh <self|spatial> <tag> [resume_ckpt]}"
TAG="${2:?usage: pretrain_rs.sh <self|spatial> <tag> [resume_ckpt]}"
RESUME_CKPT="${3:-}"

if [ "${OBJECTIVE}" != "self" ] && [ "${OBJECTIVE}" != "spatial" ]; then
    echo "[pretrain-rs] objective must be self or spatial, got: ${OBJECTIVE}" >&2
    exit 2
fi

DATASET_PATH="${PROCESSED_DIR}/train_datasets/${OBJECTIVE}_rs_${TAG}.pkl"
SAVE_DIR="${PRETRAIN_OUT_DIR}/Mocov3VITB-${OBJECTIVE}-rs-${TAG}"
PORT=$((10000 + RANDOM % 9000))

if [ ! -f "${DATASET_PATH}" ]; then
    echo "[pretrain-rs] missing dataset: ${DATASET_PATH}" >&2
    exit 1
fi

mkdir -p "${SAVE_DIR}"

RESUME_ARG=()
if [ -n "${RESUME_CKPT}" ]; then
    RESUME_ARG=(--resume "${RESUME_CKPT}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON}" "${REPO_ROOT}/pretrain/moco_rs.py" \
    -a vit_base -b 1024 \
    --optimizer=adamw --lr=1e-5 --weight-decay=1e-6 \
    --save-folder "${SAVE_DIR}" \
    --epochs=50 --warmup-epochs=10 \
    --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
    --dist-url "tcp://localhost:${PORT}" \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    "${RESUME_ARG[@]}" \
    "${DATASET_PATH}"
