#!/usr/bin/env bash
# Pretrain a ViT-B backbone on street-view imagery using either the
# self-contrastive or spatial-contrastive objective.
#
# Usage:
#   bash scripts/pretrain_sv.sh self    <city_tag>
#   bash scripts/pretrain_sv.sh spatial <city_tag> [init_ckpt]
#   bash scripts/pretrain_sv.sh <objective> <city_tag> [init_ckpt] [resume_ckpt]
#
# `init_ckpt` initializes model weights while starting a fresh optimizer and
# epoch counter. This is the appropriate mode when initializing a spatial-
# contrastive run from a self-contrastive checkpoint. `resume_ckpt` instead
# restores model, optimizer, scaler, and epoch for an interrupted same run.
#
# The corresponding training-set pickle must exist at
#   ${PROCESSED_DIR}/train_datasets/{self|spatial}_<city_tag>.pkl
#
# Manuscript pretraining settings:
#   architecture:         ViT-B
#   optimizer:            AdamW
#   total batch size:     1024
#   initial optimizer LR: 1e-5
#   weight decay:         1e-6
#   epochs:               100 for street-view imagery
#
# The MoCo-v3 training code uses a 10-epoch warm-up. The manuscript does not
# report a separate warm-up value, so the implementation default is retained
# explicitly here. The --lr value is used directly by the optimizer and is not
# rescaled by batch_size/256 in the cleaned pretraining entry point.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

OBJECTIVE="${1:?usage: pretrain_sv.sh <self|spatial> <city_tag> [init_ckpt] [resume_ckpt]}"
CITY_TAG="${2:?usage: pretrain_sv.sh <self|spatial> <city_tag> [init_ckpt] [resume_ckpt]}"
INIT_CKPT="${3:-}"
RESUME_CKPT="${4:-}"

if [ "${OBJECTIVE}" != "self" ] && [ "${OBJECTIVE}" != "spatial" ]; then
    echo "[pretrain-sv] objective must be self or spatial, got: ${OBJECTIVE}" >&2
    exit 2
fi
if [ -n "${INIT_CKPT}" ] && [ -n "${RESUME_CKPT}" ]; then
    echo "[pretrain-sv] use either init_ckpt or resume_ckpt, not both" >&2
    exit 2
fi

DATASET_PATH="${PROCESSED_DIR}/train_datasets/${OBJECTIVE}_${CITY_TAG}.pkl"
SAVE_DIR="${PRETRAIN_OUT_DIR}/Mocov3VITB-${OBJECTIVE}-${CITY_TAG}"
PORT=$((10000 + RANDOM % 9000))

if [ ! -f "${DATASET_PATH}" ]; then
    echo "[pretrain-sv] missing dataset: ${DATASET_PATH}" >&2
    exit 1
fi

mkdir -p "${SAVE_DIR}"

CHECKPOINT_ARG=()
if [ -n "${INIT_CKPT}" ]; then
    CHECKPOINT_ARG=(--pretrained "${INIT_CKPT}")
elif [ -n "${RESUME_CKPT}" ]; then
    CHECKPOINT_ARG=(--resume "${RESUME_CKPT}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON}" "${REPO_ROOT}/pretrain/moco_sv.py" \
    -a vit_base -b 1024 \
    --optimizer=adamw --lr=1e-5 --weight-decay=1e-6 \
    --save-folder "${SAVE_DIR}" \
    --epochs=100 --warmup-epochs=10 \
    --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
    --dist-url "tcp://localhost:${PORT}" \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    "${CHECKPOINT_ARG[@]}" \
    "${DATASET_PATH}"
