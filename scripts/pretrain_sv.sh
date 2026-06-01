#!/usr/bin/env bash
# Pretrain a ViT-B backbone on street-view imagery using either the
# self-contrastive or spatial-contrastive objective.
#
# Usage:
#   bash scripts/pretrain_sv.sh self      <city_tag>   # self-contrastive
#   bash scripts/pretrain_sv.sh spatial   <city_tag>   # spatial-contrastive (this paper)
#
# <city_tag> identifies the training dataset, e.g. "cities100-1m" for the
# global self-contrastive pretraining, or a single city tag for fine-tuning.
# The corresponding training-set pickle must exist at
#   ${PROCESSED_DIR}/train_datasets/{self|spatial}_<city_tag>.pkl

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_lib.sh"

OBJECTIVE="${1:?usage: pretrain_sv.sh <self|spatial> <city_tag> [resume_ckpt]}"
CITY_TAG="${2:?usage: pretrain_sv.sh <self|spatial> <city_tag> [resume_ckpt]}"
RESUME_CKPT="${3:-}"

DATASET_PATH="${PROCESSED_DIR}/train_datasets/${OBJECTIVE}_${CITY_TAG}.pkl"
SAVE_DIR="${PRETRAIN_OUT_DIR}/Mocov3VITB-${OBJECTIVE}-${CITY_TAG}"
PORT=$((10000 + RANDOM % 9000))

if [ ! -f "${DATASET_PATH}" ]; then
    echo "[pretrain-sv] missing dataset: ${DATASET_PATH}" >&2
    exit 1
fi

mkdir -p "${SAVE_DIR}"

RESUME_ARG=""
if [ -n "${RESUME_CKPT}" ]; then
    RESUME_ARG="--pretrained ${RESUME_CKPT}"
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON}" "${REPO_ROOT}/pretrain/moco_sv.py" \
    -a vit_base -b 1024 \
    --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
    --save-folder "${SAVE_DIR}" \
    --epochs=300 --warmup-epochs=40 \
    --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
    --dist-url "tcp://localhost:${PORT}" \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    ${RESUME_ARG} \
    "${DATASET_PATH}"
