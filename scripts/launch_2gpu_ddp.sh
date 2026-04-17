#!/bin/bash
# =============================================================================
# 2×H100 DDP launch (torchrun, one process per GPU)
#
# Each rank holds a full Solver+Challenger pair on its own card (LoRA makes
# that fit comfortably under 80 GB). Gradients of LoRA params are averaged
# across ranks via manual `all_reduce` inside AdversarialTrainer — the
# models themselves are NOT wrapped in DistributedDataParallel, so
# generate() / disable_adapter() / peft paths stay intact.
#
# Data is sharded as samples[rank::world_size] inside the trainer.
#
# Usage:
#   bash scripts/launch_2gpu_ddp.sh
#   bash scripts/launch_2gpu_ddp.sh --epochs 20 --group-size 4
#   NUM_GPUS=4 bash scripts/launch_2gpu_ddp.sh --epochs 20   # for >2 cards
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

NUM_GPUS="${NUM_GPUS:-2}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

echo "============================================================"
echo "Self-Evolving ICL — ${NUM_GPUS}×GPU DDP Training"
echo "  CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo "  NUM_GPUS             : ${NUM_GPUS}"
echo "  MASTER_ADDR:PORT     : ${MASTER_ADDR}:${MASTER_PORT}"
echo "  PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "============================================================"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    scripts/train_adversarial.py "$@"
