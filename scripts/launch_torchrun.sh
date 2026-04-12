#!/bin/bash
# =============================================================================
# Alternative launch via torchrun (no DeepSpeed, pure PyTorch DDP)
# Suitable for quick testing; for production use launch_8gpu.sh (DeepSpeed)
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    scripts/train_adversarial.py \
    "$@"
