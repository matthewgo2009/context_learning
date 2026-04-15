#!/bin/bash
# =============================================================================
# 2×H100 single-process launch
#
# Solver → cuda:0,  Challenger → cuda:1  (handled by AdversarialTrainer)
# No torchrun / no DDP — one process sees both GPUs.
#
# Usage:
#   bash scripts/launch_2gpu.sh
#   bash scripts/launch_2gpu.sh --epochs 20 --group-size 4
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

echo "============================================================"
echo "Self-Evolving ICL — 2×H100 Training"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "============================================================"

python scripts/train_adversarial.py "$@"
