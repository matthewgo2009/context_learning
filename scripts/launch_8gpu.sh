#!/bin/bash
# =============================================================================
# Self-Evolving ICL — Adversarial Self-Play Training
# 8×H100 80GB launch script
#
# Usage:
#   bash scripts/launch_8gpu.sh                          # default Qwen2.5-7B
#   bash scripts/launch_8gpu.sh --model Qwen/Qwen2.5-14B-Instruct  # larger model
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true  # set to false if you want W&B logging

# Judge LLM (paper: frozen evaluator, default GPT-4o)
# Set your API key here or in your shell profile:
#   export OPENAI_API_KEY="sk-..."
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "WARNING: OPENAI_API_KEY not set — Judge LLM will fall back to heuristic."
    echo "         Set it to use the frozen GPT-4o evaluator as described in the paper."
fi

NUM_GPUS=8
MASTER_PORT=${MASTER_PORT:-29500}

# ---------------------------------------------------------------------------
# Default training args (can be overridden by passing CLI args to this script)
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EPOCHS="${EPOCHS:-3}"
SOLVER_LR="${SOLVER_LR:-1e-5}"
CHALLENGER_LR="${CHALLENGER_LR:-5e-6}"
GROUP_SIZE="${GROUP_SIZE:-4}"
KL_BETA="${KL_BETA:-0.04}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/adversarial}"
SAVE_EVERY="${SAVE_EVERY:-500}"
REF_SYNC="${REF_SYNC:-200}"

echo "============================================================"
echo "Self-Evolving ICL Adversarial Training"
echo "  GPUs:            ${NUM_GPUS}"
echo "  Model:           ${MODEL}"
echo "  Epochs:          ${EPOCHS}"
echo "  Solver LR:       ${SOLVER_LR}"
echo "  Challenger LR:   ${CHALLENGER_LR}"
echo "  GRPO group size: ${GROUP_SIZE}"
echo "  KL beta:         ${KL_BETA}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Launch via DeepSpeed (recommended for ZeRO-2/3)
# ---------------------------------------------------------------------------
deepspeed \
    --num_gpus=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    scripts/train_adversarial.py \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --lr "$SOLVER_LR" \
    --challenger-lr "$CHALLENGER_LR" \
    --group-size "$GROUP_SIZE" \
    --kl-beta "$KL_BETA" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --save-every "$SAVE_EVERY" \
    --ref-sync-every "$REF_SYNC" \
    "$@"

echo "Training complete. Checkpoints saved to ${CHECKPOINT_DIR}"
