#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Stage 1: Projector Warm-up
#
# Goal:
#   Learn [SEG] hidden -> MedSAM prompt space.
#
# Train:
#   - Qwen LLM LoRA
#   - ReaSegProjector
#
# Freeze:
#   - MedSAM image encoder
#   - MedSAM prompt encoder
#   - MedSAM mask decoder
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# -----------------------------
# Paths
# -----------------------------
export DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/datasets/BraTS2023_reasonseg_rgbA}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/stage1_projector_warmup}"

# -----------------------------
# Hardware
# -----------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"

# -----------------------------
# Schedule
# For 1300 samples: 10 epochs is okay.
# For 45k samples: use 1-2 epochs.
# -----------------------------
export EPOCHS="${EPOCHS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export WORKERS="${WORKERS:-4}"

# -----------------------------
# Precision / DeepSpeed
# -----------------------------
export PRECISION="${PRECISION:-bf16}"
export ZERO_STAGE="${ZERO_STAGE:-2}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/configs/deepspeed_zero2.json}"

# -----------------------------
# LoRA
# -----------------------------
export USE_LORA="${USE_LORA:-1}"
export LORA_SCOPE="${LORA_SCOPE:-llm}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# -----------------------------
# Trainable modules
# -----------------------------
export TRAIN_SEG_PROJECTOR=1
export MASK_DECODER_TRAIN_MODE=none
export TRAIN_QWEN_VISUAL_PROJECTOR=0

# Mask loss does not update Qwen LoRA in warm-up.
export DETACH_SEG_HIDDEN_FOR_MASK=1

# -----------------------------
# Learning rates
# -----------------------------
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export SEG_PROJECTOR_LR="${SEG_PROJECTOR_LR:-1e-5}"
export MASK_DECODER_LR="${MASK_DECODER_LR:-1e-8}"

# -----------------------------
# Loss weights
# Warm-up can use balanced Dice/BCE.
# -----------------------------
export CE_LOSS_WEIGHT="${CE_LOSS_WEIGHT:-1.0}"
export DICE_LOSS_WEIGHT="${DICE_LOSS_WEIGHT:-1.0}"
export BCE_LOSS_WEIGHT="${BCE_LOSS_WEIGHT:-1.0}"

# -----------------------------
# Logging / debug
# -----------------------------
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-0}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
export DEBUG_VL_ALIGNMENT="${DEBUG_VL_ALIGNMENT:-0}"
export DEBUG_NUMERICS="${DEBUG_NUMERICS:-1}"
export DEBUG_STEPS="${DEBUG_STEPS:-5}"

echo "============================================================"
echo "Stage 1: Projector Warm-up"
echo "DATA_PATH=${DATA_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MASK_DECODER_TRAIN_MODE=${MASK_DECODER_TRAIN_MODE}"
echo "DETACH_SEG_HIDDEN_FOR_MASK=${DETACH_SEG_HIDDEN_FOR_MASK}"
echo "============================================================"

bash scripts/run_sft.sh