#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Stage 4: Optional Full / Hard Sample Fine-tuning
#
# Goal:
#   Further improve hard cases:
#     - small tumors
#     - weak edema boundaries
#     - ET/TC difficult slices
#     - Dice-low samples
#
# Recommended:
#   Use HARD_DATA_PATH if available.
#
# Default train mode:
#   head_plus_upscaling
#
# You may set:
#   MASK_DECODER_TRAIN_MODE=full
# but this is riskier.
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# -----------------------------
# Paths
# -----------------------------
FULL_DATA_PATH="${FULL_DATA_PATH:-${PROJECT_ROOT}/datasets/BraTS2023_reasonseg_rgbA}"
HARD_DATA_PATH="${HARD_DATA_PATH:-${PROJECT_ROOT}/datasets/BraTS2023_reasonseg_hard}"

if [[ -d "${HARD_DATA_PATH}" ]]; then
  export DATA_PATH="${DATA_PATH:-${HARD_DATA_PATH}}"
else
  echo "[WARN] HARD_DATA_PATH not found: ${HARD_DATA_PATH}"
  echo "[WARN] Falling back to FULL_DATA_PATH: ${FULL_DATA_PATH}"
  export DATA_PATH="${DATA_PATH:-${FULL_DATA_PATH}}"
fi

export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/stage4_hard_finetune}"

# Resume from Stage 3.
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-${PROJECT_ROOT}/outputs/stage3_partial_decoder/final_model}"
export RESUME_OPTIMIZER_STATES="${RESUME_OPTIMIZER_STATES:-0}"

# -----------------------------
# Hardware
# -----------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"

# -----------------------------
# Schedule
# Hard fine-tune should be short.
# -----------------------------
export EPOCHS="${EPOCHS:-2}"
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
export USE_LORA=1
export LORA_SCOPE="${LORA_SCOPE:-llm}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# -----------------------------
# Trainable modules
# -----------------------------
export TRAIN_SEG_PROJECTOR=1

# Safer than full. Override manually if needed:
#   MASK_DECODER_TRAIN_MODE=full bash scripts/stage4_hard_finetune.sh
export MASK_DECODER_TRAIN_MODE="${MASK_DECODER_TRAIN_MODE:-head_plus_upscaling}"

export TRAIN_QWEN_VISUAL_PROJECTOR="${TRAIN_QWEN_VISUAL_PROJECTOR:-0}"

# Fine-tune [SEG] hidden + mask branch jointly.
export DETACH_SEG_HIDDEN_FOR_MASK="${DETACH_SEG_HIDDEN_FOR_MASK:-0}"

# -----------------------------
# Learning rates
# -----------------------------
export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export SEG_PROJECTOR_LR="${SEG_PROJECTOR_LR:-2e-6}"

# Keep small. If full mode, do not increase aggressively.
export MASK_DECODER_LR="${MASK_DECODER_LR:-1e-9}"

# -----------------------------
# Loss weights
# More Dice-focused for hard cases.
# -----------------------------
export CE_LOSS_WEIGHT="${CE_LOSS_WEIGHT:-0.5}"
export DICE_LOSS_WEIGHT="${DICE_LOSS_WEIGHT:-2.0}"
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
echo "Stage 4: Hard Sample / Optional Fine-tuning"
echo "DATA_PATH=${DATA_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
echo "MASK_DECODER_TRAIN_MODE=${MASK_DECODER_TRAIN_MODE}"
echo "MASK_DECODER_LR=${MASK_DECODER_LR}"
echo "DETACH_SEG_HIDDEN_FOR_MASK=${DETACH_SEG_HIDDEN_FOR_MASK}"
echo "============================================================"

if [[ ! -d "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "[ERROR] Previous stage checkpoint not found: ${RESUME_FROM_CHECKPOINT}"
  echo "Run scripts/stage3_partial_decoder_adaptation.sh first, or set RESUME_FROM_CHECKPOINT manually."
  exit 1
fi

bash scripts/run_sft.sh