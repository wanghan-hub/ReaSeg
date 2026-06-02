#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ReaSeg SFT launch script
#
# Clean training flow:
#   Qwen2.5-VL
#     -> [SEG] hidden state
#     -> ReaSegProjector
#     -> MedSAM mask decoder
#     -> CE + Dice + BCE loss
# ============================================================

# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# -----------------------------
# Environment
# -----------------------------
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Optional NCCL settings. Usually safe on a single node.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"

# -----------------------------
# Basic paths
# -----------------------------
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${PROJECT_ROOT}/checkpoints/Qwen/Qwen2.5-VL-3B-Instruct}"
MEDSAM_CHECKPOINT="${MEDSAM_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/medsam_vit_b.pth}"

# IMPORTANT:
# DATA_PATH should be either:
#   1) a directory containing train.json and optionally val.json
#   2) a single json file
#
# Example:
#   DATA_PATH=/data/ReaSeg/pathchatseg-r1-main/reaseg/dataset
#   DATA_PATH=/data/ReaSeg/pathchatseg-r1-main/reaseg/dataset/train.json
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/dataset}"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/reaseg_sft_debug}"

# -----------------------------
# Hardware
# -----------------------------
NUM_GPUS="${NUM_GPUS:-1}"

# If you want to manually choose GPUs:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 bash scripts/run_sft.sh
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

# -----------------------------
# Model / data
# -----------------------------
SEG_TOKEN="${SEG_TOKEN:-[SEG]}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
PROMPT_DIM="${PROMPT_DIM:-256}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"

# -----------------------------
# Precision
# -----------------------------
# Recommended first:
#   bf16
# If debugging dtype issues:
#   fp32
PRECISION="${PRECISION:-bf16}"

# -----------------------------
# Training schedule
# -----------------------------
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-42}"

# -----------------------------
# Learning rates
# -----------------------------
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SEG_PROJECTOR_LR="${SEG_PROJECTOR_LR:-1e-5}"

# Keep mask decoder LR very small when enabling partial decoder.
MASK_DECODER_LR="${MASK_DECODER_LR:-1e-8}"

WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
ADAM_EPS="${ADAM_EPS:-1e-6}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.3}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"

# -----------------------------
# Loss weights
# -----------------------------
CE_LOSS_WEIGHT="${CE_LOSS_WEIGHT:-1.0}"
DICE_LOSS_WEIGHT="${DICE_LOSS_WEIGHT:-1.0}"
BCE_LOSS_WEIGHT="${BCE_LOSS_WEIGHT:-1.0}"

# -----------------------------
# Projector
# -----------------------------
PROJECTOR_INIT_STD="${PROJECTOR_INIT_STD:-1e-3}"
PROJECTOR_OUTPUT_SCALE="${PROJECTOR_OUTPUT_SCALE:-1.0}"
PROJECTOR_CLAMP_VALUE="${PROJECTOR_CLAMP_VALUE:-10.0}"

# -----------------------------
# LoRA
# -----------------------------
USE_LORA="${USE_LORA:-1}"
LORA_SCOPE="${LORA_SCOPE:-llm}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Only used when LORA_SCOPE=custom.
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-}"

# -----------------------------
# Trainable modules
# -----------------------------
TRAIN_SEG_PROJECTOR="${TRAIN_SEG_PROJECTOR:-1}"

# Recommended first-stage setting:
#   none
#
# After the frozen forward path is stable, try:
#   partial
#
# Supported:
#   none
#   partial
#   head_plus_upscaling
#   full
MASK_DECODER_TRAIN_MODE="${MASK_DECODER_TRAIN_MODE:-none}"

TRAIN_QWEN_VISUAL_PROJECTOR="${TRAIN_QWEN_VISUAL_PROJECTOR:-0}"

# Recommended first-stage setting:
#   1
# This means mask loss does not update Qwen LoRA.
DETACH_SEG_HIDDEN_FOR_MASK="${DETACH_SEG_HIDDEN_FOR_MASK:-1}"

RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
RESUME_OPTIMIZER_STATES="${RESUME_OPTIMIZER_STATES:-0}"

# -----------------------------
# DeepSpeed
# -----------------------------
ZERO_STAGE="${ZERO_STAGE:-2}"

# Optional external DeepSpeed config.
# Leave empty to use the generated config from train_utils.py.
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/configs/deepspeed_zero2.json}"

# -----------------------------
# Logging / eval / save
# -----------------------------
LOG_INTERVAL="${LOG_INTERVAL:-10}"

# 0 disables validation.
EVAL_INTERVAL="${EVAL_INTERVAL:-0}"

# 0 disables periodic checkpoint saving.
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"

MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-20}"

# -----------------------------
# Debug
# -----------------------------
DEBUG_VL_ALIGNMENT="${DEBUG_VL_ALIGNMENT:-1}"
DEBUG_NUMERICS="${DEBUG_NUMERICS:-1}"
DEBUG_STEPS="${DEBUG_STEPS:-5}"
SKIP_NONFINITE_LOSS="${SKIP_NONFINITE_LOSS:-0}"

# ============================================================
# Build command
# ============================================================

CMD=(
  deepspeed
  --num_gpus="${NUM_GPUS}"
  train/train_sft.py

  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --medsam_checkpoint "${MEDSAM_CHECKPOINT}"
  --data_path "${DATA_PATH}"
  --output_dir "${OUTPUT_DIR}"

  --seg_token "${SEG_TOKEN}"
  --image_size "${IMAGE_SIZE}"
  --prompt_dim "${PROMPT_DIM}"
  --max_seq_length "${MAX_SEQ_LENGTH}"

  --precision "${PRECISION}"
  --zero_stage "${ZERO_STAGE}"

  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
  --workers "${WORKERS}"
  --seed "${SEED}"

  --learning_rate "${LEARNING_RATE}"
  --seg_projector_lr "${SEG_PROJECTOR_LR}"
  --mask_decoder_lr "${MASK_DECODER_LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --adam_eps "${ADAM_EPS}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --warmup_steps "${WARMUP_STEPS}"

  --ce_loss_weight "${CE_LOSS_WEIGHT}"
  --dice_loss_weight "${DICE_LOSS_WEIGHT}"
  --bce_loss_weight "${BCE_LOSS_WEIGHT}"

  --projector_init_std "${PROJECTOR_INIT_STD}"
  --projector_output_scale "${PROJECTOR_OUTPUT_SCALE}"
  --projector_clamp_value "${PROJECTOR_CLAMP_VALUE}"

  --lora_scope "${LORA_SCOPE}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"

  --mask_decoder_train_mode "${MASK_DECODER_TRAIN_MODE}"

  --log_interval "${LOG_INTERVAL}"
  --eval_interval "${EVAL_INTERVAL}"
  --save_interval "${SAVE_INTERVAL}"
  --max_val_batches "${MAX_VAL_BATCHES}"
)

# Optional external DeepSpeed config.
if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  CMD+=(--deepspeed_config "${DEEPSPEED_CONFIG}")
fi

# LoRA switch.
if [[ "${USE_LORA}" == "1" ]]; then
  CMD+=(--use_lora)
else
  CMD+=(--no-use_lora)
fi

# Custom LoRA target modules.
if [[ -n "${LORA_TARGET_MODULES}" ]]; then
  CMD+=(--lora_target_modules "${LORA_TARGET_MODULES}")
fi

# Train SEG projector switch.
if [[ "${TRAIN_SEG_PROJECTOR}" == "1" ]]; then
  CMD+=(--train_seg_projector)
else
  CMD+=(--no-train_seg_projector)
fi

# Train Qwen visual projector switch.
if [[ "${TRAIN_QWEN_VISUAL_PROJECTOR}" == "1" ]]; then
  CMD+=(--train_qwen_visual_projector)
else
  CMD+=(--no-train_qwen_visual_projector)
fi

# Detach [SEG] hidden state switch.
if [[ "${DETACH_SEG_HIDDEN_FOR_MASK}" == "1" ]]; then
  CMD+=(--detach_seg_hidden_for_mask)
else
  CMD+=(--no-detach_seg_hidden_for_mask)
fi

# Debug switches.
if [[ "${DEBUG_VL_ALIGNMENT}" == "1" ]]; then
  CMD+=(--debug_vl_alignment)
fi

if [[ "${DEBUG_NUMERICS}" == "1" ]]; then
  CMD+=(--debug_numerics --debug_steps "${DEBUG_STEPS}")
fi

if [[ "${SKIP_NONFINITE_LOSS}" == "1" ]]; then
  CMD+=(--skip_nonfinite_loss)
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  CMD+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [[ "${RESUME_OPTIMIZER_STATES}" == "1" ]]; then
  CMD+=(--resume_optimizer_states)
else
  CMD+=(--no-resume_optimizer_states)
fi

# ============================================================
# Print config
# ============================================================

echo "============================================================"
echo "ReaSeg SFT"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "MEDSAM_CHECKPOINT=${MEDSAM_CHECKPOINT}"
echo "DATA_PATH=${DATA_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "PRECISION=${PRECISION}"
echo "MASK_DECODER_TRAIN_MODE=${MASK_DECODER_TRAIN_MODE}"
echo "USE_LORA=${USE_LORA}"
echo "LORA_SCOPE=${LORA_SCOPE}"
echo "TRAIN_SEG_PROJECTOR=${TRAIN_SEG_PROJECTOR}"
echo "DETACH_SEG_HIDDEN_FOR_MASK=${DETACH_SEG_HIDDEN_FOR_MASK}"
echo "============================================================"

printf 'Command:\n'
printf '%q ' "${CMD[@]}"
printf '\n'
echo "============================================================"

"${CMD[@]}"