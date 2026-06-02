#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Test final ReaSeg model
#
# Steps:
#   1. Find an existing merged fp32 checkpoint if available.
#   2. If not found, merge DeepSpeed ZeRO checkpoint into:
#        outputs/<stage>/merged_fp32_model/
#   3. Run tools/test_reaseg.py on test split.
#
# Supports:
#   - single-file checkpoint
#   - HuggingFace sharded checkpoint: *.index.json + shard .bin files
#
# Default checkpoint:
#   outputs/stage4_hard_finetune/final_model
#
# If Stage 4 was not used, override:
#   FINAL_CKPT_DIR=./outputs/stage3_partial_decoder/final_model \
#   bash scripts/test_final_model.sh
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

# -----------------------------
# Paths
# -----------------------------
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${PROJECT_ROOT}/checkpoints/Qwen/Qwen2.5-VL-3B-Instruct}"
export MEDSAM_CHECKPOINT="${MEDSAM_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/medsam_vit_b.pth}"
export DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/datasets/BraTS2023_reasonseg_rgbA}"

# Default to Stage 4 final model. Override if you stop at Stage 3.
FINAL_CKPT_DIR="${FINAL_CKPT_DIR:-${PROJECT_ROOT}/outputs/stage4_hard_finetune/final_model}"

if [[ ! -d "${FINAL_CKPT_DIR}" ]]; then
  echo "[WARN] FINAL_CKPT_DIR not found: ${FINAL_CKPT_DIR}"
  echo "[WARN] Falling back to Stage 3 final checkpoint."
  FINAL_CKPT_DIR="${PROJECT_ROOT}/outputs/stage3_partial_decoder/final_model"
fi

if [[ ! -d "${FINAL_CKPT_DIR}" ]]; then
  echo "[ERROR] No final checkpoint directory found."
  echo "Expected one of:"
  echo "  ${PROJECT_ROOT}/outputs/stage4_hard_finetune/final_model"
  echo "  ${PROJECT_ROOT}/outputs/stage3_partial_decoder/final_model"
  exit 1
fi

STAGE_OUTPUT_DIR="$(dirname "${FINAL_CKPT_DIR}")"

# Do NOT name this directory pytorch_model.bin.
# That name looks like a file but may actually become a folder.
MERGED_FP32_DIR="${MERGED_FP32_DIR:-${STAGE_OUTPUT_DIR}/merged_fp32_model}"

# Put test outputs beside final_model and merged_fp32_model.
TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-${STAGE_OUTPUT_DIR}/test_eval}"

# -----------------------------
# Test settings
# -----------------------------
SPLIT="${SPLIT:-test}"
PRECISION="${PRECISION:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WORKERS="${WORKERS:-2}"
LOGIT_THRESHOLD="${LOGIT_THRESHOLD:-0.0}"

# LoRA config must match training.
USE_LORA="${USE_LORA:-1}"
LORA_SCOPE="${LORA_SCOPE:-llm}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Optional: limit test samples for quick debugging.
LIMIT="${LIMIT:--1}"

# -----------------------------
# Helper: find state dict
# -----------------------------
find_state_dict() {
  local search_dir="$1"

  if [[ ! -d "${search_dir}" ]]; then
    return 1
  fi

  # 1. Exact common HF sharded checkpoint in current directory.
  if [[ -f "${search_dir}/pytorch_model.bin.index.json" ]]; then
    echo "${search_dir}/pytorch_model.bin.index.json"
    return 0
  fi

  # 2. Exact common single-file checkpoint in current directory.
  if [[ -f "${search_dir}/pytorch_model.bin" ]]; then
    echo "${search_dir}/pytorch_model.bin"
    return 0
  fi

  # 3. Common zero_to_fp32 output:
  #    search_dir/reaseg_fp32_model/pytorch_model.bin.index.json
  if [[ -f "${search_dir}/reaseg_fp32_model/pytorch_model.bin.index.json" ]]; then
    echo "${search_dir}/reaseg_fp32_model/pytorch_model.bin.index.json"
    return 0
  fi

  if [[ -f "${search_dir}/reaseg_fp32_model/pytorch_model.bin" ]]; then
    echo "${search_dir}/reaseg_fp32_model/pytorch_model.bin"
    return 0
  fi

  # 4. Old mistaken layout:
  #    search_dir/pytorch_model.bin/pytorch_model.bin.index.json
  if [[ -f "${search_dir}/pytorch_model.bin/pytorch_model.bin.index.json" ]]; then
    echo "${search_dir}/pytorch_model.bin/pytorch_model.bin.index.json"
    return 0
  fi

  # 5. Generic recursive search within depth 2.
  local index_file
  index_file="$(find "${search_dir}" -maxdepth 2 -type f -name "pytorch_model.bin.index.json" | head -n 1 || true)"
  if [[ -n "${index_file}" && -f "${index_file}" ]]; then
    echo "${index_file}"
    return 0
  fi

  # 6. Any HF index json.
  index_file="$(find "${search_dir}" -maxdepth 2 -type f -name "*.index.json" | head -n 1 || true)"
  if [[ -n "${index_file}" && -f "${index_file}" ]]; then
    echo "${index_file}"
    return 0
  fi

  # 7. Generic single-file checkpoint within depth 2.
  local bin_file
  bin_file="$(find "${search_dir}" -maxdepth 2 -type f -name "pytorch_model.bin" | head -n 1 || true)"
  if [[ -n "${bin_file}" && -f "${bin_file}" ]]; then
    echo "${bin_file}"
    return 0
  fi

  return 1
}

# -----------------------------
# Select checkpoint state dict
# -----------------------------
CHECKPOINT_STATE_DICT=""

# User-provided STATE_DICT has highest priority.
if [[ -n "${STATE_DICT:-}" ]]; then
  if [[ -f "${STATE_DICT}" ]]; then
    CHECKPOINT_STATE_DICT="${STATE_DICT}"
  elif [[ -d "${STATE_DICT}" ]]; then
    CHECKPOINT_STATE_DICT="$(find_state_dict "${STATE_DICT}" || true)"
  else
    echo "[ERROR] User-provided STATE_DICT does not exist: ${STATE_DICT}"
    exit 1
  fi
fi

# Prefer merged fp32 checkpoint.
if [[ -z "${CHECKPOINT_STATE_DICT}" && -d "${MERGED_FP32_DIR}" ]]; then
  CHECKPOINT_STATE_DICT="$(find_state_dict "${MERGED_FP32_DIR}" || true)"
fi

# Also support already-merged files placed directly in stage output dir.
if [[ -z "${CHECKPOINT_STATE_DICT}" && -d "${STAGE_OUTPUT_DIR}" ]]; then
  CHECKPOINT_STATE_DICT="$(find_state_dict "${STAGE_OUTPUT_DIR}" || true)"
fi

# Also support files directly inside final_model, although DeepSpeed usually
# keeps ZeRO checkpoint files there rather than merged fp32 weights.
if [[ -z "${CHECKPOINT_STATE_DICT}" && -d "${FINAL_CKPT_DIR}" ]]; then
  CHECKPOINT_STATE_DICT="$(find_state_dict "${FINAL_CKPT_DIR}" || true)"
fi

# -----------------------------
# Merge DeepSpeed ZeRO checkpoint if needed
# -----------------------------
if [[ -z "${CHECKPOINT_STATE_DICT}" ]]; then
  echo "============================================================"
  echo "No merged fp32 checkpoint found. Merging ZeRO checkpoint..."
  echo "FINAL_CKPT_DIR=${FINAL_CKPT_DIR}"
  echo "MERGED_FP32_DIR=${MERGED_FP32_DIR}"
  echo "============================================================"

  if [[ ! -f "${FINAL_CKPT_DIR}/zero_to_fp32.py" ]]; then
    echo "[ERROR] zero_to_fp32.py not found in ${FINAL_CKPT_DIR}"
    exit 1
  fi

  # Clean only our own merged output directory.
  rm -rf "${MERGED_FP32_DIR}"
  mkdir -p "${MERGED_FP32_DIR}"

  # Use a neutral prefix without .bin to avoid creating a folder named pytorch_model.bin.
  # zero_to_fp32.py may create:
  #   reaseg_fp32_model
  # or:
  #   reaseg_fp32_model.index.json + reaseg_fp32_model-xxxxx.bin shards
  MERGED_OUTPUT_PREFIX="${MERGED_FP32_DIR}/reaseg_fp32_model"

  python "${FINAL_CKPT_DIR}/zero_to_fp32.py" \
    "${FINAL_CKPT_DIR}" \
    "${MERGED_OUTPUT_PREFIX}"

  CHECKPOINT_STATE_DICT="$(find_state_dict "${MERGED_FP32_DIR}" || true)"
  
  if [[ -z "${CHECKPOINT_STATE_DICT}" && -d "${MERGED_OUTPUT_PREFIX}" ]]; then
    CHECKPOINT_STATE_DICT="$(find_state_dict "${MERGED_OUTPUT_PREFIX}" || true)"
  fi
  
  if [[ -z "${CHECKPOINT_STATE_DICT}" ]]; then
    if [[ -f "${MERGED_OUTPUT_PREFIX}.index.json" ]]; then
      CHECKPOINT_STATE_DICT="${MERGED_OUTPUT_PREFIX}.index.json"
    elif [[ -f "${MERGED_OUTPUT_PREFIX}" ]]; then
      CHECKPOINT_STATE_DICT="${MERGED_OUTPUT_PREFIX}"
    fi
  fi

  if [[ -z "${CHECKPOINT_STATE_DICT}" ]]; then
    echo "[ERROR] Failed to create/find merged fp32 checkpoint in ${MERGED_FP32_DIR}"
    echo "Files in MERGED_FP32_DIR:"
    find "${MERGED_FP32_DIR}" -maxdepth 2 -type f -o -type d | sort || true
    exit 1
  fi
fi

if [[ ! -f "${CHECKPOINT_STATE_DICT}" ]]; then
  echo "[ERROR] Failed to create/find checkpoint state dict: ${CHECKPOINT_STATE_DICT}"
  exit 1
fi

mkdir -p "${TEST_OUTPUT_DIR}"

# -----------------------------
# Run test
# -----------------------------
CMD=(
  python
  tools/test_reaseg.py

  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --medsam_checkpoint "${MEDSAM_CHECKPOINT}"
  --data_path "${DATA_PATH}"
  --split "${SPLIT}"
  --checkpoint_state_dict "${CHECKPOINT_STATE_DICT}"
  --output_dir "${TEST_OUTPUT_DIR}"

  --precision "${PRECISION}"
  --batch_size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --logit_threshold "${LOGIT_THRESHOLD}"
  --limit "${LIMIT}"

  --lora_scope "${LORA_SCOPE}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
)

if [[ "${USE_LORA}" == "1" ]]; then
  CMD+=(--use_lora)
else
  CMD+=(--no-use_lora)
fi

echo "============================================================"
echo "Testing ReaSeg final model"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "MEDSAM_CHECKPOINT=${MEDSAM_CHECKPOINT}"
echo "DATA_PATH=${DATA_PATH}"
echo "FINAL_CKPT_DIR=${FINAL_CKPT_DIR}"
echo "STAGE_OUTPUT_DIR=${STAGE_OUTPUT_DIR}"
echo "MERGED_FP32_DIR=${MERGED_FP32_DIR}"
echo "CHECKPOINT_STATE_DICT=${CHECKPOINT_STATE_DICT}"
echo "TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}"
echo "SPLIT=${SPLIT}"
echo "PRECISION=${PRECISION}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "LOGIT_THRESHOLD=${LOGIT_THRESHOLD}"
echo "USE_LORA=${USE_LORA}"
echo "LORA_SCOPE=${LORA_SCOPE}"
echo "LORA_R=${LORA_R}"
echo "LORA_ALPHA=${LORA_ALPHA}"
echo "============================================================"

printf 'Command:\n'
printf '%q ' "${CMD[@]}"
printf '\n'
echo "============================================================"

"${CMD[@]}"