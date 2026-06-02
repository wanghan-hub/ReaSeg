#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ReaSeg dataset debug script
#
# Checks:
#   - JSON loading
#   - image loading
#   - mask loading
#   - Qwen2.5-VL processor output
#   - <|image_pad|> token alignment
#   - [SEG] token count
#   - MedSAM image / mask shapes
#   - collate_fn output
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

# -----------------------------
# Paths
# -----------------------------
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${PROJECT_ROOT}/checkpoints/Qwen/Qwen2.5-VL-3B-Instruct}"

# DATA_PATH should be:
#   1) a directory containing train.json / val.json / test.json
#   2) a single JSON file
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/dataset}"

# -----------------------------
# Dataset options
# -----------------------------
SPLIT="${SPLIT:-train}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
PRECISION="${PRECISION:-bf16}"
SEG_TOKEN="${SEG_TOKEN:-[SEG]}"

# If 1, auto append [SEG] when missing in train answers.
FORCE_SEG_TOKEN="${FORCE_SEG_TOKEN:-1}"

# If 1, merge multiple masks when answer has only one [SEG].
MERGE_MASKS_FOR_SINGLE_SEG="${MERGE_MASKS_FOR_SINGLE_SEG:-1}"

CMD=(
  python
  tools/debug_dataset.py

  --data_path "${DATA_PATH}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --split "${SPLIT}"
  --sample_index "${SAMPLE_INDEX}"
  --batch_size "${BATCH_SIZE}"
  --image_size "${IMAGE_SIZE}"
  --max_seq_length "${MAX_SEQ_LENGTH}"
  --precision "${PRECISION}"
  --seg_token "${SEG_TOKEN}"
)

if [[ "${FORCE_SEG_TOKEN}" != "1" ]]; then
  CMD+=(--no_force_seg_token)
fi

if [[ "${MERGE_MASKS_FOR_SINGLE_SEG}" != "1" ]]; then
  CMD+=(--no_merge_masks_for_single_seg)
fi

echo "============================================================"
echo "ReaSeg dataset debug"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "DATA_PATH=${DATA_PATH}"
echo "SPLIT=${SPLIT}"
echo "SAMPLE_INDEX=${SAMPLE_INDEX}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "PRECISION=${PRECISION}"
echo "============================================================"

printf 'Command:\n'
printf '%q ' "${CMD[@]}"
printf '\n'
echo "============================================================"

"${CMD[@]}"