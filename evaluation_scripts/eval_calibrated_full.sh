#!/usr/bin/env bash

set -euo pipefail

export THINK_MRIS_CACHE_ROOT="${THINK_MRIS_CACHE_ROOT:-/workspace/.cache}"
export HF_HOME="${THINK_MRIS_HF_HOME:-${THINK_MRIS_CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${THINK_MRIS_HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${THINK_MRIS_TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"

REASONING_MODEL_PATH="${REASONING_MODEL_PATH:?set REASONING_MODEL_PATH to a merged Hugging Face checkpoint}"
PROCESSOR_PATH="${PROCESSOR_PATH:-${REASONING_MODEL_PATH}}"
TEST_DATA_PATH="${TEST_DATA_PATH:-/workspace/datasets/MRIS-Bench-calibrated-25k-paper-v4-final-integer}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/Think-MRIS/evaluation_outputs/calibrated-full-test}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
DEVICE="${DEVICE:-cuda:0}"

extra_args=()
if [[ "${RUN_SAM2:-1}" == "1" ]]; then
    extra_args+=(--run_sam2)
fi
if [[ "${DISABLE_VLKI:-0}" == "1" ]]; then
    extra_args+=(--disable_vlki)
fi
if [[ "${RESUME:-1}" == "1" ]]; then
    extra_args+=(--resume)
fi

python evaluation_scripts/evaluate_calibrated_dataset.py \
    --reasoning_model_path "${REASONING_MODEL_PATH}" \
    --processor_path "${PROCESSOR_PATH}" \
    --test_data_path "${TEST_DATA_PATH}" \
    --split test \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --device "${DEVICE}" \
    "${extra_args[@]}"
