#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/workspace/Think-MRIS}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/conda-envs/think-mris/bin/python}"
DATASET_PATH="${DATASET_PATH:-/workspace/datasets/MRIS-Bench-calibrated-25k-paper-v4-final}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT}/checkpoints/think_mris_paper_aligned/qwen7b_calibrated25k_paper_v4_24ep_4gpu}"
EVAL_DIR="${EVAL_DIR:-${ROOT}/evaluation_outputs/paper_aligned_v4_full_test}"
STATE_DIR="${STATE_DIR:-${ROOT}/run_state/paper_aligned_pipeline}"
TRAIN_LOG="${TRAIN_LOG:-${ROOT}/logs/paper_aligned_4gpu_train.log}"
MERGE_LOG="${MERGE_LOG:-${ROOT}/logs/paper_aligned_4gpu_merge.log}"
EVAL_LOG="${EVAL_LOG:-${ROOT}/logs/paper_aligned_4gpu_full_eval.log}"
AUDIT_LOG="${AUDIT_LOG:-${ROOT}/logs/paper_aligned_4gpu_result_audit.log}"
PREFLIGHT_LOG="${PREFLIGHT_LOG:-${ROOT}/logs/paper_aligned_vlki_preflight.log}"

export THINK_MRIS_CACHE_ROOT="${THINK_MRIS_CACHE_ROOT:-/workspace/.cache}"
export HF_HOME="${THINK_MRIS_HF_HOME:-${THINK_MRIS_CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${THINK_MRIS_HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${THINK_MRIS_TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export XDG_CACHE_HOME="${THINK_MRIS_XDG_CACHE_HOME:-${THINK_MRIS_CACHE_ROOT}}"

PROXY_URL="${THINK_MRIS_PROXY_URL:-http://127.0.0.1:7890}"
export HTTP_PROXY="${PROXY_URL}"
export HTTPS_PROXY="${PROXY_URL}"
export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"

mkdir -p "${STATE_DIR}" "$(dirname "${TRAIN_LOG}")" "${CHECKPOINT_ROOT}" "${EVAL_DIR}"
cd "${ROOT}"
EXPECTED_TEST_ROWS="$("${PYTHON_BIN}" -c \
    'import sys; from datasets import load_from_disk; print(len(load_from_disk(sys.argv[1])["test"]))' \
    "${DATASET_PATH}")"
printf '%s\n' "${DATASET_PATH}" >"${STATE_DIR}/dataset_path"
printf '%s\n' "${EXPECTED_TEST_ROWS}" >"${STATE_DIR}/expected_test_rows"

write_status() {
    printf 'state=%s checked=%s\n' "$1" "$(date -Is)" >"${STATE_DIR}/status"
}

write_status waiting_for_all_four_gpus
free_streak=0
while (( free_streak < 3 )); do
    foreign_jobs="$(
        ps -eo args= \
            | grep -F '/workspace/retinal_age_paper_20260830' \
            | grep -v -F "grep -F /workspace/retinal_age_paper_20260830" \
            || true
    )"
    gpu_jobs="$(
        nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
            | sed '/^[[:space:]]*$/d'
    )"
    if [[ -z "${foreign_jobs}" && -z "${gpu_jobs}" ]]; then
        free_streak=$((free_streak + 1))
    else
        free_streak=0
    fi
    printf 'state=waiting_for_foreign_pipeline_and_all_four_gpus free_streak=%s/3 foreign_jobs=%s gpu_jobs=%s checked=%s\n' \
        "${free_streak}" \
        "$(grep -c . <<<"${foreign_jobs}" || true)" \
        "$(grep -c . <<<"${gpu_jobs}" || true)" \
        "$(date -Is)" >"${STATE_DIR}/status"
    (( free_streak < 3 )) && sleep 30
done

write_status preflight_vlki_sam2
"${PYTHON_BIN}" -c \
    'import sys; from datasets import load_from_disk; load_from_disk(sys.argv[1])["test"][0]["image"].convert("RGB").save(sys.argv[2])' \
    "${DATASET_PATH}" "${STATE_DIR}/preflight.png"
set +e
"${PYTHON_BIN}" inference_scripts/infer_multi_object.py \
    --reasoning_model_path /workspace/pretrained_models/Qwen2.5-VL-7B-Instruct \
    --processor_path /workspace/pretrained_models/Qwen2.5-VL-7B-Instruct \
    --segmentation_model_path facebook/sam2-hiera-large \
    --image_path "${STATE_DIR}/preflight.png" \
    --text "Which region matches the described medical target?" \
    --output_path "${STATE_DIR}/preflight_vlki_sam2.png" \
    --response_override '<think>The specified region is localized using its visual appearance and spatial context.</think><answer>[{"bbox_2d":[10,10,50,50],"points_2d":[[20,20],[40,40]]}]</answer>' \
    >"${PREFLIGHT_LOG}" 2>&1
preflight_rc=$?
set -e
printf '%s\n' "${preflight_rc}" >"${STATE_DIR}/preflight.exit"
if (( preflight_rc != 0 )); then
    write_status "preflight_failed_rc_${preflight_rc}"
    exit "${preflight_rc}"
fi

write_status preflight_training_n8
"${PYTHON_BIN}" -c \
    'import sys; from pathlib import Path; from datasets import DatasetDict,load_from_disk; p=Path(sys.argv[2]); (None if p.exists() else DatasetDict({"train":load_from_disk(sys.argv[1])["train"].select(range(8))}).save_to_disk(str(p)))' \
    "${DATASET_PATH}" "${DATASET_PATH}-preflight8"
set +e
TRAIN_DATA="${DATASET_PATH}-preflight8" \
ROLLOUT_BATCH_SIZE=8 \
ROLLOUT_N=8 \
MAX_RESPONSE_LENGTH=64 \
MAX_NUM_BATCHED_TOKENS=2048 \
TOTAL_EPISODES=1 \
SAVE_FREQ=-1 \
EXPERIMENT_NAME=qwen7b_calibrated25k_paper_n8_preflight \
CHECKPOINT_PATH="${ROOT}/checkpoints/think_mris_paper_aligned/preflight_n8_4gpu" \
    bash training_scripts/run_calibrated_paper_aligned_4gpu.sh \
    >"${ROOT}/logs/paper_aligned_n8_training_preflight.log" 2>&1
train_preflight_rc=$?
set -e
printf '%s\n' "${train_preflight_rc}" >"${STATE_DIR}/train_preflight.exit"
if (( train_preflight_rc != 0 )); then
    write_status "train_preflight_failed_rc_${train_preflight_rc}"
    exit "${train_preflight_rc}"
fi

write_status training_24_epochs
set +e
TRAIN_DATA="${DATASET_PATH}" \
    bash training_scripts/run_calibrated_paper_aligned_4gpu.sh >"${TRAIN_LOG}" 2>&1
train_rc=$?
set -e
printf '%s\n' "${train_rc}" >"${STATE_DIR}/train.exit"
if (( train_rc != 0 )); then
    write_status "training_failed_rc_${train_rc}"
    exit "${train_rc}"
fi

latest_step="$(tr -d '[:space:]' <"${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt")"
actor_dir="${CHECKPOINT_ROOT}/global_step_${latest_step}/actor"
printf '%s\n' "${actor_dir}" >"${STATE_DIR}/final_actor_path"

write_status merging_final_checkpoint
set +e
"${PYTHON_BIN}" training_scripts/model_merger.py \
    --local_dir "${actor_dir}" \
    --lora_alpha 16 >"${MERGE_LOG}" 2>&1
merge_rc=$?
set -e
printf '%s\n' "${merge_rc}" >"${STATE_DIR}/merge.exit"
if (( merge_rc != 0 )); then
    write_status "merge_failed_rc_${merge_rc}"
    exit "${merge_rc}"
fi

write_status evaluating_full_test_with_vlki_sam2
set +e
REASONING_MODEL_PATH="${actor_dir}/huggingface" \
PROCESSOR_PATH="${actor_dir}/huggingface" \
TEST_DATA_PATH="${DATASET_PATH}" \
OUTPUT_DIR="${EVAL_DIR}" \
BATCH_SIZE=1 \
MAX_NEW_TOKENS=384 \
DEVICE=cuda:0 \
RUN_SAM2=1 \
DISABLE_VLKI=0 \
RESUME=1 \
    bash evaluation_scripts/eval_calibrated_full.sh >"${EVAL_LOG}" 2>&1
eval_rc=$?
set -e
printf '%s\n' "${eval_rc}" >"${STATE_DIR}/eval.exit"
if (( eval_rc != 0 )); then
    write_status "evaluation_failed_rc_${eval_rc}"
    exit "${eval_rc}"
fi

write_status auditing_full_test_result
set +e
"${PYTHON_BIN}" evaluation_scripts/audit_paper_aligned_result.py \
    --evaluation_dir "${EVAL_DIR}" \
    --dataset_dir "${DATASET_PATH}" \
    --expected_rows "${EXPECTED_TEST_ROWS}" \
    --output "${STATE_DIR}/completion_audit.json" \
    >"${AUDIT_LOG}" 2>&1
audit_rc=$?
set -e
printf '%s\n' "${audit_rc}" >"${STATE_DIR}/audit.exit"
if (( audit_rc != 0 )); then
    write_status "result_audit_failed_rc_${audit_rc}"
    exit "${audit_rc}"
fi

write_status complete
