#!/usr/bin/env bash

set -euo pipefail
set -x

# Hardware-adapted paper reproduction. The effective GRPO update batch remains
# 64 prompts x 8 candidates = 512; four GPUs therefore accumulate twice as many
# micro-batches as the paper's eight-H20 run.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export THINK_MRIS_CACHE_ROOT="${THINK_MRIS_CACHE_ROOT:-/workspace/.cache}"
export HF_HOME="${THINK_MRIS_HF_HOME:-${THINK_MRIS_CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${THINK_MRIS_HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${THINK_MRIS_TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export XDG_CACHE_HOME="${THINK_MRIS_XDG_CACHE_HOME:-${THINK_MRIS_CACHE_ROOT}}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/conda-envs/think-mris/bin/python}"
MODEL_PATH="${MODEL_PATH:-/workspace/pretrained_models/Qwen2.5-VL-7B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-/workspace/datasets/MRIS-Bench-calibrated-25k-paper-v4-final}"
N_GPUS="${N_GPUS:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-384}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
TOTAL_EPISODES="${TOTAL_EPISODES:-24}"
if [[ -z "${SAVE_FREQ:-}" ]]; then
    TRAIN_ROWS="$("${PYTHON_BIN}" -c \
        'import sys; from datasets import load_from_disk; data=load_from_disk(sys.argv[1]); print(len(data["train"] if hasattr(data, "keys") else data))' \
        "${TRAIN_DATA}")"
    SAVE_FREQ="$(( (TRAIN_ROWS + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE ))"
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen7b_calibrated25k_paper_v4_24ep_4gpu}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/workspace/Think-MRIS/checkpoints/think_mris_paper_aligned/${EXPERIMENT_NAME}}"

if [[ "${N_GPUS}" != "4" ]]; then
    echo "This audited profile is defined for exactly four visible GPUs." >&2
    exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
fi

"${PYTHON_BIN}" -m verl.trainer.main \
    config=training_scripts/think_mris_7b.yaml \
    data.train_files="${TRAIN_DATA}" \
    data.val_files=None \
    data.rollout_batch_size="${ROLLOUT_BATCH_SIZE}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.drop_last=false \
    worker.actor.global_batch_size="${ROLLOUT_BATCH_SIZE}" \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.model.tokenizer_path="${MODEL_PATH}" \
    worker.actor.model.attn_implementation=sdpa \
    worker.actor.model.lora_rank=8 \
    worker.actor.model.lora_alpha=16 \
    worker.actor.model.lora_target_modules=all-linear \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.rollout.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    worker.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    worker.rollout.n="${ROLLOUT_N}" \
    worker.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    worker.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    worker.rollout.enforce_eager=true \
    worker.rollout.enable_chunked_prefill=true \
    worker.reward.compute_score=think_mris \
    trainer.project_name=think_mris_paper_alignment \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=4 \
    trainer.total_episodes="${TOTAL_EPISODES}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.remove_previous_ckpt=true \
    trainer.test_freq=-1 \
    trainer.val_before_train=false \
    trainer.save_checkpoint_path="${CHECKPOINT_PATH}"
