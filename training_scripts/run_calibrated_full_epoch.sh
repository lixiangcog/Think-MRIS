#!/usr/bin/env bash

set -euo pipefail
set -x

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export THINK_MRIS_CACHE_ROOT="${THINK_MRIS_CACHE_ROOT:-/workspace/.cache}"
export HF_HOME="${THINK_MRIS_HF_HOME:-${THINK_MRIS_CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${THINK_MRIS_HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${THINK_MRIS_TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export XDG_CACHE_HOME="${THINK_MRIS_XDG_CACHE_HOME:-${THINK_MRIS_CACHE_ROOT}}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/conda-envs/think-mris/bin/python}"
MODEL_PATH="${MODEL_PATH:-/workspace/pretrained_models/Qwen2.5-VL-7B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-/workspace/datasets/MRIS-Bench-calibrated-25k}"
N_GPUS="${N_GPUS:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-2}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-96}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-false}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen7b_calibrated25k_full_epoch}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/workspace/Think-MRIS/checkpoints/think_mris_full/${EXPERIMENT_NAME}}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((N_GPUS - 1)))"
fi

"${PYTHON_BIN}" -m verl.trainer.main \
    config=training_scripts/think_mris_7b.yaml \
    data.train_files="${TRAIN_DATA}" \
    data.val_files=None \
    data.rollout_batch_size="${ROLLOUT_BATCH_SIZE}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.drop_last=false \
    worker.actor.global_batch_size="${ROLLOUT_BATCH_SIZE}" \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.model.tokenizer_path="${MODEL_PATH}" \
    worker.actor.model.attn_implementation=sdpa \
    worker.rollout.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    worker.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    worker.rollout.n="${ROLLOUT_N}" \
    worker.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    worker.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    worker.rollout.enforce_eager=true \
    worker.rollout.enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL}" \
    worker.reward.compute_score=think_mris \
    trainer.project_name=think_mris_full \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.total_episodes=1 \
    trainer.save_freq=80 \
    trainer.remove_previous_ckpt=true \
    trainer.test_freq=-1 \
    trainer.val_before_train=false \
    trainer.save_checkpoint_path="${CHECKPOINT_PATH}"
