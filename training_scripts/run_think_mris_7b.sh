#!/usr/bin/env bash

set -euo pipefail
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
THINK_MRIS_CACHE_ROOT="${THINK_MRIS_CACHE_ROOT:-/workspace/.cache}"
export HF_HOME="${THINK_MRIS_HF_HOME:-${THINK_MRIS_CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${THINK_MRIS_HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${THINK_MRIS_TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export XDG_CACHE_HOME="${THINK_MRIS_XDG_CACHE_HOME:-${THINK_MRIS_CACHE_ROOT}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

N_GPUS="${N_GPUS:-8}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((N_GPUS - 1)))"
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-/workspace/datasets/MRIS-Bench-calibrated-25k}"
RUN_NAME=$(basename "$0" .sh)

python3 -m verl.trainer.main \
    config=training_scripts/think_mris_7b.yaml \
    data.train_files=${TRAIN_DATA} \
    data.val_files=None \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.kl_loss_coef=5.0e-3 \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.rollout.enable_chunked_prefill=true \
    worker.rollout.n=8 \
    worker.reward.compute_score=think_mris \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.total_episodes=24 \
    trainer.save_checkpoint_path=think_mris_workdir/${RUN_NAME}

