#!/usr/bin/env bash

# Shared single-A100 launcher. Source this file; do not execute it directly.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-true}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
MAMBA_SCHEDULER_STRATEGY="${MAMBA_SCHEDULER_STRATEGY:-auto}"
RELEASE_MEMORY_FOR_TRAINING="${RELEASE_MEMORY_FOR_TRAINING:-true}"
# Keep the official test set untouched. The trainer's periodic "test" pass is
# a deterministic 500-example validation slice from the original train split.
TRAIN_DATA="${TRAIN_DATA:-train[500:]@openai/gsm8k:main}"
TEST_DATA="${TEST_DATA:-train[:500]@openai/gsm8k:main}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/a100}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/logs}"
PROJECT="${PROJECT:-Exploring-Paths-Completion}"
PROMPTS_PER_ROLLOUT="${PROMPTS_PER_ROLLOUT:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
ACTOR_TOKEN_BUDGET="${ACTOR_TOKEN_BUDGET:-8192}"
# Differentiating entropy retains additional full-vocabulary tensors. Keep
# larger minibatches for zero-coefficient controls and use a safer token pack
# only for experiments where entropy actually participates in the loss.
ENTROPY_ACTOR_TOKEN_BUDGET="${ENTROPY_ACTOR_TOKEN_BUDGET:-6144}"
# SGLang keeps its static allocation in a separate process. At 0.45 it holds
# about 35.7 GiB on an 80 GiB A100, leaving too little headroom for the actor's
# 16K-token backward pass (~58 GiB including its final log-softmax gradient).
# A 0.30 reservation retains about 24 GiB for rollout weights and KV; the 8K
# actor token pack above provides sufficient backward headroom.
ROLLOUT_GPU_FRACTION="${ROLLOUT_GPU_FRACTION:-0.30}"
MAX_STEPS="${MAX_STEPS:-150}"
TEST_FREQ="${TEST_FREQ:-25}"
# One recovery checkpoint at step 100 is enough for the 150-step screen. Each
# optimizer checkpoint is ~24 GiB, so saving at 50/100/150 exhausts the 160 GiB
# workspace after only two methods.
SAVE_FREQ="${SAVE_FREQ:-100}"
RANK_CAP="${RANK_CAP:-64}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

read -r -a EXTRA_HYDRA_ARRAY <<< "${EXTRA_HYDRA_ARGS:-}"

model_slug() {
    local value="${MODEL##*/}"
    printf '%s' "${value}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '-'
}

remove_intermediate_checkpoints() {
    local save_dir="$1"
    local checkpoint
    local removed=0

    # Called only after torchrun and save_model have completed successfully.
    # Restrict deletion to numbered step directories directly under this run.
    for checkpoint in "${save_dir}"/step[0-9]*; do
        if [[ -d "${checkpoint}" && "${checkpoint}" == "${save_dir}"/step* ]]; then
            find "${checkpoint}" -depth -delete
            removed=1
        fi
    done
    if (( removed )); then
        printf 'Removed intermediate optimizer checkpoints from %s\n' "${save_dir}"
    fi
}

run_path() {
    local objective="$1"
    local entropy_coef="$2"
    local seed="$3"
    local entropy_quantile="${4:-0.0}"
    local actor_token_budget="${ACTOR_TOKEN_BUDGET}"
    case "${entropy_coef}" in
        0|0.0|0.00|0.000) ;;
        *) actor_token_budget="${ENTROPY_ACTOR_TOKEN_BUDGET}" ;;
    esac
    local entropy_slug
    entropy_slug="$(printf '%s' "${entropy_coef}" | tr -- '.-' 'pm')"
    local quantile_slug
    quantile_slug="$(printf '%s' "${entropy_quantile}" | tr -- '.-' 'pm')"
    local name="$(model_slug)_${objective}_h${entropy_slug}_q${quantile_slug}_s${seed}"
    local save_dir="${RUN_ROOT}/${name}"
    local log_file="${LOG_ROOT}/${name}.log"

    if [[ -f "${save_dir}/latest/config.json" ]]; then
        remove_intermediate_checkpoints "${save_dir}"
        printf 'Skipping completed run: %s\n' "${name}"
        return
    fi
    local resume_args=()
    if compgen -G "${save_dir}/step*" >/dev/null; then
        resume_args+=(trainer.load_ckpt_from=latest)
        printf 'Resuming latest checkpoint for %s\n' "${name}"
    fi

    printf 'Starting %s\nLog: %s\n' "${name}" "${log_file}"
    torchrun --standalone --nproc_per_node=1 \
        -m RL2.trainer.path \
        "train_data.path='${TRAIN_DATA}'" \
        "train_data.prompts_per_rollout=${PROMPTS_PER_ROLLOUT}" \
        train_data.responses_per_prompt=1 \
        "train_data.enable_thinking=${ENABLE_THINKING}" \
        "test_data.path='${TEST_DATA}'" \
        test_data.responses_per_prompt=1 \
        "actor.model_name=${MODEL}" \
        "actor.use_liger_kernel=${USE_LIGER_KERNEL}" \
        "actor.attn_implementation=${ATTN_IMPLEMENTATION}" \
        "actor.max_length_per_device=${actor_token_budget}" \
        "actor.max_inference_length_per_device=${actor_token_budget}" \
        actor.avg_level=sequence \
        "actor.path.objective=${objective}" \
        "actor.path.rank_cap=${RANK_CAP}" \
        "actor.path.token_entropy_quantile=${entropy_quantile}" \
        actor.path.normalize_selected_tokens=false \
        "actor.entropy.coef=${entropy_coef}" \
        "rollout.train_sampling_params.max_new_tokens=${MAX_NEW_TOKENS}" \
        rollout.train_sampling_params.temperature=1.0 \
        "rollout.gpu_memory_utilization=${ROLLOUT_GPU_FRACTION}" \
        "rollout.mamba_scheduler_strategy=${MAMBA_SCHEDULER_STRATEGY}" \
        "rollout.release_memory_for_training=${RELEASE_MEMORY_FOR_TRAINING}" \
        rollout.env_path=envs/gsm8k.py \
        "trainer.project=${PROJECT}" \
        "trainer.experiment_name=${name}" \
        "trainer.seed=${seed}" \
        trainer.n_epochs=1000 \
        "trainer.max_steps=${MAX_STEPS}" \
        "trainer.test_freq=${TEST_FREQ}" \
        "trainer.save_freq=${SAVE_FREQ}" \
        "trainer.save_dir=${save_dir}" \
        "${resume_args[@]}" \
        "${EXTRA_HYDRA_ARRAY[@]}" \
        2>&1 | tee -a "${log_file}"
    remove_intermediate_checkpoints "${save_dir}"
}

run_drgrpo() {
    local seed="$1"
    local name="$(model_slug)_drgrpo_s${seed}"
    local save_dir="${RUN_ROOT}/${name}"
    local log_file="${LOG_ROOT}/${name}.log"

    if [[ -f "${save_dir}/latest/config.json" ]]; then
        remove_intermediate_checkpoints "${save_dir}"
        printf 'Skipping completed run: %s\n' "${name}"
        return
    fi
    local resume_args=()
    if compgen -G "${save_dir}/step*" >/dev/null; then
        resume_args+=(trainer.load_ckpt_from=latest)
        printf 'Resuming latest checkpoint for %s\n' "${name}"
    fi

    printf 'Starting %s\nLog: %s\n' "${name}" "${log_file}"
    torchrun --standalone --nproc_per_node=1 \
        -m RL2.trainer.ppo \
        "train_data.path='${TRAIN_DATA}'" \
        train_data.prompts_per_rollout=128 \
        train_data.responses_per_prompt=4 \
        "train_data.enable_thinking=${ENABLE_THINKING}" \
        "test_data.path='${TEST_DATA}'" \
        test_data.responses_per_prompt=1 \
        "actor.model_name=${MODEL}" \
        "actor.use_liger_kernel=${USE_LIGER_KERNEL}" \
        "actor.attn_implementation=${ATTN_IMPLEMENTATION}" \
        "actor.max_length_per_device=${ACTOR_TOKEN_BUDGET}" \
        "actor.max_inference_length_per_device=${ACTOR_TOKEN_BUDGET}" \
        "ref_actor.max_inference_length_per_device=${ACTOR_TOKEN_BUDGET}" \
        adv.estimator=reinforce \
        adv.global_norm=false \
        adv.norm_var=false \
        "rollout.train_sampling_params.max_new_tokens=${MAX_NEW_TOKENS}" \
        rollout.train_sampling_params.temperature=1.0 \
        "rollout.gpu_memory_utilization=${ROLLOUT_GPU_FRACTION}" \
        "rollout.mamba_scheduler_strategy=${MAMBA_SCHEDULER_STRATEGY}" \
        "rollout.release_memory_for_training=${RELEASE_MEMORY_FOR_TRAINING}" \
        rollout.env_path=envs/gsm8k.py \
        "trainer.project=${PROJECT}" \
        "trainer.experiment_name=${name}" \
        "trainer.seed=${seed}" \
        trainer.n_epochs=1000 \
        "trainer.max_steps=${MAX_STEPS}" \
        "trainer.test_freq=${TEST_FREQ}" \
        "trainer.save_freq=${SAVE_FREQ}" \
        "trainer.save_dir=${save_dir}" \
        "${resume_args[@]}" \
        "${EXTRA_HYDRA_ARRAY[@]}" \
        2>&1 | tee -a "${log_file}"
    remove_intermediate_checkpoints "${save_dir}"
}
