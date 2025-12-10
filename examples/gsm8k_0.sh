torchrun \
    --nproc_per_node=1 \
    --rdzv_endpoint=localhost:29500 \
    -m RL2.trainer.spo \
    train_data.path=train@openai/gsm8k:main \
    train_data.prompts_per_rollout=512 \
    train_data.responses_per_prompt=1 \
    test_data.path=test@openai/gsm8k:main \
    test_data.prompts_per_rollout=512 \
    test_data.responses_per_prompt=1 \
    actor.model_name=Qwen/Qwen3-4B-Thinking-2507 \
    actor.use_liger_kernel=true \
    actor.max_length_per_device=8576 \
    actor.avg_level=sequence \
    actor.entropy.coef=-0 \
    rollout.train_sampling_params.max_new_tokens=8192 \
    rollout.env_path=envs/gsm8k.py \
    rollout.gpu_memory_utilization=0.3 \
    trainer.project=GSM8K \
    trainer.experiment_name=qwen3-4b-thinking-2507_512_0\
    trainer.n_epochs=150 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \