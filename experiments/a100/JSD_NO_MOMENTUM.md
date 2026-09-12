# Rank-JSD without first-moment history

Queued after successful Rank-KL 100-step training and matched evaluations.
Fresh Qwen3-4B-Thinking-2507, seed 0, 100 steps, AdamW beta1=0,
beta2=0.999, learning rate 1e-6, weight decay 0.01, grad clip 1.
Other sampling, loss and evaluation settings match the focused runs.
This removes first-moment history only; second-moment adaptation is retained.
Existing historical JSD uses the older runtime, so comparison is not a pure
optimizer ablation across identical software environments.

Separate output: outputs/focused/training_jsd_beta1_zero
W&B name: qwen3-4b-thinking-2507_rank_jsd_beta1_zero_s0
Log: outputs/focused/jsd_beta1_zero_queue.log
Session: paths-jsd-beta1-zero
Launcher: bash experiments/a100/run_jsd_no_momentum_after_kl.sh
Pause: outputs/focused/STOP (also respected by current KL).
Completed prior runs retain final weights and step100 optimizer recovery;
redundant step90 and discarded speed-test step101 are removed before launch.
