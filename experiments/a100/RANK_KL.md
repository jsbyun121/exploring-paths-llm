# Rank-KL follow-up

Objective: KL(detached rank-rotated target || model). Same target rotation,
rank cap, successful-response filtering, sequence normalization, learning rate
1e-6, seed 0 and 100-step budget as the existing controls. No loss multiplier.
The sparse implementation includes -target+model terms so its gradients match
full-vocabulary KL, including the unchanged tail. Tests verify value and gradient.
Rank-1 and out-of-cap actions give zero loss and gradient, as in Rank-JSD.

Launch: bash experiments/a100/run_rank_kl_after_focused.sh
It waits on the existing GPU lock, requires completed CE/Rank-JSD comparison
and evaluation summaries, and refuses to start when outputs/focused/STOP exists.
A crash before completion prevents the follow-up from running.
Checkpoints every 10 steps, retain 2; validation every 25; matched 200-question
pass@8 evaluation after step 100, followed by CE and Rank-JSD comparisons.
Grad norm and rank coverage use the existing W&B metrics. Grad norm is before
clipping, not the Adam parameter-update norm. Larger gradients do not establish
faster convergence; this is the hypothesis being tested.

Persistent queue: tmux session paths-rank-kl.
Log: outputs/focused/rank_kl_queue.log
