# Research assessment and completion plan

Review date: 2026-08-21

Thesis reviewed: *Exploring Paths in Probabilistic Graphs for Model Training*, especially Section 4, “Limitations and Future Work.”

## Bottom line

The direction is still viable, but the defensible research question is narrower than the thesis states:

> Can a stop-gradient, rank-aware self-imitation loss on verified successful rollouts preserve useful low-probability reasoning forks better than positive-only cross-entropy, the thesis's Bernoulli-KL loss, and Dr.GRPO under matched sampling and compute?

This is worth pursuing. The rank-aware JSD objective is still sufficiently distinct to merit a careful study, particularly because it updates only sampled non-greedy decisions and naturally overlaps with the now well-supported “forking token” view of reasoning. It should not currently be presented as a solution to temporal credit assignment: it still gives the same outcome label to every token in a successful trajectory and extracts no correct partial progress from failed trajectories.

Section 4.2's fixed global entropy penalty is useful as an ablation, not as the likely final method. Work published after the thesis shows that indiscriminate entropy control is coefficient-sensitive and can create either collapse or explosion. Section 4.4's pretraining proposal is the weakest direction: cross-entropy is a proper likelihood objective, whereas standalone rank-JSD stops learning as soon as the gold token reaches rank 1 and does not calibrate the remaining distribution. A CE+rank-JSD continued-pretraining pilot is justified only if post-training rank-JSD first passes the gates below.

## What Section 4 proposes and how the field has moved

| Thesis item | Current assessment | Required response |
|---|---|---|
| 4.1: accuracy eventually collapses | Important and still current, but the thesis observes *entropy growth*, while most later RLVR work studies *entropy collapse*. This difference may be caused by the objective implementation. | Separate intended loss behavior from implementation artifacts; report both entropy of all rollout tokens and entropy on verified paths. |
| 4.2: add `L_total = L_KL + lambda H` | Viable as a diagnostic. Fixed global coefficients are now known to be brittle. | Sweep fixed penalties, then compare only the best variant with selective/high-entropy-token updates. |
| 4.3: rank-aware target + JSD | The strongest and most publishable continuation. It aligns with evidence that high-entropy fork tokens drive RLVR gains. | Use a detached target, measure exact top-k coverage, compare with positive-only CE, and validate on more than one model family. |
| 4.4: apply greedy JSD during pretraining | Standalone replacement for CE is poorly justified. | Run CE, rank-JSD, and CE+rank-JSD at a fixed token budget only after the post-training gate passes. |

Two complexity qualifications matter. The JSD arithmetic is O(k) because the unchanged tail contributes zero. Finding whether the sampled action is in the top k still scans/selects from the vocabulary, and the language-model head still emits all vocabulary logits. The thesis's claim of reducing the whole computation from O(|V|) to O(k) is therefore too strong; only the additional divergence calculation has that complexity.

## Code-level confounds found in the original implementation

These are not cosmetic; they can explain the reported training curves.

1. The maximum-token distribution is a moving target. In the original Bernoulli-KL code, `p_max` remains in the autograd graph. Optimization can lower the maximum probability as well as raise the chosen probability. That contradicts the prose claim that the chosen token is raised to a fixed snapshot of the current greedy token and offers a direct mechanism for increasing entropy.

2. The fixed-0.5 implementation applies the loss only when the chosen token already has probability greater than 0.5. Tokens below 0.5 receive zero loss, so the implementation does not provide the claimed stronger promotion of low-probability correct actions. The experiment suite exposes this as `fixed_half_legacy` and includes the mathematically corrected `fixed_half`.

3. The GSM8K verifier used substring matching. A ground truth such as `12` could accept a prediction such as `312`. This is especially dangerous for a method that trains only on positive rollouts. The verifier now uses exact symbolic/numeric checking.

4. Several old example shell scripts concatenate the experiment name and the next Hydra override because the line-continuation backslash has no preceding space. The new launchers use one argument per line and do not inherit this problem.

5. The original results are one-model, one-dataset curves without seed uncertainty, sampled pass@k, or a positive-only cross-entropy control. They cannot distinguish the proposed divergence from ordinary rejection-sampling self-training.

## Related primary research

The following papers are the most decision-relevant sources found through 2026-08-21:

- [Understanding R1-Zero-Like Training: A Critical Perspective (Dr.GRPO)](https://arxiv.org/abs/2503.20783) identifies length bias in GRPO and is the correct baseline lineage for the thesis.
- [DAPO](https://arxiv.org/abs/2503.14476) shows that clipping, dynamic sampling, overlong handling, and token-level policy-gradient design materially affect stable reasoning RL.
- [Beyond the 80/20 Rule](https://arxiv.org/abs/2506.01939), accepted at NeurIPS 2025, finds that a minority of high-entropy “forking tokens” drive most RLVR improvement and that restricting gradients to them can outperform full-token updates. This is the clearest support for testing rank-JSD.
- [Rethinking Entropy Regularization in Large Reasoning Models (SIREN)](https://arxiv.org/abs/2509.25133) finds naive entropy regularization can cause global entropy explosion and proposes selective, self-anchored control.
- [Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](https://arxiv.org/abs/2511.05993) attributes entropy dynamics to off-policy updates, data diversity, clipping, and the balance between positive- and negative-advantage tokens.
- [Understanding and Preventing Entropy Collapse in RLVR with On-Policy Entropy Flow Optimization](https://arxiv.org/abs/2605.11491) analyzes token-level entropy flow and reports that adaptive balancing is more precise than coarse entropy bonuses or penalties.
- [Process Reinforcement through Implicit Rewards (PRIME)](https://arxiv.org/abs/2502.01456) supplies a strong process-credit comparison using outcome labels to learn implicit process rewards.
- [InT: Self-Proposed Interventions Enable Credit Assignment in LLM Reasoning](https://arxiv.org/abs/2601.14209) localizes the first reasoning error and demonstrates what genuine finer-grained credit assignment looks like.
- [From Reasoning Chains to Verifiable Subproblems (SCRL)](https://arxiv.org/abs/2605.22074) converts partial progress into verifiable subproblem rewards and directly addresses the failed-trajectory limitation that the thesis method does not.
- [Does Reinforcement Learning Really Incentivize Reasoning Capacity Beyond the Base Model?](https://arxiv.org/abs/2504.13837) reports that RL often reweights existing paths while narrowing large-k coverage, making pass@k a necessary outcome.
- [RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs](https://arxiv.org/abs/2506.14245) argues that answer-only pass@k can count invalid reasoning and motivates auditing a sample of chains, not only final answers.
- [Spurious Rewards: Rethinking Training Signals in RLVR](https://arxiv.org/abs/2506.10947) shows large apparent gains can be Qwen-specific and can occur with uninformative rewards, making a second model family essential.

No directly matching published method was found that constructs the exact rank-shifted, stop-gradient current-token distribution proposed in Section 4.3. Novelty therefore remains plausible, but it depends on the corrected formulation and comparisons below.

## Experiments needed for a complete study

### Gate 0: objective correctness

The CPU tests in `tests/test_path_losses.py` compare rank-JSD with an explicit full-vocabulary reference, check its gradient direction, check rank-cap skipping, and preserve both legacy bugs as named ablations. These tests are validation of the scripts, not model experiments.

### Stage 1: mechanism screen

Use the original Qwen3-4B-Thinking-2507 model, GSM8K, one seed, 512 generated sequences per update, and 150 updates. Reserve the first 500 original training examples for validation, train on the remainder, and leave the official test split untouched until final evaluation. Compare:

1. Dr.GRPO with 128 prompts x 4 responses.
2. `positive_ce`: rejection-sampling/SFT on verified rollouts.
3. `bernoulli_kl_legacy`: the original moving-target loss.
4. `bernoulli_kl_detached`: the intended snapshot target.
5. `fixed_half_legacy`: exact reproduction of the probability-greater-than-0.5 mask.
6. `fixed_half`: corrected KL(B(0.5) || B(p_chosen)).
7. `rank_jsd`, rank cap 64.
8. `rank_jsd` with a small entropy penalty.

This screen answers whether the reported entropy trend comes from the moving target, whether the fixed-target collapse depends on the mask bug, and whether rank-JSD does more than ordinary positive-only CE.

### Stage 2: Section 4.2 entropy study

For the legacy Bernoulli-KL, detached Bernoulli-KL, and rank-JSD losses, sweep entropy coefficients `{0, -0.01, -0.03, -0.1}`. The trainer minimizes `loss - coefficient * entropy`, so negative coefficients implement the thesis's positive entropy penalty.

Do not choose the coefficient by final GSM8K test accuracy. Select it using a held-out validation subset or predeclared stability criterion, then evaluate the untouched test set once. A stable run must avoid both a greater-than-50% decline from its peak greedy accuracy and an unbounded entropy trend through update 300.

### Stage 3: rank-JSD confirmation

Run the shortlisted rank-JSD and strongest non-rank control for three seeds and 300 updates with the thesis's 8192-token generation cap. Required ablations:

- rank caps 20, 64, and 256 if cap-64 coverage is below 95%;
- all non-greedy tokens versus the top 20% highest-entropy tokens (`token_entropy_quantile=0.8`);
- entropy coefficient 0 versus the selected Section 4.2 coefficient;
- positive-only CE at the same verified-token and update budget.

The rank-cap coverage metric must be reported. An out-of-cap action is skipped, not silently clipped to a fake rank, because clipping would not preserve probability mass under the proposed shift.

### Stage 4: evaluation and generalization

For the base model and every final checkpoint, report:

- greedy accuracy with a Wilson 95% interval;
- sampled accuracy, pass@8, and unique final-answer count at temperature 0.7;
- mean generated tokens and length-truncation rate;
- entropy on all action tokens and on verified trajectories;
- rank-1 fraction, cap coverage, and mean promoted-token rank;
- peak, final, and area-under-training-curve accuracy rather than only the best checkpoint.

Repeat the winning method and strongest control on a second non-Qwen model family that fits the A100. This is mandatory for a general algorithmic claim because spurious-reward gains have been shown to vary sharply by model family. Also evaluate out of domain on a held-out math benchmark. Final-answer scores should be accompanied by a blinded manual audit of at least 100 sampled chains, stratified by method and correctness, because answer-only verification can reward invalid reasoning.

### Stage 5: Section 4.4 continued-pretraining pilot

Only run this stage if rank-JSD beats positive-only CE in Stage 3. Continue pretraining Qwen3-1.7B-Base on the same hash-partitioned OpenWebMath stream for a fixed 32.8M-token budget per condition:

- CE;
- standalone rank-JSD;
- CE + rank-JSD.

Reject standalone rank-JSD if top-k coverage is low, validation perplexity regresses, or downstream accuracy fails to improve. Advance the hybrid only if validation perplexity stays within 2% of CE and reasoning accuracy improves across seeds. A full “pretraining application” claim would eventually require training from scratch or a much larger token budget; this pilot can only establish whether the idea deserves that expense.

## Decision criteria

The post-training direction passes if, across three seeds, rank-JSD:

- improves greedy accuracy or pass@8 over both positive-only CE and detached Bernoulli-KL by at least 1 absolute point on average;
- retains the gain on a second model family or task;
- does not trade the gain for more than 10% additional generated tokens;
- has at least 95% cap coverage, or produces the same conclusion after increasing the cap;
- remains stable through 300 updates; and
- shows that its gains occur at non-greedy/high-entropy fork tokens rather than formatting tokens alone.

If it only matches positive-only CE, the new divergence is not justified. If only the legacy moving-target version works, the paper should be reframed as entropy-shaping through a coupled target rather than rank promotion. If it works only on Qwen/GSM8K, it remains an interesting case study but not a general training algorithm.

## How to run

Install the repository environment first (`uv sync` and activate `.venv`). No training or evaluation was run while preparing these files.

```bash
# Short, one-seed audit (eight sequential conditions by default)
bash experiments/a100/run_screen.sh

# Section 4.2 coefficient screen
bash experiments/a100/run_entropy_sweep.sh

# After selecting a method/coefficient, three-seed 300-step confirmation
CONDITIONS='rank_jsd:-0.03 positive_ce:0.0' \
  bash experiments/a100/run_confirm.sh

# Sampled final evaluation of a saved Hugging Face checkpoint
bash experiments/a100/evaluate_model.sh \
  outputs/a100/<run-name>/latest <evaluation-name>

# Run only if the post-training gate passes
bash experiments/a100/run_pretraining_pilot.sh
```

All launch parameters can be overridden with environment variables. Useful controls include `MODEL`, `MAX_STEPS`, `MAX_NEW_TOKENS`, `ACTOR_TOKEN_BUDGET`, `ROLLOUT_GPU_FRACTION`, `METHODS`, `OBJECTIVES`, `COEFFICIENTS`, `CONDITIONS`, `SEEDS`, `RANK_CAP`, and `EXTRA_HYDRA_ARGS`. Training stdout is saved under `outputs/a100/logs`; evaluation writes `summary.json` and per-example `predictions.jsonl`.

After a stage finishes, extract endpoint, peak, retention, and area-under-curve values with `python experiments/summarize_training_logs.py outputs/a100/logs/*.log --output outputs/a100/training_summary.json`.

For the next analysis step, return the stage-one `.log` files, the W&B export if online logging was used, and each evaluation `summary.json`. If file size is an issue, send the logs for `positive_ce`, `bernoulli_kl_legacy`, `bernoulli_kl_detached`, `rank_jsd`, and Dr.GRPO first.
