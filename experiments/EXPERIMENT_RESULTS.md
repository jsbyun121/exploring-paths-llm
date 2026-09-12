# Experiment results

Local artifact snapshot: 2026-09-12 (UTC).

Evidence is stored in [experiment_metrics.json](experiment_metrics.json), including source paths, SHA-256 hashes, evaluation settings, confidence intervals, paired comparisons, and validation histories. Evaluation aggregates were checked against every saved prediction. No new training or model evaluation is performed by this report.

## Evaluation protocol and limits

All completed evaluations below use Qwen3-4B-Thinking-2507 and the same first 200 examples of `train[:500]@openai/gsm8k:main`, held out from `train[500:]`. This subset overlaps the 500-question validation set used for checkpoint selection; it is not the official GSM8K test set. The training prompt/verifier is used, with greedy decoding and eight samples per problem at temperature 0.7, top-p 0.95, a 4096-token response cap, and seed 0. Pass@8 is the fraction of problems solved by at least one of those eight samples. Tokens are mean completion tokens per response, not equalized inference budgets.

These are exploratory single-seed recipe comparisons. Historical JSD used an older runtime; earlier resumes could omit Adam moments. The repaired recovery path restores optimizer and RNG state. Padding/batching changes can affect BF16 numerics. Beta1=0 removes first-moment history but retains second-moment adaptation. These confounds prevent a clean optimizer or loss-function attribution.

## Focused 100-step diagnostics

| Model/checkpoint | Greedy accuracy | Pass@8 | Sample accuracy | Greedy tokens | Sample tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 81.5% | 96.0% | 77.94% | 1129.5 | 1184.0 |
| Positive CE, step 100 | 90.5% | 98.0% | 91.00% | 1019.0 | 1019.3 |
| Historical rank-JSD, step 100 | 84.5% | 96.5% | 84.69% | 1297.3 | 1400.4 |
| Rank-KL, step 100 | 83.5% | 96.0% | 81.50% | 1330.7 | 1433.7 |
| Rank-JSD beta1=0, step 100 | 92.5% | 97.5% | 90.00% | 1032.6 | 1185.0 |
| Rank-KL beta1=0, step 100 | 87.5% | 97.0% | 86.31% | 1174.2 | 1278.4 |

## Long-run training status

Validation uses all 500 held-out questions. Completed arms stopped when current validation accuracy fell below both previous validation measurements. Evaluations use the saved best checkpoint, not the stopping checkpoint.

| Recipe | Last saved step | Best step | Best validation | Last validation | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Rank-JSD beta1=0 | 220 | 180 | 93.8% | 90.4% (step 220) | Completed |
| Positive CE | 320 | 300 | 93.4% | 92.2% (step 320) | Completed |
| Dr.GRPO | 120 | 100 | 93.2% | 90.8% (step 120) | Completed |
| SAPO | 70 | 60 | 91.6% | 91.6% (step 60) | Incomplete |

CE and JSD continue from step 100. Their recorded `generated_tokens`, `rollout_responses`, and `used_seconds` cover the long continuation only; they exclude the initial focused run. Dr.GRPO and SAPO start at step 0. Do not treat these counters as matched total training exposure. JSD uses beta1=0, one response per prompt and one update epoch; Dr.GRPO/SAPO use beta1=0.9, eight responses per prompt and four update epochs. See the [run configurations](a100/long_comparison/).

## Long-run best-checkpoint diagnostics

| Model/checkpoint | Greedy accuracy | Pass@8 | Sample accuracy | Greedy tokens | Sample tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rank-JSD beta1=0, best step 180 | 93.5% | 97.5% | 93.38% | 830.0 | 1059.0 |
| Positive CE, best step 300 | 95.0% | 98.0% | 94.50% | 860.9 | 885.4 |
| Dr.GRPO, best step 100 | 94.5% | 97.0% | 92.12% | 1050.8 | 1084.5 |
| SAPO, best step 60 | Pending | — | — | — | — |

CE minus JSD greedy accuracy is 1.5 percentage points (paired problem-bootstrap 95% interval: -1.5 to 5.0; 10,000 draws). This interval includes zero. The JSON includes the other existing paired comparisons and completion-token-budget results.

## Forgetting evaluation

No local lm-eval result files are available. No retention score or catastrophic-forgetting conclusion is reported.

Dr.GRPO checkpoints 100 and 120 were captured; 180 and 220 are unavailable because training stopped at 120. The JSD step-220 recovery snapshot contains step-180 best weights: export the step-220 DCP to evaluate step 220. See the [forgetting protocol](forgetting/README.md).

## Reproduce the report

With the original local outputs available:

```bash
python3 -m experiments.update_results --date 2026-09-12
```

Render from committed evidence only:

```bash
python3 -m experiments.update_results --snapshot experiments/experiment_metrics.json
```

Raw generations, checkpoints, dependency installations, and logs remain outside Git. The long-run backup/archive scripts require the deployment-specific helpers in `/workspace/backup-integration`; they are not a portable backup service. Launchers also contain deployment paths that must be adapted on another host.
