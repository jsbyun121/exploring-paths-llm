# Catastrophic-forgetting comparison

This suite tests the preregistered hypothesis that rank-shift JSD retains more of
`Qwen/Qwen3-4B-Thinking-2507`'s general capability than Dr.GRPO at matched RL
exposure. It does not assume that the hypothesis is true.

## Primary design

The primary forgetting score is the equal-weight macro accuracy delta from the
base model on MMLU, ARC-Challenge, HellaSwag, and WinoGrande. GSM8K is the
training domain and is reported separately as plasticity, never folded into the
forgetting aggregate. TruthfulQA and WikiText are secondary diagnostics.

Compare JSD and Dr.GRPO first by cumulative generated tokens, then by rollout
response count when historical token accounting is unavailable. Step-matched
results are a sensitivity analysis. The primary trajectory statistic is the
area under the JSD-minus-Dr.GRPO retention curve. Task-level claims use paired
item logs and confidence intervals; all missing runs and parse failures remain
visible.

An algorithm-level claim needs multiple training seeds. Seed 0 alone is an
exploratory checkpoint study, even if its confidence interval excludes zero.
Moreover, the existing recipes differ beyond the loss: JSD uses `beta1=0`, one
response per prompt, and one update epoch, while queued Dr.GRPO uses
`beta1=0.9`, eight responses per prompt, and four update epochs. These models
can support only a **recipe-level** comparison. A causal loss-function claim
requires the three-seed matched control specified in `preregistered_suite.json`.

## Artifact guardrail

The received `long-jsd-s0/step220` snapshot contains a step-220 recovery DCP but
its `best/` directory contains step-180 weights. Export the DCP with
`experiments/export_path_checkpoint.py`; do not evaluate `step220/best` under a
step-220 label. `models.json` encodes this constraint.

## A100 setup

Install evaluation-only dependencies without mutating the live training venv:

```bash
bash experiments/forgetting/setup_lm_eval.sh
```

After the A100 training queue finishes, stage the received JSD artifacts from
the Mac. This uploads the actual step-220 recovery and exports it on A100; it
cannot accidentally substitute the step-180 `best/` model.

```bash
bash experiments/forgetting/stage_jsd_models_to_a100.sh
```

Then print the exact commands before consuming GPU time:

```bash
bash experiments/forgetting/run_suite.sh --dry-run
```

The queued Dr.GRPO long run is watched by `capture_dr_grpo.py`. It hard-links a
completed recovery checkpoint briefly, exports actor-only Hugging Face weights
on CPU, and releases only its own hard links. It never deletes or edits the
trainer checkpoint. The registered steps are 100, 120, 180, and 220; steps
beyond an early-stopped run are explicitly recorded as unavailable.

Run one smoke case before the full matrix:

```bash
bash experiments/forgetting/run_suite.sh \
  --model-id base --task arc_challenge --limit 2
```

Then run the registered primary suite. Each model/task directory is resumable
and stores the exact command, model fingerprint, samples, result, and status.

```bash
bash experiments/forgetting/run_suite.sh
```

Summarize only after all registered runs exist:

```bash
PYTHONPATH=.forgetting-deps .venv/bin/python \
  -m experiments.forgetting.summarize
```

The outputs are written under `outputs/forgetting/`. Do not alter
`preregistered_suite.json` after looking at results; create a versioned v2 plan
for any follow-up analysis.
