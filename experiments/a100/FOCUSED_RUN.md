# Bounded CE / rank-JSD diagnostic

Run from the repository root:

```bash
mkdir -p outputs/focused
nohup bash experiments/a100/run_focused.sh > outputs/focused/launcher.log 2>&1 &
tail -f outputs/focused/launcher.log
```

The launcher holds the existing container-2 GPU lock and runs sequentially:

1. Preserve the original rank-JSD step100 actor as a Hugging Face model (CPU-only export).
2. Evaluate base and rank-JSD step100 on the same first 200 held-out training-split examples, with greedy decoding and eight temperature-0.7 samples. Both use the training prompt/verifier and a 4096-token response cap. Write a paired comparison.
3. If `CE_MODEL_PATH` and verified `CE_MODEL_STEP=100` are supplied, use those weights. Otherwise rebuild only positive CE to step100 in `outputs/focused/training`, saving every ten steps and keeping two recovery checkpoints.
4. Evaluate CE with the same prompts/sampling seeds and write `outputs/focused/ce100_vs_rank100.json`. Stop here. No entropy sweep or automatic 300-step run.

The first 200 examples are a diagnostic subset of the existing 500-example validation set, not an untouched test set. Neither a single seed nor a selected subset establishes final algorithmic superiority. The comparison includes problem-level paired bootstrap intervals, exclusive solved-problem IDs, actual token usage and success under cumulative completion-token budgets. Different samples within the same problem are not treated as independent test examples. Dr.GRPO's existing history remains a reference; this queue does not rerun it.

To pause safely, create `outputs/focused/STOP`. Training finishes the current optimizer update and writes a recovery checkpoint; evaluation finishes its current request chunk and preserves every completed example. Remove this file and rerun the same launcher to resume. Do not use a process-group kill for a graceful checkpoint: SGLang children are required to finish the update. An abrupt container stop still loses work since the last checkpoint/chunk.

Recovery checkpoints are published from temporary directories only after DCP completes and all indexed shard byte ranges exist. Only then are old published checkpoints pruned. They include optimizer, scheduler, dataloader, Python/NumPy/Torch/CUDA RNG state. Request sampling seeds are derived from training step and request index. Legacy step100 lacks RNG state, so its first continuation is explicitly a new branch. W&B attempts have parent-checkpoint metadata and source/config snapshots; rolled-back attempts are not independent seeds.

The loader also allocates saved Adam state tensors from the DCP index before loading. Previously a fresh Adam optimizer supplied an empty destination state, so DCP omitted its saved moments. This is a substantive historical resume confound: old resumed curves should not be described as uninterrupted optimizer continuations. The recovery test now compares the next optimizer update after reload with an uninterrupted update exactly, including moment/step restoration. The first speed-check step was measured before this fix and is kept only as a throughput measurement, not as a research checkpoint.

Right-padding trimming and length-bucketed batches preserve the real causal tokens and per-sequence loss normalization. Continuing padded position IDs also avoids Transformers 5's accidental packed-sequence mask. FP32 causal-model loss/gradient equivalence is tested. BF16 kernels and near-tied ranks can change gradients when batch shapes change: the benchmark records those discrepancies, and optimized continuations must not be described as bitwise reproductions of historical runs.

`experiments/benchmark_path_actor.py` measures synthetic padded/trimmed backward passes without updating model weights. The production-step validation log under `outputs/analysis/speed_validation.log` is the stronger evidence for practical speed. Synthetic results alone are not an end-to-end speed guarantee.

The project W&B evaluation runs use explicit `eval-*` names. Predictions and summaries are written locally before W&B upload; evaluations can resume even if logging or the container is interrupted. No official GSM8K test examples are used by this launcher.
