"""Bounded JSD backward throughput sweep; no updates, checkpoints or rollouts.

This isolates actor activation memory, not total training-process memory.
Compare identical 12-sequence optimizer batches with different microbatches.
"""
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from RL2.utils.comm import initialize_global_process_group
from RL2.utils.offloading import load_model_to_device
from RL2.workers import Actor


def measure(actor, width, microbatch):
    tokens = actor.tokenizer.encode('Let us solve the problem step by step. The answer is 42. ', add_special_tokens=False)
    states = torch.tensor((tokens * (width // len(tokens) + 1))[:width], device='cuda').repeat(microbatch, 1)
    mask = torch.ones_like(states)
    mask[:, :32] = 0
    batch = dict(states=states, actions=states.roll(-1, 1),
                 position_ids=torch.arange(width, device='cuda').repeat(microbatch, 1),
                 action_mask=mask, eos_mask=torch.zeros_like(states))
    batch['eos_mask'][:, -1] = 1
    times = []
    for repeat in range(3):
        actor.model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(12 // microbatch):
            losses, entropy, ranks, in_cap = actor.forward_path(batch)
            loss = (losses.sum(-1) / mask.sum(-1)).sum() / 12
            loss.backward()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            del losses, entropy, ranks, in_cap, loss
        torch.cuda.synchronize()
        if repeat:
            times.append(time.perf_counter() - start)
    return dict(width=width, microbatch=microbatch, sequences=12,
                padded_token_budget=width * microbatch, seconds=times,
                tokens_per_second=12 * width / (sum(times) / len(times)),
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)


def main():
    initialize_global_process_group()
    torch.manual_seed(17)
    cfg = OmegaConf.load('experiments/a100/long_comparison/jsd.yaml').actor
    actor = Actor(cfg, True)
    load_model_to_device(actor, 'cuda')
    actor.model.train()
    records = []
    output = Path('outputs/analysis/jsd_microbatch_benchmark.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    for width in (1536, 4096):
        for microbatch in (1, 2, 3):
            try:
                record = measure(actor, width, microbatch)
            except torch.OutOfMemoryError:
                actor.model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                record = dict(width=width, microbatch=microbatch, status='oom')
            records.append(record)
            output.write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
