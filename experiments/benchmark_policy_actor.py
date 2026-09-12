"""Compare padded/trimmed actor backward on identical inputs, without updates.

torchrun --standalone --nproc_per_node=1 -m experiments.benchmark_path_actor
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from RL2.utils.comm import initialize_global_process_group
from RL2.utils.offloading import load_model_to_device
from RL2.utils.path_batching import trim_right_padding
from RL2.workers import Actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/policy_actor_benchmark.json"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    initialize_global_process_group()
    cfg = OmegaConf.create(json.loads(Path("/workspace/wandb/wandb/run-20260906_132708-ordgy4u7/files/provenance.json").read_text())["config"]["actor"])
    cfg.optimized_batching = False
    torch.manual_seed(17)
    actor = Actor(cfg, True)
    load_model_to_device(actor, "cuda")
    actor.model.train()
    records = []
    for lengths in [(640, 1280), (2048, 3200)]:
        width = 4352
        tokens = actor.tokenizer.encode("Let us solve this problem step by step. The answer is 42. ", add_special_tokens=False)
        states = torch.tensor((tokens * (width // len(tokens) + 2))[:width], device="cuda").repeat(2, 1)
        batch = {"states": states, "actions": states.roll(-1, 1),
                 "position_ids": torch.arange(width, device="cuda").repeat(2, 1),
                 "action_mask": torch.zeros_like(states), "eos_mask": torch.zeros_like(states)}
        for row, length in enumerate(lengths):
            batch["action_mask"][row, 32:length] = 1
            batch["eos_mask"][row, length - 1] = 1
            batch["states"][row, length:] = 0
            batch["position_ids"][row, length:] = 0
        reference = None
        for mode in ["padded", "optimized"]:
            cfg.optimized_batching = mode == "optimized"
            current = batch
            times = []
            for repeat in range(args.repeats + 1):
                actor.model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                logps, entropy = actor.forward_original(current, return_entropy=True)
                loss = -(logps.sum(-1) / current["action_mask"].sum(-1)).mean()
                loss.backward()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if repeat:
                    times.append(elapsed)
            grads = [p.grad.detach().float().cpu() for p in actor.model.parameters() if p.grad is not None]
            if reference is None:
                reference = grads
                relative_error = 0.0
            else:
                delta = sum((a-b).square().sum().item() for a,b in zip(reference, grads))
                norm = sum(a.square().sum().item() for a in reference)
                relative_error = (delta / max(norm, 1e-30)) ** .5
                # BF16 reduction kernels vary with sequence shape.
                # Rank targets are discrete; BF16 kernel shape changes can flip
                # near-tied ranks. Record this separately from FP32 unit checks.
            record = {"lengths": lengths, "mode": mode, "dense_tokens": current["states"].numel(),
                      "seconds": times, "peak_allocated_gb": torch.cuda.max_memory_allocated()/1e9,
                      "loss": loss.item(), "gradient_relative_l2_error": relative_error}
            records.append(record)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)
            del grads
        del reference
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
