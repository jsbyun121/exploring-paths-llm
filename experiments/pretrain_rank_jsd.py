#!/usr/bin/env python3
"""Single-A100 continued-pretraining pilot for Section 4.4.

This is deliberately a pilot, not a claim that rank-JSD should replace the
proper cross-entropy scoring rule.  It compares CE, standalone rank-JSD, and a
CE+rank-JSD hybrid at a fixed token/update budget while logging validation
perplexity and rank-cap coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import queue
import random
import threading
import time
from pathlib import Path
from typing import Iterable, Iterator

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from RL2.utils.path_losses import rank_shift_jsd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--dataset", default="open-web-math/open-web-math")
    parser.add_argument("--dataset-config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--objective", choices=("ce", "rank_jsd", "ce_rank_jsd"), required=True
    )
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--rank-cap", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-permille", type=int, default=10)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def is_validation_text(text: str, validation_permille: int) -> bool:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 1000
    return bucket < validation_permille


def partition_rows(
    rows: Iterable[dict],
    *,
    text_column: str,
    validation_permille: int,
    validation: bool,
) -> Iterator[str]:
    for row in rows:
        text = row.get(text_column)
        if not isinstance(text, str) or not text.strip():
            continue
        if is_validation_text(text, validation_permille) == validation:
            yield text


def token_blocks(
    texts: Iterable[str], tokenizer, block_size: int, text_batch_size: int = 64
) -> Iterator[torch.Tensor]:
    buffer: list[int] = []
    cursor = 0
    pending_texts: list[str] = []

    def emit(encoded_texts):
        nonlocal buffer, cursor
        for token_ids in encoded_texts:
            buffer.extend(token_ids)
            if tokenizer.eos_token_id is not None:
                buffer.append(tokenizer.eos_token_id)
            while len(buffer) - cursor >= block_size:
                yield torch.tensor(
                    buffer[cursor : cursor + block_size], dtype=torch.long
                )
                cursor += block_size
            if cursor >= block_size * 16:
                buffer = buffer[cursor:]
                cursor = 0

    for text in texts:
        pending_texts.append(text)
        if len(pending_texts) == text_batch_size:
            encoded = tokenizer(
                pending_texts, add_special_tokens=False, padding=False
            )["input_ids"]
            yield from emit(encoded)
            pending_texts.clear()
    if pending_texts:
        encoded = tokenizer(
            pending_texts, add_special_tokens=False, padding=False
        )["input_ids"]
        yield from emit(encoded)


def batches(
    blocks: Iterable[torch.Tensor], batch_size: int
) -> Iterator[torch.Tensor]:
    pending: list[torch.Tensor] = []
    for block in blocks:
        pending.append(block)
        if len(pending) == batch_size:
            yield torch.stack(pending)
            pending.clear()


def prefetch(iterator: Iterable[torch.Tensor], depth: int = 8):
    """Overlap streaming/tokenization on CPU with the current GPU update."""

    work_queue: queue.Queue = queue.Queue(maxsize=depth)
    sentinel = object()

    def producer():
        try:
            for item in iterator:
                work_queue.put(item)
        except BaseException as exc:  # propagate producer failures to the main thread
            work_queue.put(exc)
        finally:
            work_queue.put(sentinel)

    threading.Thread(target=producer, daemon=True).start()
    while True:
        item = work_queue.get()
        if item is sentinel:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def make_streams(args: argparse.Namespace, tokenizer):
    dataset_kwargs = {
        "path": args.dataset,
        "split": args.split,
        "streaming": True,
    }
    if args.dataset_config:
        dataset_kwargs["name"] = args.dataset_config

    train_rows = load_dataset(**dataset_kwargs).shuffle(
        seed=args.seed, buffer_size=args.shuffle_buffer
    )
    validation_rows = load_dataset(**dataset_kwargs)
    block_size = args.sequence_length + 1

    train_texts = partition_rows(
        train_rows,
        text_column=args.text_column,
        validation_permille=args.validation_permille,
        validation=False,
    )
    validation_texts = partition_rows(
        validation_rows,
        text_column=args.text_column,
        validation_permille=args.validation_permille,
        validation=True,
    )
    train_batches = batches(
        token_blocks(train_texts, tokenizer, block_size), args.micro_batch_size
    )
    validation_batches = batches(
        token_blocks(validation_texts, tokenizer, block_size),
        args.micro_batch_size,
    )
    validation_cache = list(
        itertools.islice(validation_batches, args.validation_batches)
    )
    if len(validation_cache) < args.validation_batches:
        raise RuntimeError("Validation partition did not yield enough batches.")
    return iter(prefetch(train_batches)), validation_cache


def causal_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    objective: str,
    rank_cap: int,
    rank_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    metrics: dict[str, float] = {}
    ce = None
    if objective in ("ce", "ce_rank_jsd"):
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
        metrics["train/ce"] = ce.detach().item()

    rank_loss = None
    if objective in ("rank_jsd", "ce_rank_jsd"):
        # Keep the full logits in bf16 to avoid a multi-GB fp32 copy.  Only the
        # top-k slice and JSD arithmetic are promoted to fp32 downstream.
        logsumexp = torch.logsumexp(logits, dim=-1)
        per_token_rank_loss, ranks, in_cap = rank_shift_jsd(
            logits, logsumexp, labels, rank_cap=rank_cap
        )
        rank_loss = per_token_rank_loss.mean()
        metrics["train/rank_jsd"] = rank_loss.detach().item()
        metrics["train/rank_cap_coverage"] = in_cap.float().mean().item()
        metrics["train/rank1_fraction"] = ranks.eq(1).float().mean().item()

    if objective == "ce":
        loss = ce
    elif objective == "rank_jsd":
        loss = rank_loss
    else:
        loss = ce + rank_weight * rank_loss
    metrics["train/loss"] = loss.detach().item()
    return loss, metrics


@torch.no_grad()
def evaluate(model, validation_batches, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    losses = []
    iterator = iter(validation_batches)
    for _ in range(args.validation_batches):
        batch = next(iterator).to("cuda", non_blocking=True)
        inputs, labels = batch[:, :-1], batch[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inputs, use_cache=False).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
            )
        losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    model.train()
    return {
        "validation/ce": mean_loss,
        "validation/perplexity": math.exp(min(mean_loss, 20.0)),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This pilot expects exactly one visible CUDA GPU.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    warmup_steps = max(1, int(args.max_steps * args.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(args.max_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    train_batches, validation_batches = make_streams(args, tokenizer)
    log_path = args.output_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    with log_path.open("a", encoding="utf-8") as log_file:
        for step in range(1, args.max_steps + 1):
            accumulated: dict[str, list[float]] = {}
            for _ in range(args.gradient_accumulation):
                batch = next(train_batches).to("cuda", non_blocking=True)
                inputs, labels = batch[:, :-1], batch[:, 1:]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=inputs, use_cache=False).logits
                    loss, micro_metrics = causal_losses(
                        logits,
                        labels,
                        args.objective,
                        args.rank_cap,
                        args.rank_weight,
                    )
                    scaled_loss = loss / args.gradient_accumulation
                scaled_loss.backward()
                for key, value in micro_metrics.items():
                    accumulated.setdefault(key, []).append(value)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            record = {
                "step": step,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "elapsed_seconds": time.time() - start_time,
                "tokens_seen": (
                    step
                    * args.gradient_accumulation
                    * args.micro_batch_size
                    * args.sequence_length
                ),
            }
            record.update(
                {key: sum(values) / len(values) for key, values in accumulated.items()}
            )
            if step % args.eval_every == 0 or step == args.max_steps:
                record.update(evaluate(model, validation_batches, args))

            if step % args.log_every == 0 or step == 1 or "validation/ce" in record:
                line = json.dumps(record, sort_keys=True)
                print(line, flush=True)
                log_file.write(line + "\n")
                log_file.flush()

    model.save_pretrained(args.output_dir / "latest", safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir / "latest")
    (args.output_dir / "run_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
