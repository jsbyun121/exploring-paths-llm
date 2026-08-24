#!/usr/bin/env python3
"""Greedy and sampled evaluation for a completed model on one A100."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset
from math_verify import parse, verify
from sglang.srt.entrypoints.engine import Engine
from transformers import AutoTokenizer


logging.getLogger("math_verify.parser").disabled = True
logging.getLogger("math_verify.grader").disabled = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--sample-k", type=int, default=8)
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--request-chunk-size", type=int, default=256)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.88)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-generations", action="store_true")
    return parser.parse_args()


def final_gold(answer: str) -> str:
    match = re.search(r"####\s*(.+?)\s*$", answer, re.DOTALL)
    return match.group(1).strip() if match else answer


def is_correct(gold: str, response: str) -> bool:
    try:
        return bool(verify(parse(final_gold(gold)), parse(response)))
    except Exception:
        return False


def canonical_final(response: str) -> str:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", response)
    if boxed:
        return re.sub(r"\s+", "", boxed[-1])
    hashes = re.findall(r"####\s*([^\n]+)", response)
    if hashes:
        return re.sub(r"\s+", "", hashes[-1])
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/[\d,]+)?", response)
    return numbers[-1].replace(",", "") if numbers else "<unparsed>"


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2))
    return center - margin / denominator, center + margin / denominator


async def generate_requests(engine, requests, chunk_size: int):
    outputs = []
    for start in range(0, len(requests), chunk_size):
        chunk = requests[start : start + chunk_size]
        chunk_outputs = await asyncio.gather(
            *(
                engine.async_generate(
                    input_ids=input_ids,
                    sampling_params=sampling_params,
                    return_logprob=False,
                )
                for input_ids, sampling_params in chunk
            )
        )
        outputs.extend(chunk_outputs)
    return outputs


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Evaluation expects exactly one visible CUDA GPU.")
    if args.sample_k < 1:
        raise ValueError("--sample-k must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.dataset,
        args.dataset_config or None,
        split=args.split,
    )
    if args.max_examples is not None:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = []
    golds = []
    for example in dataset:
        question = example[args.question_column]
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (
                        f"{question}\n\nSolve the problem step by step and put "
                        "the final answer in \\boxed{}."
                    ),
                }
            ],
            add_generation_prompt=True,
            tokenize=True,
        )
        prompts.append(prompt)
        golds.append(str(example[args.answer_column]))

    engine = Engine(
        model_path=args.model,
        dtype="bfloat16",
        tp_size=1,
        mem_fraction_static=args.gpu_memory_fraction,
        port=31000,
    )
    greedy_params = {
        "temperature": 0.0,
        "max_new_tokens": args.max_new_tokens,
        "no_stop_trim": True,
    }
    sample_params = {
        "temperature": args.sample_temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "no_stop_trim": True,
    }

    greedy_requests = [(prompt, greedy_params) for prompt in prompts]
    sample_requests = [
        (prompt, sample_params)
        for prompt in prompts
        for _ in range(args.sample_k)
    ]
    greedy_outputs = asyncio.run(
        generate_requests(engine, greedy_requests, args.request_chunk_size)
    )
    sample_outputs = asyncio.run(
        generate_requests(engine, sample_requests, args.request_chunk_size)
    )

    rows = []
    for index, (gold, greedy_output) in enumerate(zip(golds, greedy_outputs)):
        start = index * args.sample_k
        sampled = sample_outputs[start : start + args.sample_k]
        greedy_text = greedy_output["text"]
        sample_texts = [output["text"] for output in sampled]
        sample_correct = [is_correct(gold, text) for text in sample_texts]
        finals = [canonical_final(text) for text in sample_texts]
        rows.append(
            {
                "index": index,
                "gold": final_gold(gold),
                "greedy_correct": is_correct(gold, greedy_text),
                "sample_correct": sample_correct,
                "pass_at_k": any(sample_correct),
                "unique_sample_answers": len(set(finals)),
                "greedy_tokens": greedy_output["meta_info"]["completion_tokens"],
                "greedy_truncated": (
                    greedy_output["meta_info"]["finish_reason"]["type"]
                    == "length"
                ),
                "sample_tokens": [
                    output["meta_info"]["completion_tokens"] for output in sampled
                ],
                "sample_truncated": [
                    output["meta_info"]["finish_reason"]["type"] == "length"
                    for output in sampled
                ],
                **(
                    {"greedy_text": greedy_text, "sample_texts": sample_texts}
                    if args.save_generations
                    else {}
                ),
            }
        )

    total = len(rows)
    greedy_successes = sum(row["greedy_correct"] for row in rows)
    pass_successes = sum(row["pass_at_k"] for row in rows)
    sampled_correct = sum(sum(row["sample_correct"]) for row in rows)
    sampled_total = total * args.sample_k
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "examples": total,
        "sample_k": args.sample_k,
        "greedy_accuracy": greedy_successes / total,
        "greedy_accuracy_wilson95": wilson_interval(greedy_successes, total),
        "sample_accuracy": sampled_correct / sampled_total,
        "sample_accuracy_wilson95": wilson_interval(
            sampled_correct, sampled_total
        ),
        "pass_at_k": pass_successes / total,
        "pass_at_k_wilson95": wilson_interval(pass_successes, total),
        "mean_unique_sample_answers": sum(
            row["unique_sample_answers"] for row in rows
        )
        / total,
        "mean_greedy_tokens": sum(row["greedy_tokens"] for row in rows) / total,
        "mean_sample_tokens": sum(
            sum(row["sample_tokens"]) for row in rows
        )
        / sampled_total,
        "greedy_truncation_rate": sum(
            row["greedy_truncated"] for row in rows
        )
        / total,
        "sample_truncation_rate": sum(
            sum(row["sample_truncated"]) for row in rows
        )
        / sampled_total,
        "seed": args.seed,
    }

    with (args.output_dir / "predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    if hasattr(engine, "shutdown"):
        engine.shutdown()


if __name__ == "__main__":
    main()
