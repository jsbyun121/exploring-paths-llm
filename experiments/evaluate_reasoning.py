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
import os
import hashlib
from pathlib import Path

import torch
from datasets import load_dataset
from math_verify import parse, verify
from sglang.srt.entrypoints.engine import Engine
from transformers import AutoTokenizer
from envs.gsm8k import verify_answer


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
    parser.add_argument("--prompt-style", choices=["training", "boxed"], default="training")
    parser.add_argument("--wandb-project")
    parser.add_argument("--run-name")
    parser.add_argument("--stop-file", type=Path)
    return parser.parse_args()


def final_gold(answer: str) -> str:
    match = re.search(r"####\s*(.+?)\s*$", answer, re.DOTALL)
    return match.group(1).strip() if match else answer


def is_correct(gold: str, response: str, prompt_style="training") -> bool:
    try:
        if prompt_style == "training":
            return verify_answer(response, final_gold(gold))
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


def log_evaluation(args, manifest, summary):
    marker = args.output_dir / "wandb_run.json"
    if not args.wandb_project or marker.exists():
        return
    import wandb
    with wandb.init(project=args.wandb_project,
                    name=args.run_name or f"eval-{Path(args.model).name}",
                    group="validation-pass8", job_type="evaluation", config=manifest) as run:
        run.log({f"eval/{k}": v for k,v in summary.items() if isinstance(v, (int, float))})
        for name in ["summary.json", "predictions.jsonl", "manifest.json"]:
            run.save(str(args.output_dir / name), base_path=str(args.output_dir), policy="now")
        identifier = {"id": run.id, "url": run.url}
    marker.write_text(json.dumps(identifier, indent=2))


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
    prompt_hashes = []
    for example in dataset:
        question = example[args.question_column]
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (f"{question}\n\nSolve this step by step. Write your final numerical answer after #### on a new line."
                                if args.prompt_style == "training" else (
                        f"{question}\n\nSolve the problem step by step and put "
                        "the final answer in \\boxed{}."
                    )),
                }
            ],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        prompts.append(prompt)
        golds.append(str(example[args.answer_column]))
        prompt_hashes.append(hashlib.sha256(json.dumps(prompt).encode()).hexdigest())

    if not prompts:
        raise ValueError("Evaluation dataset is empty")
    manifest = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()
                if k not in {"wandb_project", "run_name", "output_dir", "stop_file"}}
    manifest["prompt_hashes"] = prompt_hashes
    manifest["gold_hash"] = hashlib.sha256(json.dumps(golds).encode()).hexdigest()
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing evaluation uses a different model/data/prompt/sampling configuration")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    predictions_path = args.output_dir / "predictions.jsonl"
    rows = []
    if predictions_path.exists():
        # Only a final unterminated line may be discarded after interruption.
        data = predictions_path.read_bytes()
        end = data.rfind(b"\n") + 1
        rows = [json.loads(line) for line in data[:end].splitlines()]
        if any(row["index"] != i or row["prompt_sha256"] != prompt_hashes[i]
               for i,row in enumerate(rows)):
            raise ValueError("Prediction file is not a valid ordered evaluation prefix")
        if end != len(data):
            with predictions_path.open("r+b") as f:
                f.truncate(end)
    if len(rows) == len(prompts) and (args.output_dir / "summary.json").exists():
        log_evaluation(args, manifest, json.loads((args.output_dir / "summary.json").read_text()))
        print(f"Already evaluated {len(rows)} examples: {args.output_dir}", flush=True)
        return

    engine = Engine(
        model_path=args.model,
        dtype="bfloat16",
        tp_size=1,
        mem_fraction_static=args.gpu_memory_fraction,
        port=31000,
        random_seed=args.seed,
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

    examples_per_chunk = max(1, args.request_chunk_size // (args.sample_k + 1))
    def outputs_by_example():
        # Drive Engine's own loop; a separate asyncio.run strands its IPC tasks.
        for start in range(len(rows), len(prompts), examples_per_chunk):
            if args.stop_file and args.stop_file.exists():
                return
            stop = min(start + examples_per_chunk, len(prompts))
            requests = []
            for index in range(start, stop):
                requests.append((prompts[index], greedy_params))
                requests.extend((prompts[index], {**sample_params,
                    "sampling_seed": (args.seed + 1000003 * index + j) % (2**31)})
                    for j in range(args.sample_k))
            outputs = engine.loop.run_until_complete(generate_requests(engine, requests, args.request_chunk_size))
            for offset, index in enumerate(range(start, stop)):
                block = outputs[offset*(args.sample_k+1):(offset+1)*(args.sample_k+1)]
                yield index, block[0], block[1:]

    for index, greedy_output, sampled in outputs_by_example():
        gold = golds[index]
        greedy_text = greedy_output["text"]
        sample_texts = [output["text"] for output in sampled]
        sample_correct = [is_correct(gold, text, args.prompt_style) for text in sample_texts]
        finals = [canonical_final(text) for text in sample_texts]
        rows.append(
            {
                "index": index,
                "prompt_sha256": prompt_hashes[index],
                "gold": final_gold(gold),
                "greedy_correct": is_correct(gold, greedy_text, args.prompt_style),
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
        with predictions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(f"Evaluation saved {len(rows)}/{len(prompts)} examples", flush=True)

    if len(rows) < len(prompts):
        engine.shutdown()
        print(f"Evaluation paused with {len(rows)} saved examples; rerun to resume.", flush=True)
        raise SystemExit(75)
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
        "prompt_style": args.prompt_style,
        "max_new_tokens": args.max_new_tokens,
        "sample_temperature": args.sample_temperature,
        "top_p": args.top_p,
        "greedy_accuracy": greedy_successes / total,
        "greedy_accuracy_wilson95": wilson_interval(greedy_successes, total),
        "sample_accuracy": sampled_correct / sampled_total,
        # Samples of the same problem are correlated; use problem-level paired
        # bootstrap in compare_evaluations.py instead of an IID-sample interval.
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

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    if hasattr(engine, "shutdown"):
        engine.shutdown()
    log_evaluation(args, manifest, summary)


if __name__ == "__main__":
    main()
