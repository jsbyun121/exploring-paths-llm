#!/usr/bin/env python3
"""Resumable, manifest-driven lm-eval runner for retention benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def expand_model(value: str) -> str:
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(f"Unresolved model environment variable: {value}")
    return expanded


def model_fingerprint(model: str, expected_step: int) -> dict:
    path = Path(model)
    if not path.exists():
        return {"kind": "huggingface_id", "value": model}
    config = path / "config.json"
    weights = sorted(path.glob("*.safetensors"))
    if not config.is_file() or not weights:
        raise FileNotFoundError(f"Not a Hugging Face model directory: {path}")
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    training_state = path / "training_state.json"
    recorded_step = None
    if training_state.is_file():
        recorded_step = json.loads(training_state.read_text()).get("step")
    elif (path / "export.json").is_file():
        exported = json.loads((path / "export.json").read_text())
        match = re.search(r"(?:^|/)step(\d+)(?:/|$)", exported.get("checkpoint", ""))
        if match:
            recorded_step = int(match.group(1))
    if recorded_step is not None and int(recorded_step) != expected_step:
        raise ValueError(
            f"Model artifact step {recorded_step} does not match registered step "
            f"{expected_step}: {path}"
        )
    return {
        "kind": "local_hf_model",
        "path": str(path.resolve()),
        "config_sha256": config_hash,
        "recorded_training_step": recorded_step,
        "weights": [
            {"name": item.name, "size": item.stat().st_size, "mtime_ns": item.stat().st_mtime_ns}
            for item in weights
        ],
    }


def result_files(output: Path) -> list[Path]:
    return sorted(output.rglob("results*.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=HERE / "models.json")
    parser.add_argument("--suite", type=Path, default=HERE / "preregistered_suite.json")
    parser.add_argument("--output", type=Path, default=REPO / "outputs/forgetting/lm_eval")
    parser.add_argument("--model-id", action="append", help="Run only these model IDs")
    parser.add_argument("--task", action="append", help="Run only these lm-eval task names")
    parser.add_argument("--include-secondary", action="store_true")
    parser.add_argument("--limit", type=float, help="Smoke-test limit passed to lm-eval")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = json.loads(args.suite.read_text())
    registry = json.loads(args.models.read_text())
    wanted_models = set(args.model_id or [])
    models = [m for m in registry["models"] if not wanted_models or m["id"] in wanted_models]
    missing_ids = wanted_models - {m["id"] for m in models}
    if missing_ids:
        raise ValueError(f"Unknown model IDs: {sorted(missing_ids)}")

    tasks = list(suite["primary_tasks"])
    if args.include_secondary:
        tasks += suite["secondary_tasks"]
    wanted_tasks = set(args.task or [])
    tasks = [t for t in tasks if not wanted_tasks or t["name"] in wanted_tasks]
    missing_tasks = wanted_tasks - {t["name"] for t in tasks}
    if missing_tasks:
        raise ValueError(f"Tasks not enabled by this invocation: {sorted(missing_tasks)}")

    config = suite["lm_eval"]
    failures = []
    for entry in models:
        model = expand_model(entry["model"])
        fingerprint = model_fingerprint(model, int(entry["artifact_step"]))
        if entry.get("artifact_step") != entry.get("training_step"):
            raise ValueError(f"Artifact/training step mismatch for {entry['id']}")
        for task in tasks:
            destination = args.output / entry["id"] / task["name"]
            done = destination / "done.json"
            if done.exists() and not args.force:
                print(f"SKIP complete: {entry['id']} / {task['name']}")
                continue
            if destination.exists() and any(destination.iterdir()) and not args.force:
                raise RuntimeError(f"Incomplete output exists; inspect or pass --force: {destination}")
            destination.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "lm_eval",
                "--model",
                config["model_backend"],
                "--model_args",
                f"pretrained={model},dtype={config['dtype']},trust_remote_code=True",
                "--tasks",
                task["name"],
                "--batch_size",
                str(config["batch_size"]),
                "--device",
                config["device"],
                "--num_fewshot",
                str(config["num_fewshot"]),
                "--seed",
                config["seed"],
                "--output_path",
                str(destination),
            ]
            if config.get("apply_chat_template"):
                command.append("--apply_chat_template")
            if config.get("log_samples"):
                command.append("--log_samples")
            if args.limit is not None:
                command += ["--limit", str(args.limit)]
            manifest = {
                "schema": "catastrophic-forgetting-run/v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": entry,
                "model_fingerprint": fingerprint,
                "task": task,
                "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
                "command": command,
                "limit": args.limit,
            }
            atomic_json(destination / "run_manifest.json", manifest)
            print("RUN", entry["id"], task["name"], flush=True)
            print(" ".join(command), flush=True)
            if args.dry_run:
                continue
            log = destination / "run.log"
            with log.open("a") as stream:
                completed = subprocess.run(
                    command,
                    cwd=REPO,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if completed.returncode or not result_files(destination):
                failure = {
                    "model_id": entry["id"],
                    "task": task["name"],
                    "returncode": completed.returncode,
                    "log": str(log),
                }
                failures.append(failure)
                atomic_json(destination / "failed.json", failure)
                continue
            atomic_json(
                done,
                {
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "result_files": [str(p) for p in result_files(destination)],
                },
            )
    if failures:
        print(json.dumps({"failures": failures}, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
