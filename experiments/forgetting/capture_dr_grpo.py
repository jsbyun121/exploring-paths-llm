#!/usr/bin/env python3
"""Capture selected Dr.GRPO DCP milestones as actor-only HF models on CPU."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from RL2.utils.checkpointing import validate_checkpoint


REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs"
DEFAULT_RUN = OUTPUTS / "long/dr_grpo"
DEFAULT_CAPTURE = OUTPUTS / "forgetting"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def protect_checkpoint(source: Path, destination: Path) -> None:
    """Hard-link an immutable, completed DCP so retention cleanup cannot race us."""
    if destination.exists():
        validate_checkpoint(destination)
        return
    validate_checkpoint(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    stage.mkdir()
    try:
        for item in source.iterdir():
            if item.is_file():
                os.link(item, stage / item.name)
        validate_checkpoint(stage)
        os.replace(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def export_step(run: Path, capture: Path, step: int) -> bool:
    source = run / f"step{step}"
    model_root = capture / "models" / f"dr_grpo_step{step}"
    model = model_root / "best"
    if (model / "config.json").is_file():
        state = json.loads((model / "training_state.json").read_text())
        if state["step"] != step:
            raise ValueError(f"Existing export has wrong step: {model}")
        return True
    if not (source / ".metadata").is_file():
        return False

    protected = capture / "recovery/dr_grpo" / f"step{step}"
    protect_checkpoint(source, protected)
    model_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "experiments.export_path_checkpoint",
        str(protected),
        str(model),
    ]
    print(f"{now()} exporting Dr.GRPO step{step}", flush=True)
    subprocess.run(command, cwd=REPO, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    atomic_json(model / "training_state.json", {"step": step, "family": "dr_grpo"})
    atomic_json(model_root / "capture.json", {
        "schema": "catastrophic-forgetting-capture/v1",
        "family": "dr_grpo",
        "step": step,
        "source_checkpoint": str(source.resolve()),
        "protected_checkpoint": str(protected.resolve()),
        "model": str(model.resolve()),
        "captured_at": now(),
        "actor_weights_only": True,
    })
    # The actor-only export is complete; release our extra DCP hard links. This
    # never removes or changes the trainer-owned source checkpoint.
    shutil.rmtree(protected)
    print(f"{now()} captured Dr.GRPO step{step}: {model}", flush=True)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--steps", type=int, nargs="+", default=[100, 120, 180, 220])
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sorted(set(args.steps)) != args.steps or any(step <= 0 for step in args.steps):
        raise ValueError("--steps must be unique, increasing positive integers")
    args.capture_root.mkdir(parents=True, exist_ok=True)
    with (args.capture_root / ".capture-dr-grpo.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            captured = []
            for step in args.steps:
                if export_step(args.run, args.capture_root, step):
                    captured.append(step)
            completed = args.run / "long_completed.json"
            if completed.exists():
                final_step = int(json.loads(completed.read_text())["step"])
                unavailable = [step for step in args.steps if step > final_step]
                atomic_json(args.capture_root / "capture_status.json", {
                    "checked_at": now(),
                    "captured_steps": captured,
                    "unavailable_steps": unavailable,
                    "training_final_step": final_step,
                    "complete": all(step in captured or step in unavailable for step in args.steps),
                })
                if all(step in captured or step in unavailable for step in args.steps):
                    return 0
            else:
                atomic_json(args.capture_root / "capture_status.json", {
                    "checked_at": now(),
                    "captured_steps": captured,
                    "waiting_for": [step for step in args.steps if step not in captured],
                    "complete": False,
                })
            if args.once:
                return 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
