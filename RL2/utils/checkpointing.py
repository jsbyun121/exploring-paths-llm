import os
import json
import pickle
import random
import shutil
import uuid
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict
)
from transformers import AutoModelForSequenceClassification
from RL2.utils.offloading import model_offloading_manager, load_optimizer_to_device

@model_offloading_manager
def get_state_dict(worker, full_state_dict=False):

    options = StateDictOptions(
        full_state_dict=full_state_dict,
        cpu_offload=True
    )
    return get_model_state_dict(worker.model, options=options)

def get_worker_ckpt(worker):
    
    if not hasattr(worker, "state_dict"):
        worker.state_dict = get_state_dict(worker)
    return {
        "model": worker.state_dict,
        "optimizer": worker.optimizer.state_dict(),
        "scheduler": worker.scheduler.state_dict()
    }

def get_ckpt(trainer, workers, step):

    ckpt = {
        "step": step,
        "dataloader": trainer.train_dataloader.state_dict()
    }

    for idx, worker in enumerate(workers):
        if hasattr(worker, "model"):
            ckpt[f"worker{idx}"] = get_worker_ckpt(worker)

    return ckpt


def completed_checkpoints(save_dir):
    """Only numbered, published directories with DCP completion metadata."""
    return sorted(
        (p for p in Path(save_dir).glob("step*")
         if p.is_dir() and p.name[4:].isdigit() and (p / ".metadata").is_file()),
        key=lambda p: int(p.name[4:]),
    )


def capture_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_checkpoint(path):
    """Check DCP's index and that every referenced byte range was written."""
    metadata = dcp.FileSystemReader(path).read_metadata()
    for info in metadata.storage_data.values():
        file = Path(path) / info.relative_path
        if not file.is_file() or file.stat().st_size < info.offset + info.length:
            raise RuntimeError(f"Incomplete checkpoint shard: {file}")
    return metadata


def optimizer_load_template(optimizer, metadata, worker_key):
    """Allocate lazy Adam moments from the legacy DCP index before loading.

    DCP only reads leaves present in its destination template. A newly created
    AdamW has an empty state dict; passing it directly silently drops all saved
    moments. Preserve the historical numeric parameter IDs and CPU storage.
    """
    template = optimizer.state_dict()
    prefix = f"{worker_key}.optimizer.state."
    for key, item in metadata.state_dict_metadata.items():
        if not key.startswith(prefix):
            continue
        path = metadata.planner_data[key]
        if len(path) != 5 or path[:3] != (worker_key, "optimizer", "state"):
            raise ValueError(f"Unsupported optimizer checkpoint layout: {path}")
        parameter_id, field = path[3:]
        # DCP's planner metadata stringifies dictionary keys, whereas
        # Optimizer.load_state_dict maps integer IDs from param_groups.
        parameter_id = int(parameter_id)
        value = (torch.empty(item.size, dtype=item.properties.dtype, device="cpu")
                 if isinstance(item, dcp.metadata.TensorStorageMetadata) else None)
        template["state"].setdefault(parameter_id, {})[field] = value
    return template

@model_offloading_manager
def load_worker_ckpt(worker, ckpt):

    set_model_state_dict(
        worker.model, ckpt["model"]
    )
    worker.optimizer.load_state_dict(ckpt["optimizer"])
    if any(not isinstance(parameter, torch.Tensor) for parameter in worker.optimizer.state):
        raise RuntimeError("Optimizer checkpoint IDs did not map to model parameters")
    worker.scheduler.load_state_dict(ckpt["scheduler"])
    load_optimizer_to_device(worker, "cpu")
    counters = [float(state["step"]) for state in worker.optimizer.state.values() if "step" in state]
    print(f"Restored optimizer states: {len(worker.optimizer.state)}; "
          f"step range: {min(counters) if counters else None}..{max(counters) if counters else None}", flush=True)

def load_ckpt(trainer, workers):

    checkpoint_id = trainer.config.trainer.load_ckpt_from
    
    if checkpoint_id is None:
        return 0

    if checkpoint_id == "latest":
        # A failed write can leave a large step directory without DCP's
        # completion metadata. Never select such a partial checkpoint merely
        # because it has the highest step number.
        save_dirs = completed_checkpoints(trainer.config.trainer.save_dir)
        if not save_dirs:
            raise FileNotFoundError("Resume requested, but no completed checkpoint exists")
        checkpoint_id = str(save_dirs[-1])
    
    metadata = validate_checkpoint(checkpoint_id)
    ckpt = get_ckpt(trainer, workers, 0)
    for idx, worker in enumerate(workers):
        if hasattr(worker, "model"):
            ckpt[f"worker{idx}"]["optimizer"] = optimizer_load_template(
                worker.optimizer, metadata, f"worker{idx}")
    if "rng_states" in metadata.state_dict_metadata:
        ckpt["rng_states"] = b""
    dcp.load(ckpt, checkpoint_id=checkpoint_id)
    trainer.train_dataloader.load_state_dict(ckpt["dataloader"])
    for idx, worker in enumerate(workers):
        if hasattr(worker, "model"):
            load_worker_ckpt(worker, ckpt[f"worker{idx}"])
        elif worker is not None:
            if (worker.device_mesh["tp"].get_local_rank() == 0
                    and getattr(worker.config, "release_memory_for_training", True)):
                worker.llm.release_memory_occupation()
            worker.update(workers[0], ckpt["step"])
    if "rng_states" in ckpt:
        restore_rng_state(pickle.loads(ckpt["rng_states"])[dist.get_rank()])
    else:
        warnings.warn("Legacy checkpoint has no RNG state; this resume is a new sampling branch")
    trainer.loaded_checkpoint = str(checkpoint_id)
    if hasattr(trainer, "record_resume"):
        trainer.record_resume(checkpoint_id, ckpt["step"])
    print(f"Loaded checkpoint {checkpoint_id}; completed step {ckpt['step']}", flush=True)
    return ckpt["step"]

def save_ckpt(trainer, workers, step, force=False):

    frequency = trainer.config.trainer.save_freq
    if not force and (frequency is None or step % frequency != 0):
        return
    root = Path(trainer.config.trainer.save_dir)
    target = root / f"step{step}"
    if target.is_dir():
        validate_checkpoint(target)
        return
    root.mkdir(parents=True, exist_ok=True)
    temporary = [str(root / f".step{step}.{uuid.uuid4().hex}.tmp") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(temporary, src=0)
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, capture_rng_state())
    ckpt = get_ckpt(trainer, workers, step)
    ckpt["rng_states"] = pickle.dumps(states)
    dcp.save(
        ckpt, checkpoint_id=temporary[0]
    )
    dist.barrier()
    if dist.get_rank() == 0:
        validate_checkpoint(temporary[0])
        (Path(temporary[0]) / "recovery.json").write_text(json.dumps({
            "step": step, "parent_checkpoint": getattr(trainer, "loaded_checkpoint", None),
            "wandb_run_id": getattr(trainer, "wandb_run_id", None),
            "rng_saved": True,
        }, indent=2))
        os.replace(temporary[0], target)
        keep = int(getattr(trainer.config.trainer, "keep_checkpoints", 0))
        if keep > 0:
            for old in completed_checkpoints(root)[:-keep]:
                shutil.rmtree(old)
        print(f"Published recovery checkpoint {target}", flush=True)
    dist.barrier()

def save_model(trainer, worker, rm=False):

    save_dir = trainer.config.trainer.save_dir
    if trainer.config.trainer.save_freq is not None:
        save_dir += "/latest"
    state_dict = get_state_dict(
        worker, full_state_dict=True
    )
    if dist.get_rank() == 0:
        target = Path(save_dir)
        staged = (target.with_name(f".latest.{uuid.uuid4().hex}.tmp")
                  if trainer.config.trainer.save_freq is not None else target)
        staged.mkdir(parents=True, exist_ok=True)
        worker.tokenizer.save_pretrained(staged)
        # unwrap the model
        model_to_save = worker.model.module
        if rm:
            # For RM, we load token classification model for simplicity 
            # but save sequence classification model for compatibility.
            with torch.device("meta"):
                model_to_save = AutoModelForSequenceClassification.from_config(
                    model_to_save.config
                )
        model_to_save.save_pretrained(
            staged, state_dict=state_dict
        )
        completion = {"step": getattr(trainer, "completed_step", None),
                      "wandb_run_id": getattr(trainer, "wandb_run_id", None)}
        (staged / "training_state.json").write_text(json.dumps(completion, indent=2))
        if staged != target:
            previous = target.with_name(f".latest.previous.{uuid.uuid4().hex}")
            if target.exists():
                os.replace(target, previous)
            try:
                os.replace(staged, target)
            except BaseException:
                if previous.exists():
                    os.replace(previous, target)
                raise
            if previous.exists():
                shutil.rmtree(previous)
        marker = Path(trainer.config.trainer.save_dir) / "completed.json"
        temporary_marker = marker.with_suffix(".json.tmp")
        temporary_marker.write_text(json.dumps(completion, indent=2))
        os.replace(temporary_marker, marker)

    dist.barrier()
