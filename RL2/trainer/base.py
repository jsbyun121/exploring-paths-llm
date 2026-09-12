from omegaconf import OmegaConf
import random
import numpy as np
import torch.distributed as dist
import torch
from transformers import get_scheduler
import wandb
import json
import hashlib
import signal
import subprocess
import zipfile
from pathlib import Path

class Trainer:
    
    def __init__(self, config):
        
        OmegaConf.resolve(config)
        self.config = config
        self.stop_requested = False
        self.wandb_run_id = None

        seed = int(getattr(config.trainer, "seed", 0)) + dist.get_rank()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if dist.get_rank() == 0:
            print(OmegaConf.to_yaml(config))
            if config.trainer.use_wandb:
                run = wandb.init(
                    project=config.trainer.project,
                    name=config.trainer.experiment_name,
                    config=OmegaConf.to_container(config),
                    group=config.trainer.experiment_name,
                    job_type="train",
                )
                self.wandb_run_id = run.id
                self.record_provenance(run)
            else:
                wandb.log = lambda *args, **kwargs: None

    def _request_stop(self, signum, frame):
        self.stop_requested = True
        print("Stop requested; finishing the current update and saving recovery state.", flush=True)

    def enable_graceful_stop(self):
        signal.signal(signal.SIGTERM, self._request_stop)
        signal.signal(signal.SIGINT, self._request_stop)

    def should_stop(self):
        stop_file = getattr(self.config.trainer, "stop_file", None)
        local_stop = self.stop_requested or bool(stop_file and Path(stop_file).exists())
        flag = torch.tensor(int(local_stop), device=torch.cuda.current_device())
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def record_resume(self, checkpoint, step):
        if dist.get_rank() == 0 and self.wandb_run_id:
            marker = Path(checkpoint) / "recovery.json"
            parent = json.loads(marker.read_text()) if marker.exists() else {}
            wandb.config.update({"lineage": {"parent_checkpoint": str(checkpoint),
                                            "parent_step": step,
                                            "parent_run_id": parent.get("wandb_run_id"),
                                            "logical_experiment": self.config.trainer.experiment_name}},
                                allow_val_change=True)

    def record_provenance(self, run):
        root = Path(__file__).resolve().parents[2]
        hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for folder in [root / "RL2", root / "envs", root / "experiments"]
                  for p in folder.rglob("*") if p.suffix in {".py", ".sh", ".yaml"}}
        provenance = {"source_sha256": hashes, "torch": torch.__version__,
                      "config": OmegaConf.to_container(self.config, resolve=True)}
        try:
            provenance["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            (Path(run.dir) / "working-tree.patch").write_bytes(subprocess.check_output(
                ["git", "diff", "HEAD", "--", "RL2", "envs", "experiments"], cwd=root))
        except subprocess.CalledProcessError:
            pass
        (Path(run.dir) / "provenance.json").write_text(json.dumps(provenance, indent=2))
        with zipfile.ZipFile(Path(run.dir) / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for relative in hashes:
                archive.write(root / relative, relative)
            for name in ["pyproject.toml", "uv.lock"]:
                if (root / name).exists():
                    archive.write(root / name, name)
        for name in ["provenance.json", "working-tree.patch", "source.zip"]:
            if (Path(run.dir) / name).exists():
                run.save(str(Path(run.dir) / name), base_path=run.dir, policy="now")
    
    def prepare_scheduler(self, worker):

        configured_max_steps = getattr(self.config.trainer, "max_steps", None)
        rollout_steps = (
            configured_max_steps
            if configured_max_steps is not None
            else self.config.trainer.n_epochs * len(self.train_dataloader)
        )
        num_training_steps = rollout_steps * getattr(
            worker.config, "update_per_rollout", 1
        )
        num_warmup_steps = int(worker.config.warmup_ratio * num_training_steps)

        return get_scheduler(
            worker.config.scheduler,
            worker.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
