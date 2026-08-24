from omegaconf import OmegaConf
import random
import torch.distributed as dist
import torch
from transformers import get_scheduler
import wandb

class Trainer:
    
    def __init__(self, config):
        
        OmegaConf.resolve(config)
        self.config = config

        seed = int(getattr(config.trainer, "seed", 0)) + dist.get_rank()
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if dist.get_rank() == 0:
            print(OmegaConf.to_yaml(config))
            if config.trainer.use_wandb:
                wandb.init(
                    project=config.trainer.project,
                    name=config.trainer.experiment_name,
                    config=OmegaConf.to_container(config)
                )
            else:
                wandb.log = lambda *args, **kwargs: None
    
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
