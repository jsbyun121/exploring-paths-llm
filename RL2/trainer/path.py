"""Trainer for positive-only token objectives on verified trajectories."""

import hydra
import torch.distributed as dist
from tqdm import tqdm

from RL2.datasets import RLDataset, get_dataloader
from RL2.trainer import Trainer
from RL2.utils.algorithms import compute_spo_adv
from RL2.utils.checkpointing import load_ckpt, save_ckpt, save_model
from RL2.utils.comm import initialize_global_process_group
from RL2.utils.logging import time_logger
from RL2.workers import Actor, Rollout


class PathTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)
        self.actor = Actor(config.actor, True)
        self.train_dataloader = self.get_dataloader(True)
        self.test_dataloader = self.get_dataloader(False)
        self.actor.scheduler = self.prepare_scheduler(self.actor)
        self.rollout = Rollout(config.rollout)

    def get_dataloader(self, train: bool):
        data_config = self.config.train_data if train else self.config.test_data
        dataset = RLDataset(data_config, self.actor.tokenizer)
        return get_dataloader(
            dataset,
            data_config.prompts_per_rollout if train else len(dataset),
        )

    @time_logger("compute_advantages")
    def compute_advantages(self, tensor_dict, cu_seqs, step):
        compute_spo_adv(
            tensor_dict,
            cu_seqs,
            self.config.train_data.responses_per_prompt,
        )

    def train(self):
        step = load_ckpt(self, (self.actor, None, self.rollout))
        max_steps = self.config.trainer.max_steps
        stop = max_steps is not None and step >= max_steps

        for epoch in range(
            step // len(self.train_dataloader), self.config.trainer.n_epochs
        ):
            if stop:
                break
            for data_list in tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=(dist.get_rank() != 0),
                initial=step % len(self.train_dataloader),
            ):
                step += 1
                tensor_dict, cu_seqs = self.rollout(data_list, True, step)

                if dist.get_rank() == 0:
                    self.compute_advantages(tensor_dict, cu_seqs, step)

                self.actor.update_path(tensor_dict, step)
                save_ckpt(self, (self.actor, None), step)
                self.rollout.update(self.actor, step)

                test_freq = self.config.trainer.test_freq
                if test_freq is not None and step % test_freq == 0:
                    for test_data_list in self.test_dataloader:
                        self.rollout(test_data_list, False, step)

                if max_steps is not None and step >= max_steps:
                    stop = True
                    break

        save_model(self, self.actor)


@hydra.main(config_path="config", config_name="path", version_base=None)
def main(config):
    initialize_global_process_group()
    trainer = PathTrainer(config)
    trainer.train()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
