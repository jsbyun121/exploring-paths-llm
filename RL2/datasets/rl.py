import copy
from RL2.datasets.base import BaseDataset
import re


class RLDataset(BaseDataset):

    def apply_chat_template(self, messages, add_generation_prompt):
        """Render prompts with optional model-specific template settings."""

        template_kwargs = {}
        if getattr(self.config, "enable_thinking", False):
            template_kwargs["enable_thinking"] = True
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **template_kwargs,
        )

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        data = {}

        extra_info = ex.get("extra_info", {})
        extra_info["idx"] = idx
        data["extra_info"] = extra_info

        if "prompt" in ex.keys():
            data["prompt"] = ex["prompt"]
        elif "messages" in ex.keys():
            data["prompt"] = self.apply_chat_template(
                ex["messages"],
                add_generation_prompt=True,
            )
        elif "question" in ex.keys():
            # GSM8K format - add instruction about answer format
            question_with_instruction = (
                f"{ex['question']}\n\n"
                "Solve this step by step. Write your final numerical answer after #### on a new line."
            )
            data["prompt"] = self.apply_chat_template(
                [{"role": "user", "content": question_with_instruction}],
                add_generation_prompt=True,
            )
            data["extra_info"]["answer"] = re.search(r"####(.*)", ex["answer"]).group(1).strip()

        

        return data

    def collate_fn(self, batch):
        return [
            copy.deepcopy(ex)
            for ex in batch
            for _ in range(self.config.responses_per_prompt)
        ]
