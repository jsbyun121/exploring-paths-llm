import copy
from RL2.datasets.base import BaseDataset
import re


class RLDataset(BaseDataset):

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        data = {}

        extra_info = ex.get("extra_info", {})
        extra_info["idx"] = idx
        data["extra_info"] = extra_info

        if "prompt" in ex.keys():
            data["prompt"] = ex["prompt"]
        elif "messages" in ex.keys():
            data["prompt"] = self.tokenizer.apply_chat_template(
                ex["messages"],
                add_generation_prompt=True,
                tokenize=False
            )
        elif "question" in ex.keys():
            # GSM8K format - add instruction about answer format
            question_with_instruction = (
                f"{ex['question']}\n\n"
                "Solve this step by step. Write your final numerical answer after #### on a new line."
            )
            data["prompt"] = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": question_with_instruction}],
                add_generation_prompt=True,
                tokenize=False
            )
            data["extra_info"]["answer"] = re.search(r"####(.*)", ex["answer"]).group(1).strip()

        

        return data

    def collate_fn(self, batch):
        return [
            copy.deepcopy(ex)
            for ex in batch
            for _ in range(self.config.responses_per_prompt)
        ]