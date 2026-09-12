import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from RL2.utils.path_batching import length_bucketed_partitions, trim_right_padding, causal_position_ids
from RL2.utils.path_losses import rank_shift_jsd


class PaddingSemanticsTest(unittest.TestCase):
    def test_partitions_cover_once_and_bound_actual_padded_tokens(self):
        lengths = [11, 2, 9, 1, 7, 3, 14]
        parts = length_bucketed_partitions(lengths, 16)
        self.assertEqual(sorted(sum(parts, [])), list(range(len(lengths))))
        for part in parts:
            self.assertLessEqual(max(lengths[i] for i in part) * len(part), 16)
        with self.assertRaises(ValueError):
            length_bucketed_partitions([17], 16)

    def test_causal_padding_and_bucketed_accumulation_preserve_loss_and_gradients(self):
        torch.manual_seed(12)
        torch.set_num_threads(2)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, attention_dropout=0.0))
        model.set_attn_implementation("sdpa")
        model.train()
        lengths, width = [5, 9, 13], 20
        batch = {"states": torch.randint(0, 32, (3,width)),
                 "actions": torch.randint(0,32,(3,width)),
                 "position_ids": torch.arange(width).repeat(3,1),
                 "eos_mask": torch.zeros(3,width,dtype=torch.long),
                 "action_mask": torch.zeros(3,width)}
        for i,n in enumerate(lengths):
            batch["eos_mask"][i,n-1] = 1
            batch["action_mask"][i,2:n] = 1
            batch["position_ids"][i,n:] = 0
        results = []
        for optimized in [False, True]:
            model.zero_grad(set_to_none=True)
            parts = length_bucketed_partitions(lengths, 16) if optimized else [[0,1,2]]
            value = 0
            for part in parts:
                mb = {k:v[part] for k,v in batch.items()}
                if optimized:
                    mb = trim_right_padding(mb)
                logits = model(input_ids=mb["states"], position_ids=(causal_position_ids(mb)
                               if optimized else mb["position_ids"]), use_cache=False).logits
                losses,_,_ = rank_shift_jsd(logits, logits.logsumexp(-1), mb["actions"], rank_cap=16)
                loss = ((losses*mb["action_mask"]).sum(-1)/mb["action_mask"].sum(-1)).sum()/3
                value += loss.item()
                loss.backward()
            results.append((value,[p.grad.clone() for p in model.parameters() if p.grad is not None]))
        self.assertAlmostEqual(results[0][0], results[1][0], places=7)
        for a,b in zip(results[0][1],results[1][1]):
            torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-4)


if __name__ == "__main__":
    unittest.main()
