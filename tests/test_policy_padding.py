import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from RL2.workers.actor import Actor
from RL2.utils.policy_losses import policy_loss

class PolicyPaddingTest(unittest.TestCase):
    def test_real_forward_restores_width_and_preserves_gradients(self):
        torch.manual_seed(5);torch.set_num_threads(2)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,attention_dropout=0.))
        model.set_attn_implementation('sdpa')
        mesh=SimpleNamespace(size=lambda:1)
        worker=SimpleNamespace(model=model,config=SimpleNamespace(optimized_batching=False,temperature=1.),device_mesh={k:mesh for k in ['sp','tp','dp']})
        worker.compute_entropy=lambda z,l:-(z.softmax(-1)*(z-l.unsqueeze(-1))).sum(-1)
        b={'states':torch.randint(0,32,(2,16)),'actions':torch.randint(0,32,(2,16)),'position_ids':torch.arange(16).repeat(2,1),'action_mask':torch.zeros(2,16),'eos_mask':torch.zeros(2,16)}
        for i,n in enumerate([5,9]):b['action_mask'][i,2:n]=1;b['eos_mask'][i,n-1]=1;b['position_ids'][i,n:]=0
        with patch('RL2.workers.actor.compute_logsumexp',lambda z,m:z.logsumexp(-1)),patch('RL2.workers.actor.gather_action_logits',lambda z,a,m:z.gather(-1,a.unsqueeze(-1)).squeeze(-1)):
            old=Actor.forward_original.__wrapped__(worker,b).detach()
            for method in ['dr_grpo','sapo']:
                results=[]
                for optimized in [False,True]:
                    worker.config.optimized_batching=optimized;model.zero_grad(set_to_none=True)
                    logp,ent=Actor.forward_original.__wrapped__(worker,b,True)
                    self.assertEqual(logp.shape,b['action_mask'].shape);self.assertEqual(ent.shape,logp.shape)
                    loss=policy_loss(method,logp,old,torch.tensor([[1.],[-1.]]).expand_as(logp),b['action_mask'],total_sequences=2,max_completion_length=16)
                    loss.backward();results.append([p.grad.clone() for p in model.parameters() if p.grad is not None])
                for x,y in zip(*results):torch.testing.assert_close(x,y,atol=2e-7,rtol=3e-4)

if __name__=='__main__':unittest.main()
