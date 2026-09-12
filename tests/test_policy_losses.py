import unittest
import torch
from RL2.utils.policy_losses import policy_loss

class PolicyLossTests(unittest.TestCase):
    def test_drgrpo_fixed_length_no_batch_length_bias(self):
        logp=torch.zeros((2,4),requires_grad=True)
        mask=torch.tensor([[1,0,0,0],[1,1,1,0]])
        adv=torch.tensor([[1.]*4,[-1.]*4])
        loss=policy_loss('dr_grpo',logp,logp.detach(),adv,mask,total_sequences=2,max_completion_length=4)
        loss.backward()
        torch.testing.assert_close(logp.grad[0,0],-logp.grad[1,0])
        self.assertAlmostEqual(loss.item(),.25)
        self.assertEqual(logp.grad[0,1].item(),0)

    def test_sapo_matches_analytic_gradient_and_sign(self):
        z=torch.tensor([[.2,-.3]],requires_grad=True)
        adv=torch.tensor([[1.,-1.]])
        loss=policy_loss('sapo',z,torch.zeros_like(z),adv,torch.ones_like(z),total_sequences=1,max_completion_length=4)
        loss.backward()
        ratio=z.detach().exp();tau=torch.tensor([[1.,1.05]])
        gate=torch.sigmoid(tau*(ratio-1))
        expected=-4*gate*(1-gate)*ratio*adv/2
        torch.testing.assert_close(z.grad,expected)

    def test_microbatch_gradient_equals_whole_batch(self):
        for method in ['dr_grpo','sapo']:
            z=torch.tensor([[.1,.2],[-.1,.3]],requires_grad=True)
            mask=torch.tensor([[1.,0.],[1.,1.]])
            adv=torch.tensor([[1.,1.],[-1.,-1.]])
            kw=dict(total_sequences=2,max_completion_length=4)
            full=policy_loss(method,z,torch.zeros_like(z),adv,mask,**kw)
            g=torch.autograd.grad(full,z)[0]
            separate=sum(policy_loss(method,z[i:i+1],torch.zeros_like(z[i:i+1]),adv[i:i+1],mask[i:i+1],**kw) for i in range(2))
            torch.testing.assert_close(torch.autograd.grad(separate,z)[0],g)

if __name__=='__main__':unittest.main()
