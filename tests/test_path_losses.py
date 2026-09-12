import unittest

import torch

from RL2.utils.path_losses import (
    bernoulli_max_kl,
    fixed_half_kl,
    rank_shift_jsd,
)


class RankShiftJSDTest(unittest.TestCase):

    def test_matches_full_distribution_reference(self):
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]], requires_grad=True)
        action = torch.tensor([2])
        logsumexp = torch.logsumexp(logits, dim=-1)
        actual, ranks, in_cap = rank_shift_jsd(
            logits, logsumexp, action, rank_cap=4
        )

        q = torch.softmax(logits, dim=-1)
        order = torch.argsort(q, dim=-1, descending=True)[0]
        p = q.detach().clone()
        active = order[:3]
        p[0, active] = q.detach()[0, torch.roll(active, shifts=-1)]
        mixture = 0.5 * (p + q)
        expected = 0.5 * (
            (p * (p.log() - mixture.log())).sum(-1)
            + (q * (q.log() - mixture.log())).sum(-1)
        )

        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))
        self.assertEqual(ranks.item(), 3)
        self.assertTrue(in_cap.item())

    def test_gradient_promotes_chosen_token(self):
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]], requires_grad=True)
        loss, _, _ = rank_shift_jsd(
            logits,
            torch.logsumexp(logits, dim=-1),
            torch.tensor([2]),
            rank_cap=4,
        )
        loss.sum().backward()
        self.assertLess(logits.grad[0, 2].item(), 0.0)

    def test_rank_one_and_out_of_cap_are_zero(self):
        logits = torch.tensor(
            [[3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0]],
            requires_grad=True,
        )
        actions = torch.tensor([0, 3])
        losses, ranks, in_cap = rank_shift_jsd(
            logits,
            torch.logsumexp(logits, dim=-1),
            actions,
            rank_cap=2,
        )
        self.assertTrue(torch.allclose(losses, torch.zeros_like(losses)))
        self.assertEqual(ranks.tolist(), [1, 3])
        self.assertEqual(in_cap.tolist(), [True, False])


class RankKLTest(unittest.TestCase):
    def test_full_vocabulary_value_and_gradient_with_truncated_topk(self):
        # Action rank 3, cap 4, vocabulary 6: unchanged tail gradients matter.
        z = torch.tensor([[1., 4., -2., 3., 2., -1.]], requires_grad=True)
        loss, _, _ = rank_shift_jsd(
            z, torch.logsumexp(z, -1), torch.tensor([4]),
            rank_cap=4, divergence="kl",
        )
        p = z.softmax(-1)
        target = p.detach().clone()
        order = z.detach().argsort(descending=True)[0, :3]
        target[0, order] = p.detach()[0, order.roll(-1)]
        reference = (target * (target.log() - z.log_softmax(-1))).sum(-1)
        g = torch.autograd.grad(loss.sum(), z, retain_graph=True)[0]
        expected = torch.autograd.grad(reference.sum(), z)[0]
        torch.testing.assert_close(loss, reference, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(g, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(g, p.detach() - target, atol=1e-6, rtol=1e-5)
        self.assertLess(g[0, 4].item(), 0)

    def test_rank_one_and_out_of_cap_zero_gradient(self):
        z = torch.tensor([[3., 2., 1., 0.], [3., 2., 1., 0.]], requires_grad=True)
        loss, _, _ = rank_shift_jsd(
            z, torch.logsumexp(z, -1), torch.tensor([0, 3]),
            rank_cap=2, divergence="kl",
        )
        loss.sum().backward()
        torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)
        torch.testing.assert_close(z.grad, torch.zeros_like(z), atol=1e-6, rtol=0)


class LegacyObjectiveAuditTest(unittest.TestCase):

    def test_fixed_half_legacy_ignores_low_probability_action(self):
        logp = torch.tensor([[-2.0]])
        legacy = fixed_half_kl(logp, legacy_high_probability_only=True)
        corrected = fixed_half_kl(logp, legacy_high_probability_only=False)
        self.assertEqual(legacy.item(), 0.0)
        self.assertGreater(corrected.item(), 0.0)

    def test_detached_bernoulli_target_has_zero_loss_when_equal(self):
        logp = torch.tensor([[-0.4]], requires_grad=True)
        loss = bernoulli_max_kl(logp, logp, detach_target=True)
        self.assertTrue(torch.allclose(loss, torch.zeros_like(loss), atol=1e-7))


if __name__ == "__main__":
    unittest.main()
