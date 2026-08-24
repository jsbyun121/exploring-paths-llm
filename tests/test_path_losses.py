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
