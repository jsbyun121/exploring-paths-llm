import unittest
import torch
from RL2.utils.functions import aggregate_values


class MetricAggregationTest(unittest.TestCase):
    def test_sequence_bool_clip_metric_with_float_loss_backward(self):
        loss = torch.tensor([[2., 4., 0.], [6., 0., 0.]], requires_grad=True)
        clipped = torch.tensor([[True, False, False], [True, False, False]])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        value, ratio = aggregate_values((loss, clipped), mask, 'sequence', 3, 2)
        self.assertAlmostEqual(value.item(), 4.5, places=5)
        self.assertAlmostEqual(ratio.item(), .75, places=5)
        value.backward()
        torch.testing.assert_close(loss.grad[:, 0], torch.tensor([.25, .5]))


if __name__ == '__main__':
    unittest.main()
