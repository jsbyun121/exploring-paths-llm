import unittest
import json
import tempfile
from pathlib import Path

from experiments.forgetting.run_lm_eval import model_fingerprint
from experiments.forgetting.summarize import (
    metric_value,
    paired_macro_bootstrap,
    percentile,
    trapezoid_auc,
    trajectory_auc_bootstrap,
)


class SummaryStatisticsTest(unittest.TestCase):
    def test_metric_value_supports_task_groups(self):
        result = {"results": {}, "groups": {"mmlu": {"acc,none": 0.6}}}
        self.assertEqual(metric_value(result, "mmlu", "acc"), 0.6)

    def test_model_fingerprint_rejects_mislabeled_best_step(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text("{}")
            (model / "model.safetensors").write_bytes(b"weight")
            (model / "training_state.json").write_text(json.dumps({"step": 180}))
            with self.assertRaisesRegex(ValueError, "does not match"):
                model_fingerprint(str(model), expected_step=220)

    def test_percentile_interpolates(self):
        self.assertEqual(percentile([0.0, 10.0], 0.25), 2.5)

    def test_bootstrap_preserves_positive_paired_effect(self):
        low, high = paired_macro_bootstrap(
            {"a": [1.0] * 20, "b": [2.0] * 20},
            {"a": 0.5, "b": 0.5},
            replicates=200,
        )
        self.assertGreater(low, 0)
        self.assertGreater(high, low)

    def test_normalized_auc(self):
        self.assertAlmostEqual(trapezoid_auc([(10, 0.1), (20, 0.3)]), 0.2)

    def test_trajectory_bootstrap(self):
        samples = {
            "j1": {"task": {"0": 1.0, "1": 1.0}},
            "d1": {"task": {"0": 0.0, "1": 0.0}},
            "j2": {"task": {"0": 0.8, "1": 0.8}},
            "d2": {"task": {"0": 0.2, "1": 0.2}},
        }
        low, high = trajectory_auc_bootstrap(
            [
                {"jsd_model_id": "j1", "dr_grpo_model_id": "d1", "jsd_exposure": 10, "dr_grpo_exposure": 10},
                {"jsd_model_id": "j2", "dr_grpo_model_id": "d2", "jsd_exposure": 20, "dr_grpo_exposure": 20},
            ],
            samples,
            {"task": 1.0},
            replicates=50,
        )
        self.assertAlmostEqual(low, 0.8)
        self.assertAlmostEqual(high, 0.8)


if __name__ == "__main__":
    unittest.main()
