import unittest
from envs.gsm8k import verify_answer


class EvaluationVerifierTest(unittest.TestCase):
    def test_exact_answer_and_thinking_boundary(self):
        self.assertFalse(verify_answer("#### 312","12"))
        self.assertFalse(verify_answer("12","12"))
        self.assertTrue(verify_answer("<think>#### 99</think>\n#### 12","12"))
        self.assertFalse(verify_answer("<think>#### 12</think>\n#### 99","12"))
        self.assertTrue(verify_answer("#### 1,200","1200"))
