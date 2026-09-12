import json
import tempfile
import unittest
from pathlib import Path
from experiments.compare_evaluations import compare


class PairedComparisonTest(unittest.TestCase):
    def test_direction_and_refusal_of_mismatched_prompts(self):
        with tempfile.TemporaryDirectory() as root:
            left,right = [Path(root)/x for x in ["left","right"]]
            manifest = {"dataset":"gsm8k","dataset_config":"main","split":"train[:500]",
                        "prompt_style":"training","prompt_hashes":["a","b"],"gold_hash":"gold",
                        "sample_k":2,"sample_temperature":.7,"top_p":.95,"max_new_tokens":4096,"seed":0}
            for p,correct in [(left,[False,False]),(right,[True,False])]:
                p.mkdir();(p/"manifest.json").write_text(json.dumps(manifest))
                rows = [{"index":i,"prompt_sha256":manifest["prompt_hashes"][i],"gold":"12",
                         "greedy_correct":value,"sample_correct":[value,value],"pass_at_k":value,
                         "sample_tokens":[100,200]} for i,value in enumerate(correct)]
                (p/"predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            result=compare(left,right,draws=50)
            self.assertEqual(result["pass_at_k"]["difference"],.5)
            self.assertEqual(result["right_only_solved"],[0])
            manifest["prompt_hashes"]=["different","b"]
            (right/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                compare(left,right)
