"""Paired, problem-level comparison of identically configured evaluations."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(left, right, draws=10000):
    manifests = [json.loads((p / "manifest.json").read_text()) for p in [left, right]]
    keys = ["dataset", "dataset_config", "split", "prompt_style", "prompt_hashes", "gold_hash",
            "sample_k", "sample_temperature", "top_p", "max_new_tokens", "seed"]
    if any(manifests[0][k] != manifests[1][k] for k in keys):
        raise ValueError("Evaluations differ in data, prompts, or sampling settings")
    rows = [[json.loads(line) for line in (p / "predictions.jsonl").read_text().splitlines()]
            for p in [left, right]]
    if len(rows[0]) != len(manifests[0]["prompt_hashes"]) or len(rows[1]) != len(rows[0]):
        raise ValueError("Both evaluations must be complete")
    for a,b in zip(*rows):
        if (a["index"],a["prompt_sha256"],a["gold"]) != (b["index"],b["prompt_sha256"],b["gold"]):
            raise ValueError("Prediction rows do not refer to the same examples")
    n = len(rows[0]); rng = np.random.default_rng(42)
    indices = rng.integers(0,n,(draws,n))
    result = {"left": str(left), "right": str(right), "difference_direction": "right minus left",
              "examples":n,"bootstrap_unit":"problem","bootstrap_draws":draws,
              "inference_budget_note":"Same samples and per-response cap; actual generated tokens are reported separately"}
    for name,metric in [("greedy_accuracy",lambda r:float(r["greedy_correct"])),
                        ("pass_at_k",lambda r:float(r["pass_at_k"])),
                        ("sample_accuracy",lambda r:float(np.mean(r["sample_correct"])))]:
        a,b = [np.array([metric(r) for r in rs]) for rs in rows]
        delta = b-a
        result[name] = {"left":float(a.mean()),"right":float(b.mean()),"difference":float(delta.mean()),
                        "paired_bootstrap95":np.quantile(delta[indices].mean(1),[.025,.975]).tolist()}
    result["right_only_solved"] = [a["index"] for a,b in zip(*rows) if b["pass_at_k"] and not a["pass_at_k"]]
    result["left_only_solved"] = [a["index"] for a,b in zip(*rows) if a["pass_at_k"] and not b["pass_at_k"]]
    result["mean_sample_tokens"] = {name:float(np.mean([np.mean(r["sample_tokens"]) for r in rs]))
                                    for name,rs in zip(["left","right"],rows)}
    # Fixed total completion-token budgets, using samples in their saved order.
    # A sample whose completion crosses the budget cannot be counted solved.
    result["pass_within_completion_token_budget"] = {}
    for budget in [4096,8192,16384]:
        values = []
        for rs in rows:
            outcomes = []
            for r in rs:
                used = 0; solved = False
                for correct,tokens in zip(r["sample_correct"],r["sample_tokens"]):
                    used += tokens
                    if used > budget:
                        break
                    solved |= correct
                outcomes.append(solved)
            values.append(float(np.mean(outcomes)))
        result["pass_within_completion_token_budget"][str(budget)] = dict(zip(["left","right"],values))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("left",type=Path);parser.add_argument("right",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();result=compare(args.left,args.right)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
