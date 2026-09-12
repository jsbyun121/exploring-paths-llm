"""Snapshot local experiment evidence and render tables without running models.

Run from the repository root: python3 -m experiments.update_results --date YYYY-MM-DD
The JSON preserves source contents and SHA-256 digests; --snapshot renders the
report from that committed evidence without access to local outputs.
"""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FOCUSED = {
    "base": "Base",
    "positive_ce_step100": "Positive CE, step 100",
    "rank_jsd_step100": "Historical rank-JSD, step 100",
    "rank_kl_step100": "Rank-KL, step 100",
    "rank_jsd_beta1_zero_step100": "Rank-JSD beta1=0, step 100",
    "rank_kl_beta1_zero_step100": "Rank-KL beta1=0, step 100",
}
ARMS = {"jsd": "Rank-JSD beta1=0", "ce": "Positive CE", "dr_grpo": "Dr.GRPO", "sapo": "SAPO"}


def collect(as_of):
    snapshot = {"as_of": as_of, "sources": {}, "missing": []}

    def read(relative, optional=False):
        path = REPO / relative
        if optional and not path.exists():
            snapshot["missing"].append(relative)
            return None
        raw = path.read_bytes()
        value = json.loads(raw)
        snapshot["sources"][relative] = {
            "sha256": hashlib.sha256(raw).hexdigest(), "data": value,
        }
        return value

    reference = None
    for group, names in [("focused", FOCUSED), ("long", ARMS)]:
        for name in names:
            prefix = f"outputs/{group}/evaluation/{name}"
            summary = read(f"{prefix}/summary.json", optional=True)
            if summary is None:
                continue
            manifest = read(f"{prefix}/manifest.json")
            settings = {key: manifest[key] for key in (
                "dataset", "dataset_config", "split", "prompt_style", "prompt_hashes",
                "gold_hash", "sample_k", "sample_temperature", "top_p", "max_new_tokens", "seed",
            )}
            if reference is not None and settings != reference:
                raise ValueError(f"Unmatched evaluation settings: {prefix}")
            reference = settings
            raw = (REPO / prefix / "predictions.jsonl").read_bytes()
            rows = [json.loads(line) for line in raw.splitlines()]
            if len(rows) != summary["examples"] or len(rows) != len(manifest["prompt_hashes"]):
                raise ValueError(f"Incomplete predictions: {prefix}")
            for index, row in enumerate(rows):
                if row["index"] != index or row["prompt_sha256"] != manifest["prompt_hashes"][index]:
                    raise ValueError(f"Mismatched prediction: {prefix}, {index}")
                if len(row["sample_correct"]) != summary["sample_k"]:
                    raise ValueError(f"Incomplete samples: {prefix}, {index}")
            for key, actual in {
                "greedy_accuracy": sum(r["greedy_correct"] for r in rows) / len(rows),
                "pass_at_k": sum(any(r["sample_correct"]) for r in rows) / len(rows),
                "sample_accuracy": sum(sum(r["sample_correct"]) for r in rows) / (len(rows) * summary["sample_k"]),
                "mean_greedy_tokens": sum(r["greedy_tokens"] for r in rows) / len(rows),
                "mean_sample_tokens": sum(sum(r["sample_tokens"]) for r in rows) / (len(rows) * summary["sample_k"]),
            }.items():
                if abs(summary[key] - actual) > 1e-9:
                    raise ValueError(f"Summary disagrees with predictions: {prefix}/{key}")
            snapshot["sources"][f"{prefix}/predictions.jsonl"] = {
                "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(rows),
                "note": "Generations remain in local outputs; aggregate metrics verified against these rows.",
            }
    for name in ARMS:
        state = read(f"outputs/long/{name}/long_state.json")
        completed = read(f"outputs/long/{name}/long_completed.json", optional=True)
        best = read(f"outputs/long/{name}/best/completed.json", optional=True)
        if best is not None and best["step"] != (completed or state)["best_step"]:
            raise ValueError(f"Best checkpoint label disagrees with saved model: {name}")
    for pattern in ("focused/*vs*.json", "long/*vs*.json"):
        for path in sorted((REPO / "outputs").glob(pattern)):
            read(str(path.relative_to(REPO)))
    read("outputs/forgetting/capture_status.json", optional=True)
    snapshot["forgetting_result_files"] = [
        str(p.relative_to(REPO)) for p in sorted((REPO / "outputs/forgetting/lm_eval").rglob("results*.json"))
    ]
    return snapshot


def render(snapshot):
    sources = snapshot["sources"]

    def data(path):
        return sources.get(path, {}).get("data")

    lines = [
        "# Experiment results", "", f"Local artifact snapshot: {snapshot['as_of']} (UTC).", "",
        "Evidence is stored in [experiment_metrics.json](experiment_metrics.json), including source paths, "
        "SHA-256 hashes, evaluation settings, confidence intervals, paired comparisons, and validation histories. "
        "Evaluation aggregates were checked against every saved prediction. No new training or model evaluation is performed by this report.", "",
        "## Evaluation protocol and limits", "",
        "All completed evaluations below use Qwen3-4B-Thinking-2507 and the same first 200 examples of "
        "`train[:500]@openai/gsm8k:main`, held out from `train[500:]`. This subset overlaps the 500-question "
        "validation set used for checkpoint selection; it is not the official GSM8K test set. "
        "The training prompt/verifier is used, with greedy decoding and eight samples per problem at "
        "temperature 0.7, top-p 0.95, a 4096-token response cap, and seed 0. "
        "Pass@8 is the fraction of problems solved by at least one of those eight samples. "
        "Tokens are mean completion tokens per response, not equalized inference budgets.", "",
        "These are exploratory single-seed recipe comparisons. Historical JSD used an older runtime; "
        "earlier resumes could omit Adam moments. The repaired recovery path restores optimizer and RNG state. "
        "Padding/batching changes can affect BF16 numerics. Beta1=0 removes first-moment history but retains "
        "second-moment adaptation. These confounds prevent a clean optimizer or loss-function attribution.", "",
        "## Focused 100-step diagnostics", "",
    ]

    def evaluation_table(group, names):
        lines.extend([
            "| Model/checkpoint | Greedy accuracy | Pass@8 | Sample accuracy | Greedy tokens | Sample tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for name, label in names.items():
            value = data(f"outputs/{group}/evaluation/{name}/summary.json")
            if value is None:
                lines.append(f"| {label} | Pending | — | — | — | — |")
                continue
            lines.append(f"| {label} | {value['greedy_accuracy']:.1%} | {value['pass_at_k']:.1%} | "
                         f"{value['sample_accuracy']:.2%} | {value['mean_greedy_tokens']:.1f} | {value['mean_sample_tokens']:.1f} |")
        lines.append("")

    evaluation_table("focused", FOCUSED)
    lines.extend(["## Long-run training status", "",
                  "Validation uses all 500 held-out questions. Completed arms stopped when current validation "
                  "accuracy fell below both previous validation measurements. Evaluations use the saved best "
                  "checkpoint, not the stopping checkpoint.", "",
                  "| Recipe | Last saved step | Best step | Best validation | Last validation | Status |",
                  "| --- | ---: | ---: | ---: | ---: | --- |"])
    labels = {}
    for name, label in ARMS.items():
        state = data(f"outputs/long/{name}/long_completed.json") or data(f"outputs/long/{name}/long_state.json")
        complete = data(f"outputs/long/{name}/long_completed.json") is not None
        labels[name] = f"{label}, best step {state['best_step']}"
        last = state["history"][-1]
        lines.append(f"| {label} | {state['step']} | {state['best_step']} | {state['best_accuracy']:.1%} | "
                     f"{last['accuracy']:.1%} (step {last['step']}) | {'Completed' if complete else 'Incomplete'} |")
    lines.extend(["", "CE and JSD continue from step 100. Their recorded `generated_tokens`, `rollout_responses`, "
                  "and `used_seconds` cover the long continuation only; they exclude the initial focused run. "
                  "Dr.GRPO and SAPO start at step 0. Do not treat these counters as matched total training exposure. "
                  "JSD uses beta1=0, one response per prompt and one update epoch; Dr.GRPO/SAPO use beta1=0.9, "
                  "eight responses per prompt and four update epochs. See the [run configurations](a100/long_comparison/).", "",
                  "## Long-run best-checkpoint diagnostics", ""])
    evaluation_table("long", labels)
    comparison = data("outputs/long/jsd_vs_ce_best.json")
    if comparison:
        metric = comparison["greedy_accuracy"]
        lo, hi = metric["paired_bootstrap95"]
        lines.extend([f"CE minus JSD greedy accuracy is {100 * metric['difference']:.1f} percentage points "
                      f"(paired problem-bootstrap 95% interval: {100 * lo:.1f} to {100 * hi:.1f}; "
                      f"{comparison['bootstrap_draws']:,} draws). This interval includes zero. "
                      "The JSON includes the other existing paired comparisons and completion-token-budget results.", ""])
    lines.extend(["## Forgetting evaluation", ""])
    if not snapshot["forgetting_result_files"]:
        lines.append("No local lm-eval result files are available. No retention score or catastrophic-forgetting conclusion is reported.")
    else:
        lines.append(f"Found {len(snapshot['forgetting_result_files'])} local result files; use the preregistered suite summarizer for retention analysis.")
    lines.extend(["", "Dr.GRPO checkpoints 100 and 120 were captured; 180 and 220 are unavailable because training "
                  "stopped at 120. The JSD step-220 recovery snapshot contains step-180 best weights: export the "
                  "step-220 DCP to evaluate step 220. See the [forgetting protocol](forgetting/README.md).", "",
                  "## Reproduce the report", "", "With the original local outputs available:", "", "```bash",
                  f"python3 -m experiments.update_results --date {snapshot['as_of']}", "```", "",
                  "Render from committed evidence only:", "", "```bash",
                  "python3 -m experiments.update_results --snapshot experiments/experiment_metrics.json", "```", "",
                  "Raw generations, checkpoints, dependency installations, and logs remain outside Git. "
                  "The long-run backup/archive scripts require the deployment-specific helpers in "
                  "`/workspace/backup-integration`; they are not a portable backup service. "
                  "Launchers also contain deployment paths that must be adapted on another host.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.snapshot:
        snapshot = json.loads(args.snapshot.read_text())
    else:
        snapshot = collect(args.date)
        (REPO / "experiments/experiment_metrics.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    (REPO / "experiments/EXPERIMENT_RESULTS.md").write_text(render(snapshot))


if __name__ == "__main__":
    main()
