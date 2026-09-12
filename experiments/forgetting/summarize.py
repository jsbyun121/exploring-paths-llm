#!/usr/bin/env python3
"""Summarize base-relative retention and matched JSD-vs-Dr.GRPO effects."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def metric_value(result: dict, task: str, metric: str) -> float:
    task_result = result.get("results", {}).get(task)
    if task_result is None:
        task_result = result.get("groups", {}).get(task)
    if task_result is None:
        raise KeyError(f"Task/group {task!r} not found")
    candidates = [metric, f"{metric},none"]
    candidates += [key for key in task_result if key.split(",", 1)[0] == metric]
    for key in candidates:
        value = task_result.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    raise KeyError(f"Metric {metric!r} not found for {task}; keys={sorted(task_result)}")


def find_result(directory: Path) -> Path:
    paths = sorted(directory.rglob("results*.json"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected exactly one results JSON in {directory}, found {len(paths)}")
    return paths[0]


def sample_scores(directory: Path, metric: str) -> dict[str, float]:
    paths = sorted(directory.rglob("samples*.jsonl"))
    if not paths:
        return {}
    scores: dict[str, float] = {}
    for path in paths:
        match = re.match(r"samples_(.+?)_\d{4}-\d{2}-\d{2}", path.stem)
        source_task = match.group(1) if match else path.stem
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                value = row.get(metric, row.get(f"{metric},none"))
                if not isinstance(value, (int, float)):
                    continue
                task_name = str(row.get("task_name", source_task))
                doc_id = f"{task_name}:{row['doc_id']}"
                if doc_id in scores:
                    raise ValueError(f"Duplicate sample key {doc_id} in {path}")
                scores[doc_id] = float(value)
    return scores


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def paired_macro_bootstrap(
    task_differences: dict[str, list[float]],
    task_weights: dict[str, float],
    replicates: int,
    seed: int = 0,
) -> tuple[float, float]:
    """Hierarchical task/item bootstrap for a macro difference."""
    rng = random.Random(seed)
    tasks = list(task_differences)
    if not tasks or any(not task_differences[task] for task in tasks):
        raise ValueError("Every bootstrapped task needs paired item differences")
    probabilities = [task_weights[task] for task in tasks]
    total = sum(probabilities)
    probabilities = [value / total for value in probabilities]
    draws = []
    for _ in range(replicates):
        selected = rng.choices(tasks, weights=probabilities, k=len(tasks))
        task_means = []
        for task in selected:
            values = task_differences[task]
            sampled = [values[rng.randrange(len(values))] for _ in values]
            task_means.append(sum(sampled) / len(sampled))
        draws.append(sum(task_means) / len(task_means))
    return percentile(draws, 0.025), percentile(draws, 0.975)


def trapezoid_auc(points: list[tuple[float, float]]) -> float:
    points = sorted(points)
    if len(points) < 2 or points[-1][0] == points[0][0]:
        raise ValueError("AUC requires at least two distinct exposures")
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2
    return area / (points[-1][0] - points[0][0])


def trajectory_auc_bootstrap(
    pairs: list[dict],
    samples: dict[str, dict[str, dict[str, float]]],
    task_weights: dict[str, float],
    replicates: int,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap a matched trajectory while reusing item draws across steps."""
    if len(pairs) < 2:
        raise ValueError("Trajectory bootstrap requires at least two checkpoints")
    tasks = list(task_weights)
    common: dict[str, list[str]] = {}
    for task in tasks:
        item_sets = []
        for pair in pairs:
            item_sets.append(set(samples[pair["jsd_model_id"]].get(task, {})))
            item_sets.append(set(samples[pair["dr_grpo_model_id"]].get(task, {})))
        shared = set.intersection(*item_sets) if item_sets else set()
        if not shared:
            raise ValueError(f"No common sample IDs across trajectory for {task}")
        common[task] = sorted(shared)

    rng = random.Random(seed)
    probabilities = [task_weights[task] for task in tasks]
    draws = []
    for _ in range(replicates):
        selected_tasks = rng.choices(tasks, weights=probabilities, k=len(tasks))
        sampled_ids = {
            task: [
                common[task][rng.randrange(len(common[task]))]
                for _ in common[task]
            ]
            for task in set(selected_tasks)
        }
        points = []
        for pair in pairs:
            task_means = []
            for task in selected_tasks:
                jsd = samples[pair["jsd_model_id"]][task]
                dr = samples[pair["dr_grpo_model_id"]][task]
                values = [jsd[item] - dr[item] for item in sampled_ids[task]]
                task_means.append(sum(values) / len(values))
            exposure = 0.5 * (pair["jsd_exposure"] + pair["dr_grpo_exposure"])
            points.append((exposure, sum(task_means) / len(task_means)))
        draws.append(trapezoid_auc(points))
    return percentile(draws, 0.025), percentile(draws, 0.975)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=HERE / "models.json")
    parser.add_argument("--suite", type=Path, default=HERE / "preregistered_suite.json")
    parser.add_argument("--input", type=Path, default=REPO / "outputs/forgetting/lm_eval")
    parser.add_argument("--output", type=Path, default=REPO / "outputs/forgetting/summary")
    parser.add_argument("--bootstrap-replicates", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = load_json(args.suite)
    registry = load_json(args.models)
    entries = {item["id"]: item for item in registry["models"]}
    tasks = suite["primary_tasks"]
    weights = {item["name"]: float(item["weight"]) for item in tasks}
    total_weight = sum(weights.values())
    weights = {name: value / total_weight for name, value in weights.items()}
    replicates = args.bootstrap_replicates or suite["primary_endpoint"]["bootstrap_replicates"]

    rows = []
    samples: dict[str, dict[str, dict[str, float]]] = {}
    missing = []
    for model_id, entry in entries.items():
        samples[model_id] = {}
        for task in tasks:
            directory = args.input / model_id / task["name"]
            try:
                result = load_json(find_result(directory))
                score = metric_value(result, task["name"], task["metric"])
            except (FileNotFoundError, KeyError) as exc:
                missing.append({"model_id": model_id, "task": task["name"], "error": str(exc)})
                continue
            samples[model_id][task["name"]] = sample_scores(directory, task["metric"])
            rows.append({
                "model_id": model_id,
                "family": entry["family"],
                "training_step": entry["training_step"],
                "task": task["name"],
                "metric": task["metric"],
                "score": score,
            })

    score_index = {(row["model_id"], row["task"]): row["score"] for row in rows}
    base_id = next(item["id"] for item in entries.values() if item["family"] == "base")
    for row in rows:
        base = score_index.get((base_id, row["task"]))
        row["base_score"] = base
        row["delta_from_base"] = row["score"] - base if base is not None else None
        row["relative_retention"] = row["score"] / base if base else None

    macro = []
    for model_id, entry in entries.items():
        available = [row for row in rows if row["model_id"] == model_id]
        if len(available) != len(tasks):
            continue
        score = sum(weights[row["task"]] * row["score"] for row in available)
        base_score = sum(weights[row["task"]] * row["base_score"] for row in available)
        macro.append({
            "model_id": model_id,
            "family": entry["family"],
            "training_step": entry["training_step"],
            "score": score,
            "base_score": base_score,
            "delta_from_base": score - base_score,
            "relative_retention": score / base_score if base_score else None,
        })

    jsd_by_step = {e["training_step"]: e for e in entries.values() if e["family"] == "jsd"}
    dr_by_step = {e["training_step"]: e for e in entries.values() if e["family"] == "dr_grpo"}
    matched = []
    for step in sorted(set(jsd_by_step) & set(dr_by_step)):
        jsd_id, dr_id = jsd_by_step[step]["id"], dr_by_step[step]["id"]
        task_differences = {}
        task_points = {}
        for task in tasks:
            name = task["name"]
            jsd_score = score_index.get((jsd_id, name))
            dr_score = score_index.get((dr_id, name))
            if jsd_score is None or dr_score is None:
                continue
            task_points[name] = jsd_score - dr_score
            jsd_items, dr_items = samples[jsd_id].get(name, {}), samples[dr_id].get(name, {})
            shared = sorted(set(jsd_items) & set(dr_items))
            if shared:
                task_differences[name] = [jsd_items[item] - dr_items[item] for item in shared]
        if len(task_points) != len(tasks):
            continue
        effect = sum(weights[name] * value for name, value in task_points.items())
        ci = None
        if len(task_differences) == len(tasks):
            ci = paired_macro_bootstrap(task_differences, weights, replicates, seed=step)
        exposure_key = suite["exposure_matching"]["primary"]
        exposure = jsd_by_step[step].get(exposure_key)
        dr_exposure = dr_by_step[step].get(exposure_key)
        if exposure is None or dr_exposure is None:
            exposure_key = suite["exposure_matching"]["fallback"]
            exposure = jsd_by_step[step].get(exposure_key)
            dr_exposure = dr_by_step[step].get(exposure_key)
        relative_gap = abs(exposure - dr_exposure) / max(exposure, dr_exposure) if exposure and dr_exposure else 0.0
        matched.append({
            "training_step": step,
            "jsd_model_id": jsd_id,
            "dr_grpo_model_id": dr_id,
            "exposure_key": exposure_key,
            "jsd_exposure": exposure,
            "dr_grpo_exposure": dr_exposure,
            "relative_exposure_gap": relative_gap,
            "jsd_minus_dr_grpo": effect,
            "ci95": list(ci) if ci else None,
            "task_differences": task_points,
        })

    valid_auc = [
        (0.5 * (row["jsd_exposure"] + row["dr_grpo_exposure"]), row["jsd_minus_dr_grpo"])
        for row in matched
        if row["relative_exposure_gap"] <= suite["exposure_matching"]["maximum_relative_gap"]
    ]
    auc_ci = None
    valid_pairs = [
        row for row in matched
        if row["relative_exposure_gap"] <= suite["exposure_matching"]["maximum_relative_gap"]
    ]
    if len(valid_pairs) >= 2:
        try:
            auc_ci = trajectory_auc_bootstrap(valid_pairs, samples, weights, replicates)
        except ValueError:
            auc_ci = None
    primary = {
        "status": "complete" if len(valid_auc) >= 2 else "incomplete",
        "reason": None if len(valid_auc) >= 2 else "Need at least two exposure-matched JSD/Dr.GRPO checkpoints.",
        "retention_auc_jsd_minus_dr_grpo": trapezoid_auc(valid_auc) if len(valid_auc) >= 2 else None,
        "confidence_interval": list(auc_ci) if auc_ci else None,
        "note": "The trajectory CI uses common item draws across checkpoints when all lm-eval sample logs are available.",
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps({
        "schema": "catastrophic-forgetting-summary/v1",
        "primary_endpoint": primary,
        "macro_retention": macro,
        "matched_comparisons": matched,
        "missing": missing,
    }, indent=2) + "\n")
    with (args.output / "task_scores.csv").open("w", newline="") as stream:
        fields = ["model_id", "family", "training_step", "task", "metric", "score", "base_score", "delta_from_base", "relative_retention"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"primary_endpoint": primary, "completed_models": len(macro), "missing_runs": len(missing)}, indent=2))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
