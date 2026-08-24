#!/usr/bin/env python3
"""Extract comparable endpoint/peak/stability metrics from trainer stdout."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STEP_RE = re.compile(r"Step\s+(\d+),\s*(.*)")
METRIC_RE = re.compile(r"([A-Za-z0-9_./@-]+):\s*([^,]+)")


def parse_log(path: Path) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        step_match = STEP_RE.search(line)
        if not step_match:
            continue
        step = int(step_match.group(1))
        record = steps.setdefault(step, {})
        for key, raw_value in METRIC_RE.findall(step_match.group(2)):
            try:
                record[key] = float(raw_value.strip())
            except ValueError:
                continue
    return steps


def series(steps, key):
    return [(step, values[key]) for step, values in sorted(steps.items()) if key in values]


def normalized_auc(values):
    if not values:
        return None
    if len(values) == 1:
        return values[0][1]
    area = sum(
        0.5 * (left[1] + right[1]) * (right[0] - left[0])
        for left, right in zip(values[:-1], values[1:])
    )
    return area / (values[-1][0] - values[0][0])


def summarize(path: Path):
    steps = parse_log(path)
    score = series(steps, "scores/test")
    entropy = series(steps, "actor/entropy_all") or series(steps, "actor/entropy")
    response_length = series(steps, "response_length/test")
    summary = {"log": str(path), "parsed_steps": len(steps)}
    for name, values in (
        ("validation_score", score),
        ("entropy", entropy),
        ("response_length", response_length),
    ):
        if not values:
            continue
        peak_step, peak_value = max(values, key=lambda pair: pair[1])
        summary.update(
            {
                f"{name}_final": values[-1][1],
                f"{name}_peak": peak_value,
                f"{name}_peak_step": peak_step,
                f"{name}_auc": normalized_auc(values),
            }
        )
    if score:
        summary["validation_retained_from_peak"] = score[-1][1] / max(
            max(value for _, value in score), 1e-12
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summaries = [summarize(path) for path in args.logs]
    output = json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
