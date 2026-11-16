# scripts/check_model_gate.py
"""Quality gate: exit 0 if metrics.json[metric] >= threshold, else exit 1 (the pipeline stops on 1).

Defaults come from the gate: section of params.yaml; flags override them.
Run: python scripts/check_model_gate.py [--metric f1 --threshold 0.80 --metrics metrics.json]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def report(line: str) -> None:
    print(line)
    if os.environ.get("GITHUB_STEP_SUMMARY"):  # shows the verdict on the Actions run page
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(f"### Quality gate\n\n{line}\n")


def main(argv: list[str] | None = None) -> int:
    gate = yaml.safe_load((ROOT / "params.yaml").read_text()).get("gate", {})
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=str(ROOT / "metrics.json"))
    parser.add_argument("--metric", default=gate.get("metric", "f1"))
    parser.add_argument("--threshold", type=float, default=gate.get("threshold", 0.80))
    args = parser.parse_args(argv)
    value = json.loads(Path(args.metrics).read_text()).get(args.metric)
    if value is None:
        report(f"FAIL: metric '{args.metric}' not in {args.metrics}")
        return 1
    if value >= args.threshold:  # agreed rule: equal to the threshold passes
        report(f"PASS: {args.metric}={value:.4f} >= {args.threshold}")
        return 0
    report(f"FAIL: {args.metric}={value:.4f} < {args.threshold}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
