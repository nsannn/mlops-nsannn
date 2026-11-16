# tests/test_gate.py
import json

from check_model_gate import main


def run_gate(tmp_path, metrics: dict, threshold: str) -> int:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics))
    return main(["--metrics", str(path), "--metric", "f1", "--threshold", threshold])


def test_pass_above_threshold(tmp_path):
    assert run_gate(tmp_path, {"f1": 0.83}, "0.80") == 0


def test_fail_below_threshold(tmp_path):
    assert run_gate(tmp_path, {"f1": 0.79}, "0.80") == 1


def test_equal_passes(tmp_path):
    assert run_gate(tmp_path, {"f1": 0.80}, "0.80") == 0


def test_missing_metric_fails(tmp_path):
    assert run_gate(tmp_path, {"accuracy": 0.9}, "0.80") == 1
