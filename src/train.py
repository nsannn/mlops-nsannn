# src/train.py
"""Fine-tune the pinned HF model and log the run to MLflow.

Run: python src/train.py [--learning_rate 3e-5 --epochs 3 ...]  (CLI overrides params.yaml)
Tracking server comes from MLFLOW_TRACKING_URI; it is never hard-coded.
build/fit/score are shared with scripts/hpo.py, so the search trains exactly like the pipeline.
"""
import argparse
import os
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

# Explicit BERT classes: always load these checkpoints (Auto* fails on old configs without model_type).
from transformers import BertForSequenceClassification, BertTokenizerFast

from model_utils import encode, predict, setup_determinism

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "tweet-emotion"
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]  # dair-ai/emotion label ids 0..5
LOG_EVERY = 10  # optimizer steps between train_loss points


def git_commit() -> str:
    if os.environ.get("GIT_COMMIT"):  # Docker/K8s have no .git
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=8", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):  # no git binary or no .git
        return "unknown"


def dvc_md5(dvc_file: Path) -> str:
    """Raw-data md5 from the .dvc pointer: the link between MLflow and DVC."""
    try:
        return yaml.safe_load(dvc_file.read_text())["outs"][0]["md5"]
    except (OSError, yaml.YAMLError, KeyError, IndexError, TypeError):  # missing or malformed pointer
        return "untracked"


def load_params(argv: list[str] | None = None) -> dict:
    p = yaml.safe_load((ROOT / "params.yaml").read_text())["train"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float)
    for key in ("epochs", "batch_size", "max_length", "random_seed"):
        parser.add_argument(f"--{key}", type=int)
    for key, value in vars(parser.parse_args(argv)).items():
        if value is not None:
            p[key] = value
    # MLflow stores params as text ("5e-05"); reproduce.py writes them back, so cast here.
    p["learning_rate"] = float(p["learning_rate"])
    for key in ("epochs", "batch_size", "max_length", "random_seed", "num_labels"):
        p[key] = int(p[key])
    return p


def run_tags(p: dict) -> dict:
    """Lineage on every run: code version, data version, seed and where it ran."""
    tags = {"git_commit": git_commit(),
            "dvc_raw_md5": dvc_md5(ROOT / "data/raw/dataset.parquet.dvc"),
            "random_seed": str(p["random_seed"]),
            "runner": os.environ.get("RUNNER", "laptop")}
    if os.environ.get("REPRODUCES"):
        tags["reproduces"] = os.environ["REPRODUCES"]
    return tags


def build(p: dict):
    tok = BertTokenizerFast.from_pretrained(p["hf_model"], revision=p["hf_revision"])
    model = BertForSequenceClassification.from_pretrained(
        p["hf_model"], revision=p["hf_revision"], num_labels=p["num_labels"],
        id2label=dict(enumerate(LABELS)), label2id={name: i for i, name in enumerate(LABELS)})
    return tok, model


def fit(model, tok, df: pd.DataFrame, p: dict) -> None:
    """Training loop; logs train_loss per step and epoch_loss per epoch to the active run."""
    loader = DataLoader(encode(tok, df, p["max_length"]), batch_size=p["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(p["random_seed"]))
    opt = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"])
    model.train()
    step = 0
    for epoch in range(1, p["epochs"] + 1):
        total = 0.0
        for ids, mask, y in loader:
            loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            total += loss.item()
            if step % LOG_EVERY == 0:
                mlflow.log_metric("train_loss", loss.item(), step=step)
        mlflow.log_metric("epoch_loss", total / len(loader), step=epoch)


def score(model, tok, df: pd.DataFrame, p: dict) -> dict:
    preds = predict(model, tok, df, p["max_length"])
    return {"accuracy": accuracy_score(df["label"], preds), "f1": f1_score(df["label"], preds, average="macro")}


def main() -> None:
    p = load_params()
    setup_determinism(p["random_seed"])
    train = pd.read_parquet(ROOT / "data/processed/train.parquet")
    test = pd.read_parquet(ROOT / "data/processed/test.parquet")
    tok, model = build(p)

    mlflow.set_experiment(EXPERIMENT)
    mlflow.set_system_metrics_sampling_interval(2)  # CPU, RAM, disk, network every 2 s (needs psutil)
    run_name = f"bert-l4-lr{p['learning_rate']:g}-ep{p['epochs']}-len{p['max_length']}"
    with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
        mlflow.log_params(p)  # seed, model name and revision are params too
        mlflow.set_tags(run_tags(p))
        mlflow.log_input(mlflow.data.from_pandas(train, source="data/processed/train.parquet",
                                                 targets="label", name="train"), context="training")
        mlflow.log_input(mlflow.data.from_pandas(test, source="data/processed/test.parquet",
                                                 targets="label", name="test"), context="evaluation")
        fit(model, tok, train, p)
        metrics = score(model, tok, test, p)
        # Explicit pip_requirements: auto-detection looks for torchvision and crashes on CPU-only torch.
        # input_example gives the model a signature (string in, label + score out).
        info = mlflow.transformers.log_model({"model": model, "tokenizer": tok}, name="model",
                                             task="text-classification",
                                             input_example=["i feel wonderful today"],
                                             pip_requirements=str(ROOT / "requirements-train.txt"))
        for key, value in metrics.items():
            mlflow.log_metric(key, value, model_id=info.model_id)

        model.save_pretrained(ROOT / "models/model")  # DVC output read by evaluate.py
        tok.save_pretrained(ROOT / "models/model")
        (ROOT / "models/run_id.txt").write_text(run.info.run_id + "\n")
        print(f"run_id={run.info.run_id} model_id={info.model_id} "
              f"accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
