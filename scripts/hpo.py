# scripts/hpo.py
"""Hyperparameter search: Optuna TPE, each trial a nested MLflow run under one parent run.

Run: python scripts/hpo.py [--trials 6]   (needs MLFLOW_TRACKING_URI)
Trials train 1 epoch and are scored on a validation split carved from train.parquet,
so the test split stays untouched for the final model trained by dvc repro.
"""
import argparse
import sys
from pathlib import Path

import mlflow
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from model_utils import setup_determinism
from train import EXPERIMENT, build, fit, load_params, run_tags, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=6)
    args = parser.parse_args()
    base = load_params([])  # params.yaml train section, no CLI overrides
    df = pd.read_parquet(ROOT / "data/processed/train.parquet")
    train, val = train_test_split(df, test_size=0.15, random_state=base["random_seed"], stratify=df["label"])

    def objective(trial: optuna.Trial) -> float:
        p = {**base, "epochs": 1,
             "learning_rate": trial.suggest_float("learning_rate", 2e-5, 5e-4, log=True),
             "max_length": trial.suggest_categorical("max_length", [16, 24, 32]),
             "batch_size": trial.suggest_categorical("batch_size", [16, 32])}
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True, log_system_metrics=True):
            mlflow.log_params(p)
            mlflow.set_tags({**run_tags(p), "optuna_trial": str(trial.number)})
            setup_determinism(p["random_seed"])
            tok, model = build(p)
            fit(model, tok, train, p)
            metrics = score(model, tok, val, p)
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        return metrics["f1"]

    mlflow.set_experiment(EXPERIMENT)
    mlflow.set_system_metrics_sampling_interval(2)
    with mlflow.start_run(run_name="hpo-optuna", log_system_metrics=True):
        mlflow.set_tags({**run_tags(base), "hpo": "optuna"})
        mlflow.log_params({"n_trials": args.trials, "sampler": "TPE", "objective": "val_f1", "trial_epochs": 1})
        mlflow.log_input(mlflow.data.from_pandas(train, source="data/processed/train.parquet",
                                                 targets="label", name="train-hpo"), context="training")
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=base["random_seed"]))
        study.optimize(objective, n_trials=args.trials)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_f1", study.best_value)
        print(f"best trial {study.best_trial.number}: {study.best_params} val_f1={study.best_value:.4f}")


if __name__ == "__main__":
    main()
