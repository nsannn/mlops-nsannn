# scripts/registry.py
"""Model Registry steps of the pipeline, with aliases (not the deprecated stages).

  register  version from models/run_id.txt -> alias @candidate   (deploy.sh calls it only after the gate passes)
  promote   @champion -> the @candidate version; the old champion keeps alias @previous
  rollback  @champion -> @previous, or --version N; the replaced version becomes @previous

Run: python scripts/registry.py register|promote|rollback [--version N]   (needs MLFLOW_TRACKING_URI)
"""
import argparse
import os
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

NAME = "tweet-emotion"
ROOT = Path(__file__).resolve().parents[1]


def version_of(client: MlflowClient, alias: str) -> str | None:
    try:
        return client.get_model_version_by_alias(NAME, alias).version
    except MlflowException:
        return None


def register(client: MlflowClient, args: argparse.Namespace) -> None:
    run_id = (ROOT / "models/run_id.txt").read_text().strip()
    # Idempotent: a pipeline re-run on the same training run reuses its version instead of adding one.
    existing = client.search_model_versions(f"name = '{NAME}' and run_id = '{run_id}'")
    if existing:
        version = max(existing, key=lambda v: int(v.version)).version
    else:
        version = mlflow.register_model(f"runs:/{run_id}/model", NAME).version
        commit = client.get_run(run_id).data.tags.get("git_commit", "unknown")
        client.set_model_version_tag(NAME, version, "git_commit", commit)
    client.set_registered_model_alias(NAME, "candidate", version)
    print(f"{NAME} v{version} (run {run_id}) -> @candidate")


def promote(client: MlflowClient, args: argparse.Namespace) -> None:
    new = version_of(client, "candidate")
    if new is None:
        sys.exit("no @candidate version: run register first")
    old = version_of(client, "champion")
    if old == new:
        print(f"{NAME} v{new} is already @champion")
        return
    if old is not None:
        client.set_registered_model_alias(NAME, "previous", old)
    client.set_registered_model_alias(NAME, "champion", new)
    print(f"@champion: v{old} -> v{new}")


def rollback(client: MlflowClient, args: argparse.Namespace) -> None:
    target = args.version or version_of(client, "previous")
    if target is None:
        sys.exit("no @previous version: pass --version N")
    old = version_of(client, "champion")
    client.set_registered_model_alias(NAME, "champion", str(target))
    if old is not None and old != str(target):
        client.set_registered_model_alias(NAME, "previous", old)
    print(f"rollback @champion: v{old} -> v{target}")


def main() -> None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        sys.exit("MLFLOW_TRACKING_URI is not set: refusing to use a local ./mlruns")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["register", "promote", "rollback"])
    parser.add_argument("--version", help="rollback target (default: @previous)")
    args = parser.parse_args()
    {"register": register, "promote": promote, "rollback": rollback}[args.command](MlflowClient(), args)


if __name__ == "__main__":
    main()
