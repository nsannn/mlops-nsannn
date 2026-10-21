# scripts/fetch_data.py
"""Download the pinned HF emotion dataset and keep a seeded, stratified 10k-tweet sample.

Run: python scripts/fetch_data.py
Output: data/raw/dataset.parquet, then tracked with `dvc add`.
"""
from pathlib import Path

import pandas as pd
from datasets import load_dataset

REPO = "dair-ai/emotion"
REVISION = "cab853a1dbdf4c42c2b3ef2173804746df8825fe"  # a commit hash, never a branch name
N_ROWS, SEED = 10_000, 42
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ds = load_dataset(REPO, "split", revision=REVISION)
    df = pd.concat([ds[s].to_pandas() for s in ds], ignore_index=True)[["text", "label"]]
    # stratified sample: keeps the natural class balance (joy ~34 %, surprise ~4 %)
    df = df.groupby("label", group_keys=False).sample(frac=N_ROWS / len(df), random_state=SEED)
    out = ROOT / "data/raw/dataset.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(df.shape, df["label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()