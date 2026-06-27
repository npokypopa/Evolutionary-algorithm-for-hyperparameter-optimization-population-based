"""Generate the project's canonical dataset.

Mirrors the spec used across the repository: a synthetic binary-classification
problem of 1250 samples / 20 features, split into 1000 training and 250
validation rows, written as ``data/train.csv`` and ``data/validation.csv`` with
a ``target`` column.

Run with:  uv run python examples/generate_data.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def main() -> None:
    X, y = make_classification(
        n_samples=1250,  # 1000 train + 250 validation
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    train_df = df.iloc[:1000]
    validation_df = df.iloc[1000:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "validation.csv"
    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(val_path, index=False)

    print(f"Wrote {len(train_df)} rows -> {train_path}")
    print(f"Wrote {len(validation_df)} rows -> {val_path}")


if __name__ == "__main__":
    main()
