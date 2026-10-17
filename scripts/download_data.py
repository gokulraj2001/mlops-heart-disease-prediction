"""Download the UCI Heart Disease (Cleveland) dataset.

Usage:
    python scripts/download_data.py     # writes data/raw/heart_disease_raw.csv

Strategy: try the official `ucimlrepo` client first (dataset id 45); if that
fails (network / API change), fall back to the raw Cleveland file on the UCI
archive. Both paths produce the same 14-column CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "heart_disease_raw.csv"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
FALLBACK_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


def via_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=45)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    # the target column is called "num" in the repo; keep that name
    return df


def via_raw_file() -> pd.DataFrame:
    return pd.read_csv(FALLBACK_URL, header=None, names=COLUMNS, na_values="?")


def main() -> int:
    df = None
    for name, fn in (("ucimlrepo", via_ucimlrepo), ("raw UCI file", via_raw_file)):
        try:
            df = fn()
            print(f"Downloaded via {name}: {df.shape[0]} rows x {df.shape[1]} cols")
            break
        except Exception as exc:  # noqa: BLE001 - try the next source
            print(f"[warn] {name} failed: {exc}", file=sys.stderr)
    if df is None:
        print("All download methods failed. Manually download "
              "'processed.cleveland.data' from "
              "https://archive.ics.uci.edu/dataset/45/heart+disease and save "
              f"it with a header row to {OUT_PATH}", file=sys.stderr)
        return 1

    missing = set(COLUMNS) - set(df.columns)
    if missing:
        print(f"Unexpected schema, missing columns: {missing}", file=sys.stderr)
        return 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
