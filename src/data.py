"""Loading and light cleaning of the raw heart-disease CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config


def load_raw(path: Path | str = config.RAW_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy with a binary `target` column.

    - drops exact duplicate rows
    - coerces every column to numeric ('?' and blanks become NaN)
    - binarises the 0-4 severity label: 0 -> 0 (healthy), 1-4 -> 1 (disease)

    Missing values are deliberately NOT imputed here: imputation lives inside
    the sklearn Pipeline so it is fitted on training folds only (no leakage)
    and is applied identically at serving time.
    """
    out = df.copy().drop_duplicates().reset_index(drop=True)
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if config.TARGET_RAW not in out.columns:
        raise KeyError(f"Expected raw target column '{config.TARGET_RAW}'")
    out = out.dropna(subset=[config.TARGET_RAW])
    out[config.TARGET] = (out[config.TARGET_RAW] > 0).astype(int)
    return out.drop(columns=[config.TARGET_RAW])


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features as float64 (so missing values are representable at serving
    time and the MLflow model signature matches the API schema)."""
    return df[config.FEATURES].astype("float64"), df[config.TARGET]


def save_processed(df: pd.DataFrame, path: Path | str = config.PROCESSED_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
