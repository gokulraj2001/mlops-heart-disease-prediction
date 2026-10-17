"""Synthetic stand-in for the UCI Cleveland data (same schema, fake values).

Used ONLY for offline unit tests / sandbox smoke tests, never for reporting
results. Real results must come from the real dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_raw(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(29, 78, n)
    sex = rng.integers(0, 2, n)
    cp = rng.integers(1, 5, n)
    thalach = rng.normal(150, 22, n).clip(70, 200).round()
    oldpeak = rng.gamma(1.2, 0.9, n).round(1)
    exang = rng.integers(0, 2, n)
    ca = rng.integers(0, 4, n).astype(float)
    thal = rng.choice([3.0, 6.0, 7.0], n, p=[0.55, 0.05, 0.40])
    risk = (0.04 * (age - 54) + 0.8 * (cp == 4) + 0.7 * exang + 0.6 * oldpeak
            + 0.7 * ca - 0.02 * (thalach - 150) + 0.9 * (thal == 7)
            + rng.normal(0, 0.8, n))
    num = np.where(risk > 2.6, rng.integers(1, 5, n), 0)
    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp,
        "trestbps": rng.normal(131, 17, n).round(),
        "chol": rng.normal(246, 51, n).round(),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.choice([0, 1, 2], n, p=[0.5, 0.01, 0.49]),
        "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": rng.integers(1, 4, n), "ca": ca, "thal": thal, "num": num,
    })
    # sprinkle a few missing values like the real ca / thal columns
    df.loc[rng.choice(n, 4, replace=False), "ca"] = np.nan
    df.loc[rng.choice(n, 2, replace=False), "thal"] = np.nan
    return df
