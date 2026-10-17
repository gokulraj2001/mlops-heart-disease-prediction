"""Feature engineering + preprocessing shared by training and serving."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config


def add_hr_reserve(X: pd.DataFrame) -> pd.DataFrame:
    """Add heart-rate reserve: (220 - age) - thalach.

    220 - age is the common estimate of a person's maximum heart rate. A small
    (or negative) reserve means the heart reached its predicted ceiling during
    the stress test, which is clinically more informative than thalach alone
    because the same thalach means different things at age 35 vs age 70.
    The input frame is not modified.
    """
    out = X.copy()
    out["hr_reserve"] = 220.0 - out["age"] - out["thalach"]
    return out


def build_preprocessor(include_hr_reserve: bool = True) -> Pipeline:
    numeric_cols = list(config.NUMERIC_FEATURES)
    steps = []
    if include_hr_reserve:
        steps.append(("hr_reserve", FunctionTransformer(add_hr_reserve)))
        numeric_cols += config.DERIVED_FEATURES

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    steps.append(("columns", ColumnTransformer([
        ("num", numeric, numeric_cols),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])))
    return Pipeline(steps)
