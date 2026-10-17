import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src import data  # noqa: E402
from src.preprocessing import build_preprocessor  # noqa: E402
from tests.synthetic import make_raw  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    return make_raw(300, seed=1)


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return data.clean(raw_df)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, clean_df):
    """A small fitted pipeline saved the same way train.py saves it."""
    import joblib
    import json
    X, y = data.split_xy(clean_df)
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=1000))]).fit(X, y)
    d = tmp_path_factory.mktemp("model")
    joblib.dump(pipe, d / "model.joblib")
    (d / "metrics.json").write_text(json.dumps({"selected_model": "test_lr"}))
    return d
