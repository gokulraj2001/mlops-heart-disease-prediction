import numpy as np
import pandas as pd
import pytest

from src import config, data


def test_clean_binarises_target(raw_df):
    out = data.clean(raw_df)
    assert set(out[config.TARGET].unique()) <= {0, 1}
    assert config.TARGET_RAW not in out.columns
    # severity 1-4 must all map to 1
    n_pos = (raw_df[config.TARGET_RAW] > 0).sum()
    assert out[config.TARGET].sum() == n_pos


def test_clean_coerces_question_marks_to_nan():
    df = pd.DataFrame({"age": [50, 60], "ca": ["?", "1"], "num": [0, 2]})
    out = data.clean(df)
    assert np.isnan(out.loc[0, "ca"]) and out.loc[1, "ca"] == 1.0


def test_clean_drops_duplicates():
    row = {"age": 50, "num": 0}
    out = data.clean(pd.DataFrame([row, row, {"age": 51, "num": 1}]))
    assert len(out) == 2


def test_clean_requires_target_column():
    with pytest.raises(KeyError):
        data.clean(pd.DataFrame({"age": [1, 2]}))


def test_clean_keeps_missing_for_pipeline_to_impute(clean_df):
    assert clean_df[["ca", "thal"]].isna().sum().sum() > 0


def test_split_xy_shapes_and_dtype(clean_df):
    X, y = data.split_xy(clean_df)
    assert list(X.columns) == config.FEATURES
    assert len(X) == len(y)
    assert (X.dtypes == "float64").all()


def test_load_raw_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        data.load_raw(tmp_path / "nope.csv")


def test_save_processed_roundtrip(clean_df, tmp_path):
    p = data.save_processed(clean_df, tmp_path / "sub" / "c.csv")
    assert pd.read_csv(p).shape == clean_df.shape
