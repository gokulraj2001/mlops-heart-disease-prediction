import numpy as np

from src import config, data
from src.preprocessing import add_hr_reserve, build_preprocessor


def test_output_has_no_nans_even_with_missing_input(clean_df):
    X, _ = data.split_xy(clean_df)
    assert X.isna().any().any()
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()


def test_numeric_columns_are_standardised(clean_df):
    X, _ = data.split_xy(clean_df)
    prep = build_preprocessor().fit(X)
    cols = config.NUMERIC_FEATURES + config.DERIVED_FEATURES
    num = (prep.named_steps["columns"].named_transformers_["num"]
           .transform(add_hr_reserve(X)[cols]))
    assert np.allclose(num.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(num.std(axis=0), 1, atol=1e-6)


def test_unseen_category_does_not_crash(clean_df):
    X, _ = data.split_xy(clean_df)
    prep = build_preprocessor().fit(X)
    odd = X.head(2).copy()
    odd["cp"] = 99.0
    assert prep.transform(odd).shape[0] == 2


def test_imputer_uses_training_statistics_only(clean_df):
    X, _ = data.split_xy(clean_df)
    prep = build_preprocessor().fit(X.iloc[:100])
    expected = X.iloc[:100]["chol"].median()
    imp = prep.named_steps["columns"].named_transformers_["num"].named_steps["impute"]
    idx = config.NUMERIC_FEATURES.index("chol")
    assert imp.statistics_[idx] == expected


def test_hr_reserve_formula_and_no_mutation(clean_df):
    X, _ = data.split_xy(clean_df)
    before = X.copy()
    out = add_hr_reserve(X)
    assert (out["hr_reserve"] == 220.0 - X["age"] - X["thalach"]).all()
    assert "hr_reserve" not in X.columns
    assert X.equals(before)


def test_hr_reserve_known_value():
    import pandas as pd
    row = pd.DataFrame({"age": [60.0], "thalach": [150.0]})
    assert add_hr_reserve(row).loc[0, "hr_reserve"] == 10.0


def test_feature_switch_changes_output_width(clean_df):
    X, _ = data.split_xy(clean_df)
    with_hrr = build_preprocessor(True).fit_transform(X).shape[1]
    without = build_preprocessor(False).fit_transform(X).shape[1]
    assert with_hrr == without + 1


def test_hr_reserve_handles_missing_inputs(clean_df):
    X, _ = data.split_xy(clean_df)
    X = X.copy()
    X.loc[X.index[0], "thalach"] = np.nan
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()
