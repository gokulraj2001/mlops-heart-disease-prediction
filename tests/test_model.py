import json

import joblib
import mlflow
import pytest

from src import config, data, train


FAMILIES = {"logistic_regression", "random_forest", "gradient_boosting"}


def test_model_spaces_contain_three_families():
    assert set(train.model_spaces(quick=True)) == FAMILIES
    assert set(train.model_spaces(quick=False)) == FAMILIES
    full = train.model_spaces(quick=False)
    assert len(full["random_forest"][1]["clf__max_depth"]) >= 2


def test_saved_pipeline_predicts_probabilities(model_dir, clean_df):
    pipe = joblib.load(model_dir / "model.joblib")
    X, _ = data.split_xy(clean_df)
    proba = pipe.predict_proba(X.head(5))
    assert proba.shape == (5, 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_train_end_to_end_quick(tmp_path, raw_df, monkeypatch):
    csv = tmp_path / "raw.csv"
    raw_df.to_csv(csv, index=False)
    out = tmp_path / "models"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    monkeypatch.setattr("sys.argv", [
        "train", "--quick", "--data", str(csv), "--model-dir", str(out),
        "--cv-folds", "3", "--experiment", "pytest-run"])
    train.main()

    assert (out / "model.joblib").exists()
    summary = json.loads((out / "metrics.json").read_text())
    assert summary["selected_model"] in FAMILIES
    assert set(summary["all_models"]) == FAMILIES
    assert 0.5 < summary["test_metrics"]["test_roc_auc"] <= 1.0
    assert summary["features"] == config.FEATURES

    # MLflow actually recorded runs, params and metrics
    exp = mlflow.get_experiment_by_name("pytest-run")
    runs = mlflow.search_runs([exp.experiment_id])
    assert len(runs) >= 7  # parent + 3 best + candidates
    best = runs[runs["tags.stage"] == "best_per_family"]
    assert len(best) == 3
    assert best["metrics.cv_roc_auc_mean"].notna().all()


@pytest.mark.parametrize("metric", ["accuracy", "precision", "recall", "roc_auc"])
def test_required_metrics_are_configured(metric):
    assert metric in train.CV_SCORING


def test_baseline_flag_runs_without_hr_reserve(tmp_path, raw_df, monkeypatch):
    csv = tmp_path / "raw.csv"
    raw_df.to_csv(csv, index=False)
    out = tmp_path / "base"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    monkeypatch.setattr("sys.argv", [
        "train", "--quick", "--without-hr-reserve", "--data", str(csv),
        "--model-dir", str(out), "--cv-folds", "3", "--experiment", "pytest-base"])
    train.main()
    summary = json.loads((out / "metrics.json").read_text())
    assert summary["derived_features"] == []
    pipe = joblib.load(out / "model.joblib")
    assert "hr_reserve" not in pipe.named_steps["prep"].named_steps
