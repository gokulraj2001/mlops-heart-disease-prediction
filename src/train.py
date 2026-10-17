"""Train, tune, evaluate and track candidate models with MLflow.

Usage:
    python -m src.train                 # full grids
    python -m src.train --quick         # small grids (used in CI)

For every model family we run GridSearchCV (stratified k-fold, scoring =
ROC-AUC). Each tried hyper-parameter combination is logged as a nested MLflow
run; the tuned best model gets its own run with cross-validated metrics,
hold-out test metrics, ROC / confusion-matrix plots and the fitted pipeline.
The family with the best CV ROC-AUC is saved to models/model.joblib.
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402

from src import config, data  # noqa: E402
from src.preprocessing import build_preprocessor  # noqa: E402

CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def model_spaces(quick: bool) -> dict:
    """Candidate estimators and their hyper-parameter grids."""
    lr_grid = {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]}
    rf_grid = {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [3, 5, None],
        "clf__min_samples_leaf": [1, 3],
    }
    gb_grid = {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.03, 0.1],
        "clf__max_depth": [2, 3],
        "clf__subsample": [0.8, 1.0],
    }
    if quick:
        lr_grid = {"clf__C": [0.1, 1]}
        rf_grid = {"clf__n_estimators": [50], "clf__max_depth": [3, None]}
        gb_grid = {"clf__n_estimators": [50], "clf__max_depth": [2, 3]}
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
            lr_grid),
        "random_forest": (
            RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
            rf_grid),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=config.RANDOM_STATE),
            gb_grid),
    }


def test_metrics(pipe: Pipeline, X, y) -> dict:
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    return {
        "test_accuracy": accuracy_score(y, pred),
        "test_precision": precision_score(y, pred, zero_division=0),
        "test_recall": recall_score(y, pred, zero_division=0),
        "test_f1": f1_score(y, pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y, proba),
    }


def log_plots(name: str, pipe: Pipeline, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title(f"ROC curve - {name} (test set)")
    mlflow.log_figure(fig, f"plots/roc_{name}.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipe, X_test, y_test, ax=ax, cmap="Blues",
        display_labels=["No disease", "Disease"])
    ax.set_title(f"Confusion matrix - {name}")
    mlflow.log_figure(fig, f"plots/confusion_{name}.png")
    plt.close(fig)


def tune_and_track(name, estimator, grid, X_train, X_test, y_train, y_test, cv,
                   include_hrr=True):
    pipe = Pipeline([("prep", build_preprocessor(include_hrr)), ("clf", estimator)])
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, refit=True,
                          n_jobs=1, return_train_score=False)
    search.fit(X_train, y_train)

    # One nested run per candidate hyper-parameter combination
    res = search.cv_results_
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"{name}_cand_{i:02d}", nested=True):
            mlflow.set_tag("stage", "tuning_candidate")
            mlflow.log_params({k.replace("clf__", ""): str(v) for k, v in params.items()})
            mlflow.log_metric("cv_roc_auc_mean", float(res["mean_test_score"][i]))
            mlflow.log_metric("cv_roc_auc_std", float(res["std_test_score"][i]))
            mlflow.log_metric("rank", int(res["rank_test_score"][i]))

    best = search.best_estimator_
    # Honest CV estimate for the winning configuration (all metrics)
    cvres = cross_validate(clone(best), X_train, y_train, cv=cv,
                           scoring=CV_SCORING, n_jobs=1)
    cv_metrics = {}
    for m in CV_SCORING:
        cv_metrics[f"cv_{m}_mean"] = float(cvres[f"test_{m}"].mean())
        cv_metrics[f"cv_{m}_std"] = float(cvres[f"test_{m}"].std())
    t_metrics = test_metrics(best, X_test, y_test)

    with mlflow.start_run(run_name=f"{name}_best", nested=True):
        mlflow.set_tag("stage", "best_per_family")
        mlflow.set_tag("model_family", name)
        mlflow.log_params({k.replace("clf__", ""): str(v)
                           for k, v in search.best_params_.items()})
        mlflow.log_param("cv_folds", cv.get_n_splits())
        mlflow.log_metrics({**cv_metrics, **t_metrics})
        log_plots(name, best, X_test, y_test)
        sig = infer_signature(X_test, best.predict_proba(X_test)[:, 1])
        mlflow.sklearn.log_model(best, artifact_path="model", signature=sig,
                                 input_example=X_test.head(3))
    return best, cv_metrics, t_metrics, search.best_params_


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small grids (CI)")
    ap.add_argument("--data", default=None, help="path to raw or cleaned CSV")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--experiment", default="heart-disease-risk")
    ap.add_argument("--model-dir", default=str(config.MODEL_DIR))
    ap.add_argument("--without-hr-reserve", action="store_true",
                    help="baseline run without the engineered hr_reserve feature")
    args = ap.parse_args()

    if args.data:
        df = pd.read_csv(args.data)
        if config.TARGET not in df.columns:
            df = data.clean(df)
    else:
        df = data.clean(data.load_raw())
    X, y = data.split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=config.RANDOM_STATE)
    cv = StratifiedKFold(args.cv_folds, shuffle=True, random_state=config.RANDOM_STATE)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI",
                                      f"file:{config.ROOT / 'mlruns'}"))
    mlflow.set_experiment(args.experiment)

    results = {}
    with mlflow.start_run(run_name="model_selection"):
        mlflow.log_params({
            "n_rows": len(df), "n_features": X.shape[1],
            "test_size": args.test_size, "cv_folds": args.cv_folds,
            "random_state": config.RANDOM_STATE, "quick_mode": args.quick,
            "positive_rate": round(float(y.mean()), 4),
            "hr_reserve_feature": not args.without_hr_reserve,
        })
        for name, (est, grid) in model_spaces(args.quick).items():
            print(f"\n=== Tuning {name} ===")
            results[name] = tune_and_track(name, est, grid, X_train, X_test,
                                           y_train, y_test, cv,
                                           include_hrr=not args.without_hr_reserve)
            _, cvm, tm, bp = results[name]
            print(f"best params: {bp}")
            print(f"CV ROC-AUC {cvm['cv_roc_auc_mean']:.3f} "
                  f"| test ROC-AUC {tm['test_roc_auc']:.3f}")

        winner = max(results, key=lambda k: results[k][1]["cv_roc_auc_mean"])
        best_pipe, cvm, tm, bp = results[winner]
        mlflow.set_tag("selected_model", winner)
        mlflow.log_metric("selected_cv_roc_auc", cvm["cv_roc_auc_mean"])
        mlflow.log_metric("selected_test_roc_auc", tm["test_roc_auc"])

        os.makedirs(args.model_dir, exist_ok=True)
        model_path = os.path.join(args.model_dir, "model.joblib")
        joblib.dump(best_pipe, model_path)
        summary = {
            "selected_model": winner,
            "best_params": {k: str(v) for k, v in bp.items()},
            "features": config.FEATURES,
            "derived_features": ([] if args.without_hr_reserve
                                 else config.DERIVED_FEATURES),
            "cv_metrics": cvm, "test_metrics": tm,
            "all_models": {k: {"cv_roc_auc_mean": v[1]["cv_roc_auc_mean"],
                               "test_roc_auc": v[2]["test_roc_auc"]}
                           for k, v in results.items()},
        }
        metrics_path = os.path.join(args.model_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(model_path, "final_model")
        mlflow.log_artifact(metrics_path, "final_model")
        print(f"\nSelected: {winner} -> {model_path}")


if __name__ == "__main__":
    main()
