"""Exploratory data analysis: cleans the raw CSV and writes figures.

Usage: python -m src.eda
Outputs: data/processed/heart_disease_clean.csv and PNGs in reports_out/eda/
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config, data  # noqa: E402
from src.preprocessing import add_hr_reserve  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
OUT = config.PLOTS_DIR / "eda"


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def plot_missing(df) -> None:
    miss = df[config.FEATURES].isna().sum()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(5, 3))
    if miss.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.axis("off")
    else:
        miss.plot.bar(ax=ax, color="#c0392b")
        ax.set_ylabel("Missing count")
        ax.set_title("Missing values per feature")
    _save(fig, "missing_values.png")


def plot_class_balance(df) -> None:
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars = ax.bar(["No disease (0)", "Disease (1)"], counts.values,
                  color=["#2e86c1", "#c0392b"])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1,
                f"{v} ({v / counts.sum():.1%})", ha="center")
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    _save(fig, "class_balance.png")


def plot_histograms(df) -> None:
    cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(data=df, x=col, hue=config.TARGET, kde=True,
                     palette=["#2e86c1", "#c0392b"], ax=ax, element="step")
        ax.set_title(f"Distribution of {col}")
    axes.ravel()[-1].axis("off")
    _save(fig, "numeric_histograms.png")


def plot_boxplots(df) -> None:
    cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(1, len(cols), figsize=(15, 3.8))
    for ax, col in zip(axes, cols):
        sns.boxplot(data=df, x=config.TARGET, y=col, ax=ax,
                    palette=["#2e86c1", "#c0392b"], hue=config.TARGET, legend=False)
    _save(fig, "numeric_boxplots_by_target.png")


def plot_categorical(df) -> None:
    cols = config.CATEGORICAL_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), cols):
        sns.countplot(data=df, x=col, hue=config.TARGET, ax=ax,
                      palette=["#2e86c1", "#c0392b"])
        ax.set_title(f"{col} by target")
        ax.legend(title="target", fontsize=8)
    _save(fig, "categorical_counts_by_target.png")


def plot_hr_reserve(df) -> None:
    d = add_hr_reserve(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(data=d, x="hr_reserve", hue=config.TARGET, kde=True, element="step",
                 palette=["#2e86c1", "#c0392b"], ax=axes[0])
    axes[0].set_title("Heart-rate reserve = (220 - age) - thalach")
    sns.boxplot(data=d, x=config.TARGET, y="hr_reserve", hue=config.TARGET,
                palette=["#2e86c1", "#c0392b"], legend=False, ax=axes[1])
    r_new = d["hr_reserve"].corr(d[config.TARGET])
    r_old = d["thalach"].corr(d[config.TARGET])
    axes[1].set_title(f"corr with target: hr_reserve {r_new:.2f} vs thalach {r_old:.2f}")
    _save(fig, "hr_reserve.png")
    print(f"corr(target): hr_reserve={r_new:.3f}  thalach={r_old:.3f}")


def plot_correlation(df) -> None:
    df = add_hr_reserve(df)
    corr = df[config.FEATURES + config.DERIVED_FEATURES + [config.TARGET]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = None
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                mask=mask, square=True, cbar_kws={"shrink": 0.8}, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Correlation heatmap (Pearson)")
    _save(fig, "correlation_heatmap.png")


def main() -> None:
    raw = data.load_raw()
    df = data.clean(raw)
    path = data.save_processed(df)
    print(f"Rows: {len(df)} | Features: {len(config.FEATURES)}")
    print("Missing values:\n", df[config.FEATURES].isna().sum()[lambda s: s > 0])
    print("Class balance:\n", df[config.TARGET].value_counts(normalize=True).round(3))
    print(f"Saved cleaned data -> {path}")
    for fn in (plot_missing, plot_class_balance, plot_histograms, plot_boxplots,
               plot_categorical, plot_hr_reserve, plot_correlation):
        fn(df)
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
