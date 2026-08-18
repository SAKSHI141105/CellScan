"""Run once to dump the standard EDA figures into reports/figures/.

Not wired into the dashboard — this is a one-off exploration script, the
kind you'd actually run from a notebook first and then port over once you
know what you want. Kept as a .py so it's diffable and CI-friendly.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_preprocessing.tabular_preprocessing import basic_quality_report, clean_tabular, load_raw_tabular
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid", palette="deep")


def main():
    cfg = load_config()
    fig_dir = PROJECT_ROOT / cfg["paths"]["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_tabular()
    report = basic_quality_report(df)
    logger.info("Quality report: %s", report)
    df = clean_tabular(df)

    # class balance — this is the number that justifies the whole SMOTE section later
    counts = df["diagnosis"].value_counts().rename({0: "Benign", 1: "Malignant"})
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", color=["#2a9d8f", "#e76f51"], ax=ax)
    ax.set_title(f"Class distribution (n={len(df)})")
    ax.set_ylabel("count")
    for i, v in enumerate(counts):
        ax.text(i, v + 3, f"{v} ({v/len(df):.0%})", ha="center")
    fig.tight_layout()
    fig.savefig(fig_dir / "tabular_class_distribution.png", dpi=150)
    plt.close(fig)

    # correlation heatmap — 30 features is already borderline unreadable, so we
    # only keep the "mean" family here and let feature_selection.py do the real work
    mean_cols = [c for c in df.columns if c.endswith("_mean")] + ["diagnosis"]
    corr = df[mean_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Correlation — mean-family features")
    fig.tight_layout()
    fig.savefig(fig_dir / "tabular_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # outliers on a few of the features that end up mattering most (see feature_selection output)
    key_feats = ["radius_mean", "concavity_mean", "texture_mean", "area_mean"]
    fig, axes = plt.subplots(1, len(key_feats), figsize=(4 * len(key_feats), 4))
    for ax, feat in zip(axes, key_feats):
        sns.boxplot(x="diagnosis", y=feat, hue="diagnosis", data=df, ax=ax, palette=["#2a9d8f", "#e76f51"], legend=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Benign", "Malignant"])
    fig.tight_layout()
    fig.savefig(fig_dir / "tabular_outliers_boxplots.png", dpi=150)
    plt.close(fig)

    logger.info("Figures written to %s", fig_dir)


if __name__ == "__main__":
    main()
